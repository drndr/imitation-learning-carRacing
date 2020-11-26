import numpy as np
import gzip
import pickle
from PIL import Image
from skimage.color import rgb2gray
import matplotlib as plt

actions = np.array([
    [ 0.0, 0.0, 0.0],  # STRAIGHT
    [ 0.0, 1.0, 0.0],  # ACCELERATE
    [ 1.0, 0.0, 0.0],  # RIGHT
    [ 1.0, 0.0, 0.8],  # RIGHT_BRAKE
    [ 0.0, 0.0, 0.8],  # BRAKE
    [-1.0, 0.0, 0.8],  # LEFT_BRAKE
    [-1.0, 0.0, 0.0],  # LEFT
	[1.0, 1.0, 0.0],   # RIGHT_ACCELERATE
	[-1.0, 1.0, 0.0]   # LEFT_ACCELERATE
], dtype=np.float32)
n_actions = len(actions)



def read_data():
	with gzip.open('./data/data.pkl.gzip','rb') as f:
		data = pickle.load(f)
	X = join_episodes(data['state'])
	y = join_episodes(data['action'])
	return X,y



def join_episodes(arr,cutoff=50):
	# Stack all  epsiodes in to one array, discard first x fps
	stack = np.array(arr[0][cutoff:],dtype=np.float32)
	for i in range(1,len(arr)):
		stack = np.vstack((stack,arr[i][cutoff:]))
	return stack

def preprocess_state(X):

	X_processed = np.array(X)

	# Normalize rgb channel
	X_processed = X_processed / 255.0

	# Convert to grayscale
	X_processed = rgb2gray(X_processed)

	# Color whole indicator-bar black
	X_processed[:,84:] = 0

	return X_processed

def show_state_as_img(state):
	img = (state*255).astype(np.uint8)
	img = Image.fromarray(img,'L') # L - grayscale RGB - rgb
	img.save('sample.png')
	img.show()

#TODO
def detect_invalid_actions(actions):
	pass

def one_hot_encode(action_ids):
	one_hot_labels = np.zeros(action_ids.shape + (n_actions,))
	for c in range(n_actions):
		one_hot_labels[action_ids == c, c] = 1.0
	return(one_hot_labels)

def one_hot_decode(one_hot_labels):
    return np.argmax(one_hot_labels, axis=1)

def preprocess_actions(In_actions):
	#Convert array format action to Ids
	detect_invalid_actions(In_actions) #Need this to make sure no invalid actions present
	ids = []
	for action in In_actions:
		id = np.where(np.all(actions==action, axis=1))
		ids.append(id[0][0])
	return np.array(ids)

with gzip.open('./data/data.pkl.gzip','rb') as f:
	data = pickle.load(f)

c = join_episodes(data['state'])
c = preprocess_state(c)
show_state_as_img(c[66])
print(c.shape)
processed_actions = preprocess_actions(data['action'][5])
hot_encoded = one_hot_encode(processed_actions)
print(hot_encoded)
unhot_encoded = one_hot_decode(hot_encoded)
print(unhot_encoded)
