import numpy as np
import gzip
import pickle
from PIL import Image
from skimage.color import rgb2gray
import matplotlib as plt

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

	
with gzip.open('./data/data.pkl.gzip','rb') as f:
	data = pickle.load(f)
	
c = join_episodes(data['state'])
c = preprocess_state(c)
show_state_as_img(c[66])
print(c.shape)
