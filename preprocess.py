import numpy as np
import gzip
import pickle
from PIL import Image
from skimage.color import rgb2gray
import matplotlib as plt

actions = np.array([
	[ 0.0, 0.0, 0.0 ],  # STRAIGHT
	[ 0.0, 1.0, 0.0 ],  # ACCELERATE
	[ 0.0, 0.0, 0.8 ],  # BRAKE
	[ 1.0, 0.0, 0.0 ],  # RIGHT
	[ 1.0, 1.0, 0.0 ],  # RIGHT_ACCELERATE
	[ 1.0, 0.0, 0.8 ],  # RIGHT_BRAKE
	[-1.0, 0.0, 0.0 ],  # LEFT
	[-1.0, 1.0, 0.0 ],  # LEFT_ACCELERATE
	[-1.0, 0.0, 0.8 ]   # LEFT_BRAKE
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
	
# Should be called in preprocessing, but rn one-hot encode is not used
def balance_actions(X, y, drop_prob):
    """ Balance samples. Gets hide of a share of the most common action (accelerate) """
    # Enconding of the action accelerate
    acceler = np.zeros(9)
    acceler[1] = 1.
    # Find out what samples are labeled as accelerate
    is_accel = np.all(y==acceler, axis=1)
    # Get the index of all other samples (not accelerate)
    other_actions_index = np.where(np.logical_not(is_accel))
    # Randomly pick drop some accelerate samples. Probabiliy of dropping is given by drop_prob
    drop_mask = np.random.rand(len(is_accel)) > drop_prob
    accel_keep = drop_mask * is_accel
    # Get the index of accelerate samples that were kept
    accel_keep_index = np.where(accel_keep)
    # Put all actions that we want to keep together
    final_keep = np.squeeze(np.hstack((other_actions_index, accel_keep_index)))
    final_keep = np.sort(final_keep)
    X_bal, y_bal = X[final_keep], y[final_keep]

    return X_bal, y_bal

#TODO
def detect_invalid_actions(actions):
	pass

# Not used rn, the optimizer does the hot encode
def one_hot_encode(action_ids):
	one_hot_labels = np.zeros(action_ids.shape + (n_actions,))
	for c in range(n_actions):
		one_hot_labels[action_ids == c, c] = 1.0
	return(one_hot_labels)
def one_hot_decode(one_hot_labels):
    """ Returns actions in the environment understandable format"""
    ids = np.argmax(one_hot_labels, axis=1)
    return(actions[ids])

def preprocess_actions(In_actions):
    """ Returns actions in id format"""
    #detect_invalid_actions(In_actions) #Need this to make sure no invalid actions present
    ids = []
    for action in In_actions:
        id = np.where(np.all(actions==action, axis=1))
        ids.append(id[0][0])
    return np.array(ids)

def preprocess_data(X,y):
	# Return preprocessed and balanced states with action labels
	X = preprocess_state(X)
	y = preprocess_actions(y)
	#return balance_actions(X,y,0.5)
	return X,y
	
