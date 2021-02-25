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
	[-1.0, 0.0, 0.0 ],  # LEFT
	[ 1.0, 1.0, 0.0 ],  # RIGHT_ACCELERATE
	[-1.0, 1.0, 0.0 ],  # LEFT_ACCELERATE
	[ 1.0, 0.0, 0.8 ],  # RIGHT_BRAKE
	[-1.0, 0.0, 0.8 ]   # LEFT_BRAKE
], dtype=np.float32)

n_actions = len(actions)
	
def read_data():
	with gzip.open('./data/data_19k.pkl.gzip','rb') as f:
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
	
def convert_model2env(id):
	if id==1:
		return [ 0.0, 0.3, 0.0 ]
	return actions[id]

def replace_color(old_color, new_color, X):
	mask = np.all(X == old_color, axis=3)
	X[mask] = new_color

def preprocess_state(X):

	X_processed = np.array(X)

	new_grass_color = [102., 229., 102.]
	replace_color([102., 229., 102.], new_grass_color, X_processed)
	replace_color([102., 204., 102.], new_grass_color, X_processed)

	new_road_color = [102.0, 102.0, 102.0]
	replace_color([102., 102., 102.], new_road_color, X_processed)
	replace_color([105., 105., 105.], new_road_color, X_processed)
	replace_color([107., 107., 107.], new_road_color, X_processed)

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
	


def preprocess_actions(In_actions):
	ids = []
	for action in In_actions:
		id = np.where(np.all(actions==action, axis=1))
		ids.append(id[0][0])
	return np.array(ids)
	
	
def balance_actions(X, y):
	# Find out what action samples are labeled as straight
	is_straight = np.where(y==0)[0]

	# Get the index of all other non straight and non advanced turning (direction+accelerate/brake) action samples
	other_actions = np.where(np.logical_and(y!=0,y<5))[0]

	# Randomly pick a given amount of straight action
	straight_keep = np.random.choice(is_straight,int(len(is_straight)*0.5))

	# Put all actions that we want to keep together
	final_keep = np.squeeze(np.hstack((other_actions, straight_keep)))
	final_keep = np.sort(final_keep)
	X_bal, y_bal = X[final_keep], y[final_keep]

	return X_bal, y_bal

def preprocess_data(X,y):
	# Return preprocessed and balanced states with action labels
	X = preprocess_state(X)
	y = preprocess_actions(y)
	return balance_actions(X,y)
	
