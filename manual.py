# modified from: https://github.com/gui-miotto/DeepLearningLab/blob/master/Assignment%2003/Code/drive_manually.py

from __future__ import print_function
from pyglet.window import key
import gym
import numpy as np
import pickle
import os
from datetime import datetime
import gzip
import json

import copy



def key_press(k, mod):
    global restart
    #if k == 0xff0d: restart = True
    if k == key.ESCAPE: restart = True
    if k == key.UP:    
        a[1] = +1.0
    if k == key.LEFT:  
        a[0] = -1.0
    if k == key.RIGHT: 
        a[0] = +1.0
    if k == key.DOWN:  
        a[2] = +0.8

def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0: 
        a[0] = 0.0
    if k == key.RIGHT and a[0] == +1.0: 
        a[0] = 0.0
    if k == key.UP:    
        a[1] = 0.0
    if k == key.DOWN:  
        a[2] = 0.0


def save_data(data, datasets_dir="./data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)


def save_results(episode_rewards):
    # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    #results["std_all_episodes"] = np.array(episode_rewards).std()
 
    fname = "sample results.json"
    fh = open(fname, "w")
    json.dump(results, fh)
    print('save completed')


if __name__ == "__main__":
   
    good_samples = {
        "state": [],
        "reward": [],
        "action": []
    }
    episode_samples = copy.deepcopy(good_samples)
    
    episode_rewards = []

    env = gym.make('CarRacing-v0').unwrapped
    env.reset()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    a = np.zeros(3, dtype=np.float32)
    
    # Episode loop
    while True:
        episode_steps = 0
        episode_samples["state"] = []
        episode_samples["action"] = []
        episode_samples["reward"] = []
        episode_reward = 0
        state = env.reset()
        restart = False
        # State loop
        while True:
            # Save current state        
            episode_samples["state"].append(state) # state has shape (96, 96, 3)
            
            # Change state
            state, r, done, info = env.step(a[:3])
            episode_reward += r

            # Save action and reward taken from previous state
            episode_samples["action"].append(np.array(a[:3]))     # action has shape (1, 3)
            episode_samples["reward"].append(r) # reward not needed for now
            
            episode_steps += 1

            if episode_steps % 1000 == 0 or done:
                print("\nstep {}".format(episode_steps))

            env.render()
            if done or restart: 
                break
        
        if not restart:
            episode_rewards.append(episode_reward)
            
            good_samples["state"].append(episode_samples["state"])
            good_samples["action"].append(episode_samples["action"])
            good_samples["reward"].append(episode_samples["reward"])

            print('saving data and results')
            save_data(good_samples, "./data")
            save_results(episode_rewards)

    env.close()

