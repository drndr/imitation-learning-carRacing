
#Pytorch code for linking to env
import numpy as np
import tensorflow as tf
import gym
import preprocess as pp
from gym.wrappers.monitor import Monitor
import torch_model as tm
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES']='-1'

pretrained_model = './car_racing_new_trained.pth'

print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
model = tm.Net().double()
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

#model.summary()

env = gym.make('CarRacing-v0').unwrapped
env.reset()

env_act = np.array([[0,1,0]], dtype=np.float32)
model_state = np.zeros((1,96,96,3))
processed_state = np.zeros((1,96,96,1))
preprocessed_state_torch = torch.zeros((1,96,96,1))


#model_state = np.zeros((96,96,3))
#env = Monitor(env, "video-test2", force=True)

rewards = []
for i in range(10):
	env_act = np.array([[0,1,0]], dtype=np.float32)
	episode_steps = 0
	episode_reward = 0
	anti_freeze = 0;
	restart_gas = 0;
	env.reset()
	while True:
		state, r, done, info = env.step(env_act[0])
		model_state[0] = state
		processed_state = pp.preprocess_state(model_state).reshape((1,96,96))
		#processed_state = processed_state.reshape(processed_state.shape[0],1,96,96)
		preprocessed_state_torch = torch.from_numpy(processed_state).unsqueeze(0)
		output = model(preprocessed_state_torch)
		prediction = output.max(1,keepdim=True)[1]
		if (prediction.item()==0 or prediction.item()==2 or prediction.item()==4):
			anti_freeze+=1
		if anti_freeze>50:
			pred = pp.convert_model2env(1)
			restart_gas+=1
			if restart_gas>3:
				anti_freeze=0
				restart_gas=0
		else:
			pred = pp.convert_model2env(prediction.item())
		env_act = np.array([pred], dtype=np.float32)
		episode_reward += r
		human_exit +=1
		#env.render()
		if done or episode_steps>5000:
			print(episode_reward)
			rewards.append(episode_reward)
			env.close()
			break

print(rewards)
