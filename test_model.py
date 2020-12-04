import numpy as np
import tensorflow as tf
import gym
import preprocess as pp
from gym.wrappers.monitor import Monitor


model = tf.keras.models.load_model('models/model2')

model.summary()

env = gym.make('CarRacing-v0').unwrapped
env.reset()


env_act = np.array([[0,1,0]], dtype=np.float32)
print(env_act)
episode_reward = 0
human_exit = 0
model_state = np.zeros((1,96,96,3))
processed_state = np.zeros((1,96,96,1))
env = Monitor(env, "video-test2", force=True)
env.reset()
while True:    
	state, r, done, info = env.step(env_act[0])
	model_state[0] = state
	processed_state[0] = pp.preprocess_state(model_state).reshape((96,96,1))	
	prediction = np.argmax(model.predict(processed_state.T), axis=-1)
	env_act = pp.convert_model2env(prediction)
	episode_reward += r
	human_exit +=1
	#env.render()
	if done or human_exit>1000:
		print(episode_reward) 
		break
	
print(episode_reward)
