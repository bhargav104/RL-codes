import gym
import numpy as np
from gym.spaces.box import Box

class AtariRescale105x80(gym.ObservationWrapper):

	def __init__(self, env):
		super().__init__(env)
		self.observation_space = Box(0.0, 1.0, [1, 105, 80])

	def observation(self, img):
		img = img[::2, ::2]
		img = np.mean(img, axis=2).astype(np.float32)
		return np.expand_dims(img, axis=0) / 255.0