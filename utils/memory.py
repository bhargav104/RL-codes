from collections import namedtuple
import random
import sys
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask'))

class Memory(object):

	def __init__(self, past_frames=1, limit=None):
		self.memory = []
		self.limit = limit
		self.past_frames = past_frames

	def push(self, *args):
		self.memory.append(Transition(*args))
		if self.limit is not None:
			if len(self) > self.limit:
				del self.memory[self.past_frames - 1]

	def sample(self, size=None):

		if size is None:
			return Transition(*zip(*self.memory))
		else:
			batch = random.sample(self.memory, size)
			return Transition(*zip(*batch))

	def __len__(self):
		return len(self.memory)