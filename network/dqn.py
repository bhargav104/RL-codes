import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):

	def __init__(self, inp_channels, num_actions):
		super().__init__()
		self.conv1 = nn.Conv2d(inp_channels, 16, 8, stride=4)
		self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
		self.size_out = 32 * 11 * 8
		self.fc1 = nn.Linear(self.size_out, 256)
		self.fc2 = nn.Linear(256, num_actions)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = x.view(-1, self.size_out)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

	def get_best_action(self, x):
		q_vals = self(x).detach()
		return torch.argmax(q_vals, dim=1).unsqueeze(1)

'''
inp = torch.ones(1, 4, 105, 80)
net = DQN(4, 10)
print(net(inp))
'''