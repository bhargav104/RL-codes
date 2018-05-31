import numpy as np
import sys
import argparse
import os
import torch.optim as optim
import torch
import torch.nn as nn
import gym
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.memory import Memory
from network.dqn import DQN
from core.dqn_step import update_dqn
from utils.process_env import AtariRescale105x80

parser = argparse.ArgumentParser('Training DQN')

parser.add_argument('--env-name', required=True, type=str, help='Name of environment')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--batch-size', type=int, default=32, help='training batch size')
parser.add_argument('--eps-end', type=float, default=0.1, help='least value of epsilon')
parser.add_argument('--replay-size', type=int, default=100000, help='size of replay buffer')
parser.add_argument('--train-frames', type=int, default=10000000, help='number of training frames')
parser.add_argument('--lr', type=float, default=0.00025, help='learning rate')
parser.add_argument('--past-frames', type=int, default=4, help='number of past frames')
parser.add_argument('--cpu', action='store_false', default=True, help='train on cpu')

tensor = torch.FloatTensor
args = parser.parse_args()

def get_epsilon(iter_num, min_eps):
	value = 1.0 - ((1.0 - min_eps) * iter_num) / 1e6
	return max(value, min_eps)

def train_model(model, env, memory, args, device):

	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	criterion = nn.MSELoss()
	step_no = 0
	state = env.reset()
	state_list = []
	initial = np.zeros((1, 105, 80))
	for i in range(args.past_frames - 1):
		state_list.append(initial[0])
	state_list.append(state[0])
	eps_reward = 0

	while step_no < args.train_frames:
		
		stacked_state = np.stack(state_list, axis=0)
		state_var = tensor(stacked_state).unsqueeze(0).to(device)
		epsilon = get_epsilon(step_no, args.eps_end)

		if random.random() <= epsilon:
			action = env.action_space.sample()
		else:
			action = model.get_best_action(state_var).item()

		next_state, reward, done, _ = env.step(action)
		del state_list[0]
		state_list.append(next_state[0])

		mask = 0 if done else 1
		eps_reward += reward
		reward = max(min(reward, 1), -1)
		memory.push(stacked_state, action, reward, np.stack(state_list, axis=0), mask)

		if done:
			print(step_no, 'epsiode reward- ' + str(eps_reward))
			eps_reward = 0
			state = env.reset()
			state_list = []
			for i in range(args.past_frames - 1):
				state_list.append(initial[0])
			state_list.append(state[0])

		if len(memory) < args.batch_size:
			batch = memory.sample()
		else:
			batch = memory.sample(size=args.batch_size)

		update_dqn(model, batch, args, criterion, optimizer, device)
		step_no += 1

inp_channels = 4
env = gym.make(args.env_name)
env = AtariRescale105x80(env)
num_actions = env.action_space.n
dqn = DQN(inp_channels, num_actions)
memory = Memory(limit=args.replay_size)
device = torch.device('cuda')
dqn.to(device)

train_model(dqn, env, memory, args, device)


