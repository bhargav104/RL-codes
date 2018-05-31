import numpy as np
import sys
import torch

tensor = torch.FloatTensor
longt = torch.LongTensor
gpu = torch.device('cuda')

def update_dqn(model, batch, args, criterion, optimizer):
	
	state = tensor(np.stack(batch.state)).to(gpu)
	next_state = tensor(np.stack(batch.next_state)).to(gpu)
	reward = tensor(np.stack(batch.reward)).unsqueeze(1).to(gpu)
	action = longt(np.stack(batch.action)).unsqueeze(1).to(gpu)
	mask = tensor(np.stack(batch.mask)).unsqueeze(1).to(gpu)
	
	target_vals = reward + args.gamma * torch.max(model(next_state).detach(), dim=1)[0].unsqueeze(1)
	target_vals = target_vals * mask
	pred_vals = model(state)
	taken_vals = torch.gather(pred_vals, 1, action)
	loss = criterion(taken_vals, target_vals)
	model.zero_grad()
	loss.backward()
	optimizer.step()
	
