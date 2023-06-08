from agents.VanillaDQN import *


class NMixDQN(VanillaDQN):
	'''
	Implementation of Maxmin DQN with target network and replay buffer

	We can update all Q_nets for every update. However, this makes training really slow.
	Instead, we randomly choose one to update.
	'''
	def __init__(self, cfg):
		super().__init__(cfg)
		self.k = cfg['agent']['target_networks_num'] # number of target networks
		# Create k different: Q value network, Target Q value network and Optimizer
		self.Q_net = [None] * self.k
		self.Q_net_target = [None] * self.k
		self.optimizer = [None] * self.k
		for i in range(self.k):
			self.Q_net[i] = self.createNN(cfg['env']['input_type']).to(self.device)
			self.Q_net_target[i] = self.createNN(cfg['env']['input_type']).to(self.device)
			self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(), **cfg['optimizer']['kwargs'])
			# Load target Q value network
			self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
			self.Q_net_target[i].eval()

	def learn(self):
		# Choose a Q_net to udpate
		self.update_Q_net_index = np.random.choice(list(range(self.k)))
		super().learn()

	def update_target_net(self):
		if self.step_count % self.cfg['target_network_update_steps'] == 0:
			for i in range(self.k):
				self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())

	def compute_q_target(self, batch):
		with torch.no_grad():
				q_values = self.Q_net_target[0](batch.next_state).clone()
				q_max_min_val, q_max_min_action = q_values.max(1)
				for i in range(1, self.k):
						q_values = self.Q_net_target[i](batch.next_state)
						for i, (q_max_val, q_max_action) in enumerate(zip(q_values.max(1)[0], q_values.max(1)[1])):
							q_max_val, q_max_action=q_max_val.item(), q_max_action.item()
							if q_max_val < q_max_min_val[i].item():
								q_max_min_val[i] = q_max_val
								q_max_min_action[i] = q_max_action

		q_target = batch.reward + self.discount * q_max_min_val * batch.mask
		return q_target

	def get_action(self, mode='Train'):
		'''
		Uses the local Q network and an epsilon greedy policy to pick an action
		PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
		a "fake" dimension to make it a mini-batch rather than a single observation
		'''
		state = to_tensor(self.state[mode], device=self.device)
		state = state.unsqueeze(0) # Add a batch dimension (Batch, Channel, Height, Width)
		
		q_values = self.Q_net[0](state)
		q_max_min_val, q_max_min_action = q_values.max(1)
		for i in range(1, self.k):
			q_values = self.Q_net[i](state)
			for i, (q_max_val, q_max_action) in enumerate(zip(q_values.max(1)[0], q_values.max(1)[1])):
				q_max_val, q_max_action=q_max_val.item(), q_max_action.item()
				if q_max_val < q_max_min_val[i].item():
					q_max_min_val[i] = q_max_val
					q_max_min_action[i] = q_max_action
				
		if mode == 'Test':
			action = q_max_min_action # During test, select best action
		elif mode == 'Train':
			action = self.exploration.select_action(q_values, self.step_count, q_max_min_action)
		return action
