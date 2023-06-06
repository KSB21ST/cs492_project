from agents.VanillaDQN import *
import numpy as np


class NMIX(VanillaDQN):
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
      q = self.Q_net_target[0](batch.next_state).clone() # torch.Size([32, 3])
      q_value, q_action = torch.max(q, dim=1)  # torch.Size([32])
      q_value = q_value.reshape(1, -1)
      for i in range(1, self.k):
        _q = self.Q_net_target[i](batch.next_state)
        c_q_value, c_q_action = torch.max(_q, dim=1)  # torch.Size([32])
        q_value = torch.cat((q_value, c_q_value.reshape(1, -1)), dim=0)
      q_next, q_min_idx = torch.min(q_value, dim=0) #torch.Size([32])
      q_target = batch.reward + self.discount * q_next * batch.mask
    return q_target #torch.Size([32])
  
  def compute_q(self, batch):
    # Convert actions to long so they can be used as indexes
    action = batch.action.long().unsqueeze(1)
    q = self.Q_net[self.update_Q_net_index](batch.state).gather(1, action).squeeze()
    return q
  
  def get_action_selection_q_values(self, state):
    q = self.Q_net[0](state).cpu()
    q = to_numpy(q.cpu()).flatten()
    for i in range(1, self.k):
      _q = self.Q_net[i](state)
      _q = to_numpy(_q.cpu()).flatten()
      q = np.vstack((q, _q))
    return q
  
  def get_action(self, mode='Train'):
    '''
    Uses the local Q network and an epsilon greedy policy to pick an action
    PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
    a "fake" dimension to make it a mini-batch rather than a single observation
    '''
    state = to_tensor(self.state[mode], device=self.device)
    state = state.unsqueeze(0) # Add a batch dimension (Batch, Channel, Height, Width)
    q_values = self.get_action_selection_q_values(state) # q_values shape: (self.k, action_space)
    q_min_val, q_min_idx = torch.max(torch.tensor(q_values), dim=1) # q_min_val shape: (self.k)
    val_idx = torch.argmin(q_min_val)
    if mode == 'Test':
      action = q_min_idx[val_idx]
    elif mode == 'Train':
      temp_q_values = np.zeros((q_values.shape[1]),)
      temp_q_values[q_min_idx[val_idx]] = 1
      action = self.exploration.select_action(temp_q_values, self.step_count)
    self.action_probs.append(action)
    return action