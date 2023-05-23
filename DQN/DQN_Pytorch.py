import sys
import gym
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import wrappers

if len(sys.argv) != 3:
    raise Exception('exp_idx and gpu number. e.g.) python DQN_Pytorch.py 999 0')
exp_idx = sys.argv[1]
gpu_num = sys.argv[2]

print('exp_idx: {}'.format(exp_idx))

random.seed(int(exp_idx))
random_seed = random.randint(0, 10000)

print('Random Seed:', random_seed)

torch.manual_seed(random_seed)
np.random.seed(random_seed)

envCartPole = gym.make('CartPole-v1')
envCartPole.seed(random_seed)
envCartPole.action_space.seed(random_seed)

EPISODES = 1000

def discount_rate(): #Gamma
    return 0.95

def learning_rate(): #Alpha
    return 0.001

def batch_size():
    return 64

class Q_Model(nn.Module):
    def __init__(self, nS, nA):
        super(Q_Model, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(nS, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, nA)
        )
    
    def forward(self, x):
        pred = self.model(x)
        return pred
        
class DeepQNetwork():
    def __init__(self, states, actions, alpha, gamma, epsilon,epsilon_min, epsilon_decay, device='cuda:0'):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.gamma = gamma
        #Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.model = Q_Model(self.nS, self.nA).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = alpha)
        self.loss = []
        self.q_value_estimate_each_period = []

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA) #Explore
        action_vals = self.model(torch.Tensor(state).to(self.device)).detach().cpu().numpy() #Exploit: Use the NN to predict the correct action from this state
        return np.argmax(action_vals[0])

    def test_action(self, state): #Exploit
        action_vals = self.model(torch.Tensor(state).to(self.device)).detach().cpu().numpy()
        return np.argmax(action_vals[0])

    def store(self, state, action, reward, nstate, done):
        #Store the experience in memory
        self.memory.append( (state, action, reward, nstate, done) )

    def experience_replay(self, batch_size):
        #Execute the experience replay
        minibatch = random.sample( self.memory, batch_size ) #Randomly sample from memory

        #Convert to numpy for speed by vectorization
        x = []
        y = []
        np_array = np.array(minibatch)
        st = np.zeros((0,self.nS)) #States
        nst = np.zeros( (0,self.nS) )#Next States
        for i in range(len(np_array)): #Creating the state and next state np arrays
            st = np.append( st, np_array[i,0], axis=0)
            nst = np.append( nst, np_array[i,3], axis=0)
        st_predict = self.model(torch.Tensor(st).to(self.device)).detach().cpu().numpy() #Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model(torch.Tensor(nst).to(self.device)).detach().cpu().numpy()
        index = 0
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            #Predict from state
            nst_action_predict_model = nst_predict[index]
            if done == True: #Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:   #Non terminal
                target = reward + self.gamma * np.amax(nst_action_predict_model)
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        #Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size,self.nS)
        y_reshape = torch.Tensor(np.array(y)).to(self.device)
        epoch_count = 1
        
        pred = self.model(torch.Tensor(x_reshape).to(self.device))
        loss = self.loss_fn(pred, y_reshape)
        self.optimizer.zero_grad()
        loss.backward()
        self.loss.append(loss.item())
        self.optimizer.step()
        
        #Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def estimate_q_value_from_replay_buffer(self, estimation_size):
        #Execute the experience replay
        minibatch = random.sample(self.memory, estimation_size ) #Randomly sample from memory

        #Convert to numpy for speed by vectorization
        x = []
        y = []
        np_array = np.array(minibatch)
        st = np.zeros((0,self.nS)) #States
        for i in range(len(np_array)): #Creating the state and next state np arrays
            st = np.append( st, np_array[i,0], axis=0)
        st_action_vals = self.model(torch.Tensor(st).to(self.device)).detach().cpu().numpy() # action_vals
        st_action_max_vals = np.max(st_action_vals, axis=1)
        
        self.q_value_estimate_each_period.append(np.mean(st_action_max_vals))

#Create the agents
nS = envCartPole.observation_space.shape[0] #This is only 4
nA = envCartPole.action_space.n #Actions
dqn = DeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.001, 0.995, 'cuda:{}'.format(gpu_num))

batch_size = batch_size()

#Training
estimation_size = batch_size

rewards = [] #Store rewards for graphing
epsilons = [] # Store the Explore/Exploit
for e in tqdm(range(EPISODES)): #*****#
    state = envCartPole.reset()
    state = np.reshape(state, [1, nS]) # Resize to store in memory to pass to .predict
    tot_rewards = 0
    for time in range(210): #200 is when you "solve" the game. This can continue forever as far as I know
        
        #################################
        ##### Do Q value estimation #####
        
        if len(dqn.memory) > estimation_size:
            dqn.estimate_q_value_from_replay_buffer(estimation_size)
        
        #################################
        #################################
        
        action = dqn.action(state)
        nstate, reward, done, _= envCartPole.step(action)
        nstate = np.reshape(nstate, [1, nS])
        tot_rewards += reward
        dqn.store(state, action, reward, nstate, done) # Resize to store in memory to pass to .predict
        state = nstate
            
        #Experience Replay
        if len(dqn.memory) > batch_size:
            dqn.experience_replay(batch_size)
                
        #done: CartPole fell. 
        #time == 209: CartPole stayed upright
        if done or time == 209:
            rewards.append(tot_rewards)
            epsilons.append(dqn.epsilon)
            break # stop the episode

plt.plot(dqn.q_value_estimate_each_period)
plt.savefig('./results/exp_{}_Q_value_estimates_per_step.png'.format(exp_idx), dpi=300)
plt.close()
q_value_estimate_each_period_np_array = np.array(dqn.q_value_estimate_each_period)
np.save('./results/exp_{}_q_value_estimate_each_period.npy'.format(exp_idx), q_value_estimate_each_period_np_array)

plt.plot(rewards)
plt.savefig('./results/exp_{}_Reward_per_episode.png'.format(exp_idx), dpi=300)
plt.close()
rewards_np_array = np.array(rewards)
np.save('./results/exp_{}_rewards.npy'.format(exp_idx), rewards_np_array)

#Testing
TEST_Episodes = 5
print('Training complete. Testing started...')
#TEST Time
#   In this section we ALWAYS use exploit don't train any more
for e_test in range(TEST_Episodes):
    state = envCartPole.reset()
    state = np.reshape(state, [1, nS])
    tot_rewards = 0
    for t_test in range(210):
        action = dqn.test_action(state)
        nstate, reward, done, _ = envCartPole.step(action)
        nstate = np.reshape(nstate, [1, nS])
        tot_rewards += reward
        #DON'T STORE ANYTHING DURING TESTING
        state = nstate
        #done: CartPole fell. 
        #t_test == 209: CartPole stayed upright
        if done or t_test == 209: 
            rewards.append(tot_rewards)
            epsilons.append(0) #We are doing full exploit
            print("episode: {}/{}, score: {}, e: {}"
                  .format(e_test, TEST_Episodes, tot_rewards, 0))
            break
