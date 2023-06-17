import gym
from gym import spaces
import numpy as np
from gym.utils import seeding


# Adapted from https://github.com/facebookresearch/RandomizedValueFunctions/blob/master/qlearn/envs/nchain.py
class NChainEnv(gym.Env):
  ''' N-Chain environment
  The environment consists of a chain of N states and the agent always starts in state s2,
  from where it can either move left or right.
  In state s1, the agent receives a small reward of r = 0.001 and a larger reward r = 1 in state sN.
  Check [Deep Exploration via Bootstrapped DQN](https://papers.nips.cc/paper/6501-deep-exploration-via-bootstrapped-dqn.pdf) for a detailed description.
  '''
  def __init__(self, n=10):
    self.state = 2  # Start at state s3 term(s1) - B(s2) - A(s3) - term(s4)
    self.action_space = spaces.Discrete(2)
    self.seed()
    self.init(n)
    self.mu = 0.7
    
  def init(self, n=10):
    self.n = 4
    self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)
    self.max_steps = 2
    self.mu = 0.1
  
  def reward(self, s, a):
    if s == self.n-1 and a==1:
      return 0.0
    elif s==1 and a==0:
      r = self.mu + np.random.uniform(-1,1)
      return r
    elif s==2 and a==0:
      return 0.0
    else:
      return 0

  def step(self, action):
    assert self.action_space.contains(action)
    v = np.arange(self.n)
    
    r = self.reward(self.state, action)
    if action == 1:
      if self.state != self.n - 1:
        self.state += 1
    else:
      if self.state != 0:
        self.state -= 1
    self.steps += 1
    if self.steps >= self.max_steps:
      is_done = True
    if self.state == 3 or self.state == 0:
      is_done = True
    else:
      is_done = False
    return (v <= self.state).astype('float32'), r, is_done, {}

  def reset(self):
    v = np.arange(self.n)
    self.state = 2
    self.steps = 0
    return (v <= self.state).astype('float32')
  
  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return seed

  def render(self, mode='human'):
    pass

  def close(self):
    return 0
  

if __name__ == '__main__':
  env = NChainEnv()
  env.seed(0)
  print('Action space:', env.action_space)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  cfg = {'n':5}
  env.init(**cfg)
  print('New obsevation space:', env.observation_space)
  print('New Obsevation space high:', env.observation_space.high)
  print('New Obsevation space low:', env.observation_space.low)
  
  for i in range(1):
    ob = env.reset()
    while True:
      action = env.action_space.sample()
      ob, reward, done, _ = env.step(action)
      print('Observation:', ob)
      print('Reward:', reward)
      print('Done:', done)
      if done:
        break
  env.close()