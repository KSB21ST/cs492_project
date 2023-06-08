import gym
from gym import spaces
import numpy as np
from gym.utils import seeding


# Adapted from https://github.com/facebookresearch/RandomizedValueFunctions/blob/master/qlearn/envs/nchain.py
class NChainEnv(gym.Env):
  ''' N-Chain environment
  The environment consists of a chain of N states and the agent always starts in state s1, from where it can either move left or right.
  Original Implementaion has four states: terminal-B-A-terminal, But we have terminal-A-terminal, for easier implementation. 
  Still, there is no difference in dynamics since the agent only can select moving left at B
  '''
  def __init__(self, n=3):
    self.state = 1  # Start at state s2 terminal(s1) - A(s2) - terminal(s3) (0, 1, and 2 respectively)
    self.action_space = spaces.Discrete(2)
    self.seed()
    self.init(n)
    self.mu = 0.1
    
  def init(self, n=3):
    self.n = n
    self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)
    self.max_steps = 1 # in this implementation, after choosing any action, the agent moves to a terminal state
    self.reward_at_B = []
    for m in range(4):
      v = np.random.uniform(-1,1)
      self.reward_at_B.append(v)
      self.reward_at_B.append(-v)
  
  def reward(self, s, a):
    if s==1 and a==0: # Moving Left at B
      r = np.random.choice(self.reward_at_B) # one of 8 pre-defined rewards
      return r
    else:
      return 0.0

  def step(self, action):
    assert self.action_space.contains(action)
    v = np.arange(self.n)
    r = self.reward(self.state, action)

    if action == 1:
      self.state += 1
    else:
      self.state -= 1
    
    self.steps +=1
    is_done = True # always

    return (v <= self.state).astype('float32'), r, is_done, {}

  def reset(self):
    v = np.arange(self.n)
    self.state = 1
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

  cfg = {'n':3}
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