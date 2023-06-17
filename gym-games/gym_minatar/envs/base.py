import gym
from gym import spaces

from minatar import Environment
import seaborn as sns
import numpy as np


class BaseEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self, game, display_time=50, use_minimal_action_set=False, **kwargs):
    self.game_name = game
    self.display_time = display_time
    self.game_kwargs = kwargs
    self.game = Environment(env_name=self.game_name, **kwargs)
    if use_minimal_action_set:
      self.action_set = self.game.minimal_action_set()
    else:
      self.action_set = list(range(self.game.num_actions()))
    self.action_space = spaces.Discrete(len(self.action_set))
    self.observation_space = spaces.Box(0.0, 1.0, shape=self.game.state_shape(), dtype=bool)
    self.visualized = False
    self.closed = False

  def step(self, action):
    action = self.action_set[action]
    reward, done = self.game.act(action)
    return (self.game.state(), reward, done, {})
    
  def reset(self):
    self.game.reset()
    return self.game.state()
  
  def seed(self, seed=None):
    self.game = Environment(
      env_name=self.game_name,
      random_seed=seed,
      **self.game_kwargs
    )
    return seed

  def render(self, mode='human'):
    if mode == 'rgb_array':
      state = self.game.state()
      n_channels = state.shape[-1]
      # cmap = sns.color_palette("cubehelix", n_channels)
      # cmap.insert(0, (0,0,0))
      # numerical_state = np.amax(
      #     state * np.reshape(np.arange(n_channels) + 1, (1,1,-1)), 2)
      # rgb_array = np.stack(cmap)[numerical_state]
      
      if not self.visualized:
            self.cmap = sns.color_palette("cubehelix", n_channels)
            self.cmap.insert(0, (0,0,0))
            self.cmap = colors.ListedColormap(self.cmap)
            bounds = [i for i in range(n_channels+2)]
            self.norm = colors.BoundaryNorm(bounds, n_channels+1)
            _, self.ax = plt.subplots(1,1)
            plt.show(block=False)
            self.visualized = True
        if self.closed:
            _, self.ax = plt.subplots(1,1)
            plt.show(block=False)
            self.closed = False
        state = self.game.state()
        numerical_state = np.amax(
            state * np.reshape(np.arange(n_channels) + 1, (1,1,-1)), 2) + 0.5
        self.ax.imshow(
            numerical_state, cmap=self.cmap, norm=self.norm, interpolation='none')
        plt.pause(time / 1000)
        plt.cla()
      
      
      
      
      
      return rgb_array
      # return self.game.state()
    elif mode == 'human':
      self.game.display_state(self.display_time)

  def close(self):
    if self.game.visualized:
      self.game.close_display()
    return 0