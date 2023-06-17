import gym
import gym_minatar
import gym_pygame
import gym_exploration
from gym_recorder import Recorder
import os
import numpy as np
import skvideo.io
from PIL import Image

class RandomAgent(object):
  def __init__(self, action_space):
    self.action_space = action_space

  def act(self, observation, reward, done):
    return self.action_space.sample()

if __name__ == '__main__':
  # game = 'Catcher-PLE-v0'
  # game = 'FlappyBird-PLE-v0'
  # game = 'Pixelcopter-PLE-v0'
  # game = 'PuckWorld-PLE-v0'
  # game = 'Pong-PLE-v0'
  
  game = 'Asterix-MinAtar-v0'
  # game = 'Breakout-MinAtar-v0'
  # game = 'Freeway-MinAtar-v0'
  # game = 'Seaquest-MinAtar-v0'
  # game = 'SpaceInvaders-MinAtar-v0'

  # game = 'Asterix-MinAtar-v1'
  # game = 'Breakout-MinAtar-v1'
  # game = 'Freeway-MinAtar-v1'
  # game = 'Seaquest-MinAtar-v1'
  # game = 'SpaceInvaders-MinAtar-v1'

  # game = 'NChain-v1'
  # game = 'LockBernoulli-v0'
  # game = 'LockGaussian-v0'
  # game = 'SparseMountainCar-v0'
  # game = 'DiabolicalCombLock-v0'

  env = gym.make(game)
  # video = VideoRecorder(env, "./videos/test.mp4")
  # env = Recorder(env, episode_num=10)
  if game in ['NChain-v1', 'LockBernoulli-v0', 'LockGaussian-v0', 'DiabolicalCombLock-v0']:
    game_cfg = {
      'NChain-v1': {'n':5},
      'LockBernoulli-v0': {'horizon':10, 'dimension':10, 'switch':0.1},
      'LockGaussian-v0': {'horizon':9, 'dimension':9, 'switch':0.1, 'noise':0.1},
      'DiabolicalCombLock-v0': {"horizon":5, "swap":0.5}
    }
    env.init(**game_cfg[game])
  env.seed(0)

  print('Game:', game)
  print('Action space:', env.action_space)
  print('Obsevation space:', env.observation_space)
  # try:
    # print('Obsevation space high:', env.observation_space.high)
    # print('Obsevation space low:', env.observation_space.low)
  # except:
    # pass
  
  def save_video(video, dir):
    fps = '1'
    crf = '17'
    vid_out = skvideo.io.FFmpegWriter(f'{dir}.mp4', 
                # inputdict={'-r': fps},
                outputdict={'-r': fps, '-c:v': 'libx264', '-crf': crf, 
                            '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'}
    )

    for frame in video:
        vid_out.writeFrame(frame)
    vid_out.close()

  for i in range(1):
    ob = env.reset()
    done = False
    img_array = []
    step_cnt = 0
    while not done:
      action = env.action_space.sample()
      step_cnt += 1
      action = 0
      n_ob, reward, done, _ = env.step(action)
      # env.render('rgb_array') # default render mode is 'human'
      # video.capture_frame()
      # env.render('human')
      img = env.render('rgb_array')
      img_array.append(img)
      Image.fromarray(img, "RGB").save('./videos/sample' + str(step_cnt) + '.png', 'png')
      # print('Observation:', type(ob))
      # print('Reward:', reward)
      # print('Done:', done)
      # env.txtqueue.append(f"episode:{i}")
      # env.txtqueue.append(f"obs:{ob}")
      # env.txtqueue.append(f"action:{action}")
      # env.txtqueue.append(f"reward:{reward}")
      # env.txtqueue.append(f"next obs:{n_ob}")
      ob = n_ob
      # if done:
      #   break
    print(np.array(img_array).shape)
    save_video(np.array(img_array), './sample')

  env.close()