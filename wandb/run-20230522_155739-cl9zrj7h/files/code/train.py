#Imports and gym creation
import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam
import random
import datetime
import pickle
import csv
import os
import argparse
import wandb 
from model.ddqn import DoubleDeepQNetwork
from model.dqn import DeepQNetwork

parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")
parser.add_argument("--model", default="DoubleDeepQNetwork",
                    choices=["DoubleDeepQNetwork", "DeepQNetwork"], 
                    help="Model name")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--discounting", default=0.95,
                    type=float, help="Discounting factor.")
parser.add_argument("--learning_rate", default=0.001,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--batch_size", default=24, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--episodes", default=200, type=int, metavar="E",
                    help="Episode number")

#Create Gym
# envCartPole.seed(50)

def train(flags, rewards, epsilons, q_values, total_e_q, TEST_Episodes):
    env = gym.make('CartPole-v1')
    env_test = gym.make('CartPole-v1')
    nS = env.observation_space.shape[0] #This is only 4
    nA = env.action_space.n #Actions
    #self, states, actions, alpha, gamma, epsilon,epsilon_min, epsilon_decay
    if flags.model == 'DoubleDeepQNetwork':
        model = DoubleDeepQNetwork(nS, nA, flags.learning_rate, flags.discounting, 1, 0.001, 0.995 )
    elif flags.model == 'DeepQNetwork':
        model = DeepQNetwork(nS, nA, flags.learning_rate, flags.discounting, 1, 0.001, 0.995 )
    batch_size = flags.batch_size
    EPISODES = flags.episodes
    #Training
    print("Start Training ...")
    print(flags)
    avg_rew = 0
    train_step = 0
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, nS]) # Resize to store in memory to pass to .predict
        tot_rewards = 0
        
        #for evaluating q values
        e_state = env_test.reset()
        e_state = np.reshape(e_state, [1, nS])
        epi_rew = 0
        total_e_rew = []
        for time in range(210): #200 is when you "solve" the game. This can continue forever as far as I know
            action = model.action(state, q_values)
            nstate, reward, done, _ = env.step(action)
            nstate = np.reshape(nstate, [1, nS])
            tot_rewards += reward
            epi_rew += reward * (flags.discounting**time)
            model.store(state, action, reward, nstate, done) # Resize to store in memory to pass to .predict
            state = nstate

            #evaluation of each freezer parameter
            print("episode: {} timestep: {}/{}, start evaluating q values for 50 steps".format(e, time, 210))
            max_step_q = []
            min_step_q = []
            # step_rew = 0
            if len(model.memory) > 50:
                train_step+= 1
                e_minibatch = random.sample( model.memory, 50 )
                for e_state, e_action, e_reward, e_nstate, e_done in e_minibatch:
                    e_action = model.test_action(e_state, max_step_q, min_step_q)
                    e_nstate, e_reward, e_done, _ = env_test.step(e_action)
                    # step_rew += e_reward
            else:
                for _time in range(50):
                    train_step+= 1
                    e_action = model.test_action(e_state, max_step_q, min_step_q)
                    e_nstate, e_reward, e_done, _ = env_test.step(e_action)
                    # step_rew += e_reward
                    e_nstate = np.reshape(e_nstate, [1, nS])
                    e_state = e_nstate
                    if e_done:
                        print("time: {}/{} timestep: {}/{}, terminated->restart".format(time, 210, _time, 50))
                        e_state = env_test.reset()
                        e_state = np.reshape(e_state, [1, nS])
                        continue
            max_state_q = sum(max_step_q) / len(max_step_q)
            min_state_q = sum(min_step_q) / len(min_step_q)
            avg_state_q = (max_state_q + min_state_q)/2
            print("episode: {} timestep: {}/{}, average q value: {}".format(e, time, 210, max_state_q))
            total_e_q.append(max_state_q)
            # total_e_rew.append(step_rew/50)
            wandb.log({'Traning_step': train_step, 'Max_q_value': max_state_q, 'Min_q_value': min_state_q, 'Avg_q_value': avg_state_q})
                

            #done: CartPole fell. 
            #time == 209: CartPole stayed upright
            if done or time == 209:
                rewards.append(tot_rewards)
                epsilons.append(model.epsilon)
                print("episode: {}/{}, timestep: {}/{}, score: {}, e: {}"
                    .format(e, EPISODES, time, 210, tot_rewards, model.epsilon))
                break
            #Experience Replay
            if len(model.memory) > batch_size:
                model.experience_replay(batch_size)
        avg_rew += epi_rew
        
        #log
        with open("./logs/results/" + flags.model + '.csv', "a") as f:
            writer = csv.writer(f)
            for idx in range(time):
                data = [e, idx, total_e_q[idx], avg_rew] #total_e_rew[idx], 
                writer.writerow(data)
        
        if flags.model == 'DoubleDeepQNetwork':
            #Update the weights after each episode (You can configure this for x steps as well
            model.update_target_from_model()
        #If our current NN passes we are done
        #I am going to use the last 5 runs
        if len(rewards) > 5 and np.average(rewards[-5:]) > 195:
            #Set the rest of the EPISODES for testing
            TEST_Episodes = EPISODES - e
            TRAIN_END = e
            break
        if e % 50 == 0:
            model.save_models()
    
    
        
def test(flags, rewards, epsilons, q_values, total_e_q, TEST_Episodes):
    env = gym.make('CartPole-v1')
    nS = env.observation_space.shape[0] #This is only 4
    nA = env.action_space.n #Actions
    model = keras.models.load_model("my_model")
    batch_size = flags.batch_size
    #Testing
    eval_q = []
    print('Testing started...')
    #TEST Time
    #   In this section we ALWAYS use exploit don't train any more
    for e_test in range(10):
        state = env.reset()
        state = np.reshape(state, [1, nS])
        tot_rewards = 0
        for t_test in range(210):
            action = model.test_action(state, eval_q)
            nstate, reward, done, _ = env.step(action)
            nstate = np.reshape( nstate, [1, nS])
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
            
if __name__ == "__main__":
    flags = parser.parse_args()
    rewards = [] #Store rewards for graphing
    epsilons = [] # Store the Explore/Exploit
    q_values = []
    total_e_q = []
    TEST_Episodes = 0
    
    wandb.init(project=flags.model,
           config=flags,
           save_code=True,
           sync_tensorboard=True,
           )
    
    if flags.mode == "train":
        train(flags, rewards, epsilons, q_values, total_e_q, TEST_Episodes)
    if flags.mode == "test":
        test(flags, rewards, epsilons, q_values, total_e_q, TEST_Episodes)
    wandb.finish()