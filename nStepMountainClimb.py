# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:40:57 2019

@author: aidin
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

import mountainClimbRBF 
from mountainClimbRBF import FeatureTransformer, Model, plotAvgReward

class SDGRegressor:
    def __init__(self, **kwargs):
        self.w = None
        self.lr = 10e-3
        
    
    
    def partial_fit(self, X, Y):
        if self.w is None:
            D = X.shape[1]
            self.w = np.random.random(D) / np.sqrt(D)
        self.w += self.lr *(Y-X.dot(self.w)).dot(X)
        
        
    def predict(self,X):
        return X.dot(self.w)
    

mountainClimbRBF.SGDRegressor = SDGRegressor





def playEpisode(model, env, eps,gamma, n=5):
    observation = env.reset()
    done  = False 
    totalReward = 0
    rewards = []
    states = []
    actions = []
    iters = 0
    
    multiplier = np.array([gamma]*n)**np.arange(n)
    
    while not done and iters < 10000:
        
        
        
        action = model.epsilonGreedy(observation, eps)
        
        states.append(observation)
        actions.append(action)
        
        
#        prevAction = action
        observation, reward, done, info = env.step(action)
        
        rewards.append(reward)
        
        
        
        
        if  len(rewards)>=n:
            returnNstep = multiplier.dot(rewards[-n:])
            G = returnNstep + (gamma**n)*np.max(model.evaluateState(observation)[0])
            model.updateValue(states[-n],actions[-n],G)
            
        totalReward += reward
        iters += 1
        
    if n==1:
        rewards = []
        states = []
        actions = []
    else:
        rewards = rewards[-n+1:]
        states = states[-n+1:]
        actions = actions[-n+1:]
    
    
    if observation[0] >= 0.5:
        
        
        while len(rewards) > 0:
            G = multiplier[:len(rewards)].dot(rewards)
            model.updateValue(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
    else:
        while len(rewards) > 0:
            guess_rewards = rewards + [-1]*(n - len(rewards))
            G = multiplier.dot(guess_rewards)
            model.updateValue(states[0], actions[0], G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)
            
    return totalReward
    
    
if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")
    gamma = 0.99
    
    
    N = 500
    totalrewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        # eps = 1.0/(0.1*n+1)
        eps = 0.1*(0.97**n)
        totalreward = playEpisode(model, env, eps, gamma)
        totalrewards[n] = totalreward
        print("episode:", n, "total reward:", totalreward)
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())
    
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()
    
    plotAvgReward(totalrewards)