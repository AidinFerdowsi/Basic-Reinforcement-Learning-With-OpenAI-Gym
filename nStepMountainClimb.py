# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:40:57 2019

@author: aidin
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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
        
        
        
        action = model.epsilonGreed(observation, eps)
        
        states.append(observation)
        actions.append(action)
        
        
        prevAction = action
        observation, reward, done, info = env.step(action)
        
        rewards.append(reward)
        
        
        
        
        if  len(rewards)>n:
            returnNstep = multiplier.dot(rewards[-n:])
            G = returnNstep + (gamma**n)*np.max(model.evaluateState(observation)[0])
            model.updateValue(states[-n],actions[-n],G)
            
        totalReward += reward
        iters += 1
    