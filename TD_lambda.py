# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:35:30 2019

@author: aidin
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

#import mountainClimbRBF 
from mountainClimbRBF import FeatureTransformer, plotAvgReward




class ModelEligibility:
    def __init__(self, D):
        self.w = np.random.rand(D)/np.sqrt(D)
        
        
    def partial_fit(self,input_,target,eligibility,lr = 10e-3):
        self.w += lr*(target - input_.dot(self.w))*eligibility
        
        
    def predict(self,X):
        X = np.array(X)
        return X.dot(self.w)
    
    


class Model:
    def __init__(self,env,feature_transformer):
        self.env = env
        self.ft = feature_transformer
        self.models = []
        
        
        D = feature_transformer.dimensions
        self.eligibilities = np.zeros((env.action_space.n,D))
        
        for i in range(env.action_space.n):
            self.models.append(ModelEligibility(D))
            
            
    def evaluateState (self,s):
        X = self.ft.featureTransformer([s])
        assert(len(X.shape)==2)
        return np.array([m.predict(X)[0] for m in self.models])
    
    
    
    def updateValue(self,s,a,G,gamma,lambda_):
        X = self.ft.featureTransformer([s])
        assert(len(X.shape)==2)
        self.eligibilities *=gamma*lambda_
        self.eligibilities[a] +=X[0]
        self.models[a].partial_fit(X[0],G,self.eligibilities[a])
        
        
    def epsilonGreedy(self,s,eps):
        if np.random.random() <eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.evaluateState(s))

def playEpisode(model,env,eps,gamma,lambda_):
    observation = env.reset()
    done = False
    totalReward = 0
    iterations = 0
    
    while not done and iterations < 10000:
        action = model.epsilonGreedy(observation,eps)
        previousObservation = observation
        observation, reward, done, info = env.step(action)
        
        totalReward += reward
         
        G = reward + gamma*np.max(model.evaluateState(observation)[0])
        model.updateValue(previousObservation,action,G,gamma,lambda_)
        
        iterations += 1
    return totalReward
        

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    gamma = 0.99
    lambda_ = 0.7
    
    N = 300
    totalRewards = np.empty(N)
    costs = np.empty(N)
    
    for n in range(N):
        eps = 0.1*(0.97**n)
        totalRewards[n] = playEpisode(model, env, eps, gamma,lambda_)
#        if (n + 1) % 100 == 0:
        print("episode:", n+1, "total reward:", totalRewards[n])
    
    print("avg reward for last 100 episodes:", totalRewards[-100:].mean())
    print("total steps:", -totalRewards.sum())
    
    plt.plot(totalRewards)
    plt.title("Rewards")
    plt.show()

    plotAvgReward(totalRewards)