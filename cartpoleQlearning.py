# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 19:15:28 2019

@author: aidin
"""

import numpy as np
import gym
import matplotlib.pyplot as plt


def intStates(featureArr):
    return int(''.join(map(str,featureArr)))

def makeBin(inp,bins):
    return np.digitize([inp],bins)[0]


class FeatureTransformer:
    def __init__(self):
        
        #10 Bins for every feature of cartpole environment
        self.cartPosition = np.linspace(-2.4,2.4,9)
        self.cartVelocity = np.linspace(-3,3,9)
        self.poleAngle = np.linspace(-0.3,0.3,9)
        self.poleVelocity = np.linspace(-3,3,9)
        
    def transfomer(self, observation):
        cPos, cVel, pAng, pVel = observation
        
        return intStates([
                makeBin(cPos, self.cartPosition),
                makeBin(cVel, self.cartVelocity),
                makeBin(pAng, self.poleAngle),
                makeBin(pVel, self.poleVelocity)])
    


class Environment:
    def __init__(self,env,FeatureTransformer):
        self.env = env
        self.FT = FeatureTransformer
        self.alpha = 10e-3
        
        numState = 10**4
        numActions = env.action_space.n
        
        self.Q = np.random.uniform(low=-1,high = 1,size = (numState,numActions))
      
    def stateActions(self,s):
        x = self.FT.transfomer(s) 
        return self.Q[x]
    
    def update(self,s,a,G):
        x = self.FT.transfomer(s)
        self.Q[x,a] = self.alpha * (G - self.Q[x,a])
        
    def epsilonGreedy(self,s,eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            x = self.FT.transfomer(s)
            return np.argmax(self.Q[x])
    
    
def playEpisode(Env,eps,gamma):
    observation = env.reset()
    done = False
    totalReward = 0
    iterations = 0
    
    while not done and iterations < 10000:
        action = Env.epsilonGreedy(observation,eps)
        previousObservation = observation
        observation, reward, done, info = env.step(action)
        
        totalReward += reward
        
        if done and iterations < 199:
            reward = -100
         
        G = reward + gamma*np.max(Env.stateActions(observation))
        Env.update(previousObservation,action,G)
        
        iterations += 1
    
    return totalReward



if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    ft = FeatureTransformer()
    Env = Environment(env,ft)
    gamma = 0.9
    
    
    N = 10000
    totalReward = np.zeros(N)
    
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        
        totalReward[n] = playEpisode(Env,eps,gamma)
        if n % 100 ==0:
            print("Episode",n,"Total reward:", totalReward[n],"eps:",eps)
    
    plt.plot(totalReward)