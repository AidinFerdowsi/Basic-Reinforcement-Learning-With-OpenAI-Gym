# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 17:46:40 2019

@author: walids
"""

from __future__ import print_function, division
import gym 
import numpy as np
from builtins import range
import matplotlib.pyplot as plt


def act(s,w):
    return 1 if s.dot(w) > 0 else 0

def playEpisode(env,weights):
    observation = env.reset()
    done = False
    nEpisode = 10000
    t = 0
    while not done and t < nEpisode:
#        env.render()
        t +=1
        action = act(observation,weights)
        observation, reward, done, _ = env.step(action)
        if done:
            break
    return t

def simulate(env,nSim,weights):
    episodeLenghts = np.empty(nSim)
    for i in range(nSim):
        episodeLenghts[i] = playEpisode(env,weights)
    
    
    avgLen = episodeLenghts.mean()
    print("Average Length", avgLen)
    return avgLen

def randomSearch(env):
    best = 0
    weights = np.random.random(4)*2-1
    numSearch = 100
    episodeLen = np.empty(numSearch)
    for t in range(numSearch):
        newWeights = np.random.random(4)*2-1
        episodeLen [t] = simulate(env,100,newWeights)
        
        if episodeLen[t] > best:
            weights = newWeights
            episodeLen[t] = episodeLen[t]

    return episodeLen, weights

if __name__ == '__main__':
        
  env = gym.make('CartPole-v0')
  episodeLengths, weights = randomSearch(env)
  plt.plot(episodeLengths)
  plt.show()