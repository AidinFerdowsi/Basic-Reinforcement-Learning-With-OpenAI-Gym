# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:45:11 2019

@author: Aidin
"""

import gym
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor


class FeatureTransformer:
    def __init__(self,env,nComponents = 500):
        examples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(examples)
        
        
        
        featureExtractor = FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=nComponents)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=nComponents)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=nComponents)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=nComponents))
                ])
    
        exampleFeatures = featureExtractor.fit_transform(scaler.transform(examples))
        
        self.dimensions = exampleFeatures.shape[1]
        self.scaler = scaler
        self.featureExtractor = featureExtractor
        
        
    def featureTransformer(self,observation):
        return self.featurizer.transform(self.scaler.transform(observation))
    
    
class Model:
    def __init__(self,env,featureTransformer,learningRate):
        self.env = env
        self.models = []
        self.featureTransformer = featureTransformer
        
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learningRate)
            model.partial_fit(featureTransformer.featureTransfomer(env.reset()),[0])
            self.models.append(model)
            
    def evaluateState(self,s):
        x = self.featureTransformer.feaTransformer(s)
        return np.array([model.predict(x)[0] for model in self.models])
    
    def updateValue(self,s,a,G):
        x = self.featureTransformer.featureTransformer(s)
        self.models[a].partial_fit(x,[G])
        
        
    def epsilonGreedy(self,s,eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.evaluateState(s))
        
        
def playEpisode(model,eps,gamma):
    observation = env.reset()
    done = False
    totalReward = 0
    iterations = 0
    
    while not done and iterations < 10000:
        action = model.epsilonGreedy(observation,eps)
        previousObservation = observation
        observation, reward, done, info = env.step(action)
        
        totalReward += reward
         
        G = reward + gamma*np.max(model.stateState(observation)[0])
        model.update(previousObservation,action,G)
        
        iterations += 1
    return totalReward