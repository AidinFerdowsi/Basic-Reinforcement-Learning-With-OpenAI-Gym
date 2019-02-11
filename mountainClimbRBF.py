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