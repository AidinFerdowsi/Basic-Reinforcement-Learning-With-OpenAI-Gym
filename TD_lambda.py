# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:35:30 2019

@author: aidin
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

import mountainClimbRBF 
from mountainClimbRBF import FeatureTransformer, plotAvgReward




class ModelEligibility:
    def __init__(self, D):
        self.w = np.random.rand(D)/np.sqrt(D)
        
        
    def partial_fit(self,input_,target,eligibility,lr = 100e-3):
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
        X = self.ft.transform([s])
        assert(len(X.shape)==2)
        return np.array([m.predict(X)[0] for m in self.models])
    
    
    
    def updateValue(self,s,a,G,gamma,lambda_):
        X = self.ft.transform([s])
        assert(len(X.shape)==2)
        self.eligibilities *=gamma*lambda_
        self.eligibilities[a] +=X[0]
        self.models[a].partial_fit(X[0],G,self.eligibilities)
        
        
    def takeAction(self,s,eps):
        if np.random() <eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))
        
        