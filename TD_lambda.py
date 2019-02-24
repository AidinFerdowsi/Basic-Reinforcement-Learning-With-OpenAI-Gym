# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:35:30 2019

@author: aidin
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

import mountainClimbRBF 
from mountainClimbRBF import FeatureTransformer, Model, plotAvgReward




class FitEligibility:
    def __init__(self, D):
        self.w = np.random.rand(D)/np.sqrt(D)
        
        
    def partial_fit(self,input_,target,eligibility,lr = 100e-3):
        self.w += lr*(target - input_.dot(self.w))*eligibility
        
        
    def predict(self,X):
        X = np.array(X)
        return X.dot(self.w)