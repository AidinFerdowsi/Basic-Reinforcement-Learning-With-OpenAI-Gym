# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 18:24:24 2019

@author: aidin
"""

import theano
import theano.tensor as T
import numpy as np


class SGDRegressor:
    def __init__(self,D):
        w = np.random.randn(D)/np.sqrt(D)
        self.w = theano.shared(w)
        self.lr = 10e-2
        
        X = T.matrix('X')
        Y = T.vector('Y')
        
        Y_hat = X.dot(self.w)
        delta = Y - Y_hat
        cost = delta.dot(delta)
        grad = T.grad(cost,self.w)
        
        updates = [(self.w,self.w-self.lr*grad)]
        
        self.train_op = theano.function(
                inputs = [X,Y],
                updates = updates
                )
        
        self.predict_op = theano.function(
                inputs = [X],
                outputs = Y_hat
                )
        
        
    def partial_fit(self,X,Y):
        self.train_op
    
    
    def predict(self,X):
        return self.predict_op
    
    
    
if __name__ == '__main__':

    sgdr = SGDRegressor(3)
    sgdr.partial_fit(np.random.random(10),5 * np.random.random(10))
    print(sgdr.predict(2.5))

        