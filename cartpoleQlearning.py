# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 19:15:28 2019

@author: aidin
"""

import numpy as np


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
    
    