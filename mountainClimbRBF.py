# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:45:11 2019

@author: Aidin
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
        return self.featureExtractor.transform(self.scaler.transform(observation))
    
    
class Model:
    def __init__(self,env,featureTransformer,learningRate):
        self.env = env
        self.models = []
        self.featureTransformer = featureTransformer
        
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learningRate)
            model.partial_fit(featureTransformer.featureTransformer([env.reset()]),[0])
            self.models.append(model)
            
    def evaluateState(self,s):
        x = self.featureTransformer.featureTransformer([s])
        return np.array([model.predict(x)[0] for model in self.models])
    
    def updateValue(self,s,a,G):
        x = self.featureTransformer.featureTransformer([s])
        self.models[a].partial_fit(x,[G])
        
        
    def epsilonGreedy(self,s,eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.evaluateState(s))
        
        
def playEpisode(model,env,eps,gamma):
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
        model.updateValue(previousObservation,action,G)
        
        iterations += 1
    return totalReward

def plotCost2Go(env,model,nCells = 20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=nCells)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=nCells)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(model.evaluateState(_)), 2, np.dstack([X, Y]))
    
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z,
    rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Cost-To-Go == -V(s)')
    ax.set_title("Cost-To-Go Function")
    fig.colorbar(surf)
    plt.show()
    
def plotAvgReward(totalRewards):
    N = len(totalRewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalRewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()
    
    
def main():
    env = gym.make('MountainCar-v0')
    ft = FeatureTransformer(env)
    model = Model(env, ft, "constant")
    gamma = 0.99
    
    N = 300
    totalRewards = np.empty(N)
    
    for n in range(N):
        eps = 0.1*(0.97**n)
        if n == 199:
            print("eps:", eps)
        totalRewards[n] = playEpisode(model, env, eps, gamma)
        if (n + 1) % 100 == 0:
            print("episode:", n, "total reward:", totalRewards[n])
    
    print("avg reward for last 100 episodes:", totalRewards[-100:].mean())
    print("total steps:", -totalRewards.sum())
    
    plt.plot(totalRewards)
    plt.title("Rewards")
    plt.show()

    plotAvgReward(totalRewards)

    # plot the optimal state-value function
#    plotCost2Go(env, model)
if __name__ == '__main__':

  main()