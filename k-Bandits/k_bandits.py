#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 18:39:09 2020

@author: abhigyan
"""
import numpy as np
from matplotlib import pyplot as plt

# Class which takes in a function and its parameters from which rewards are sampled for
# a particular bandit.
# If non-stationary, send a single mean-variance pair of Gaussian distribution, according
# to which the parameters will be varied.
class Sampling_Function(object):
    
    def __init__(self, sampling_type='Gaussian', parameters=(0,1), stationary=True,
                 non_stationary_params = (0,1)):
        self.sampling_type = sampling_type
        self.parameters = parameters
        self.stationary = stationary
        self.non_st_params = non_stationary_params
    
    def gaussian(self):
        if len(self.parameters) != 2:
            raise ValueError("Incorrect number of Parameters !!!")
        else:
            return np.random.normal(loc=self.parameters[0], scale=self.parameters[1])
    
    def updateParameters(self):
        if self.sampling_type == 'Gaussian':
            self.parameters[0] += np.random.normal(loc=self.non_st_params[0], 
                                                   scale=self.non_st_params[1])
        
    def getSample(self):
        if self.sampling_type == 'Gaussian':
            return self.gaussian()
   
     
        
# Class to define the updation scheme used. User can easily add new updation schemes. 
# Takes in the scheme and other pramaters related to schemes.
class Updation_Scheme(object):
    
    def __init__(self, scheme = 'average', alpha=0.1):
        self.scheme = scheme
        self.alpha = 0.1
        
    def average(self, estimatedQ, sampledQ, track):
        updateQ = (sampledQ - estimatedQ) / track
        return updateQ
    
    def exponentialConstant(self, estimatedQ, sampledQ):
        updateQ = self.alpha*(sampledQ-estimatedQ)
        return updateQ
    
    def update(self, estimatedQ, sampledQ, track=None):
        if self.scheme == 'average':
            updateQ = self.average(estimatedQ, sampledQ, track)
        elif self.scheme == 'exponential_constant':
            updateQ = self.exponentialConstant(estimatedQ, sampledQ)
        return updateQ
        
        
# Class used to define the updation scheme to be used in k-bandit
# problem.
class Action_Selection(object):
    
    def __init__(self, scheme):
        self.scheme = scheme
        
    def greedy(self, QValueArray):
        self.action = np.argmax(QValueArray)
        #print(QValueArray)
        #print(self.action)
    
    def epsilonGreedy(self, QValueArray, epsilon):
        p1 = epsilon
        p2 = (1-epsilon)/(len(QValueArray)-1)
        index = np.argmax(QValueArray)
        p = len(QValueArray)*[p2]
        p[index] = p1
        self.action = np.random.choice(range(len(QValueArray)), p=p)
        
    def getAction(self, QValueArray, epsilon=0.8):
        if self.scheme == 'greedy':
            self.greedy(QValueArray)
        elif self.scheme == 'epsilon_greedy':
            self.epsilonGreedy(QValueArray, epsilon)
        return self.action
        
        



"""Takes in a number of Bandits with default number=10, a list of objects of "Sampling" 
   class which values will be sampled,  a list of objects of "Updation" class from which 
   updates will be tracked, a list of objects from "Action_Selection" class which will
   decide the action to be chosen. The function plots the action value for 1000 steps. 
   Function returns the rewards attained over the entire run."""
   
def K_Armed_Bandit(bandits=10, sampling_function=None, updation_scheme=None, 
                   action_selection=None, steps=1000, QValues=None):
    
    # Initializa the Q Values
    if QValues == None:
        QValues = np.zeros(shape=bandits)
    elif type(QValues) == int:
        QValues = np.random.random(bandits) + QValues
    else:
        QValues = np.array(QValues)
    
    # Used to Track rewards
    rewards = []
    # Track the number of updates to each k-bandit
    track = np.zeros(shape=bandits)
    
    # Start the run over 1000 steps
    for i in range(steps):
        action = action_selection.getAction(QValues)
        track[action] += 1
        sampledQ = sampling_function[action].getSample()
        updateQ = updation_scheme.update(QValues[action], sampledQ, track[action])
        QValues[action] +=  updateQ
        rewards.append(sampledQ)
    
    return rewards
            
            
"""    
Example 1 (10 bandits, sampling=Gaussian (for all with a random mean between -5, 5 and variance = 2), 
           updation=average, action=greedy, steps=1000, Q value=5)
"""



sampling_functions = []
store_parameters = []
steps=1000
runs = 1000

# Creating sampling function for each of the bandits    
for i in range(20):
    mean = np.random.uniform(low=-5.0, high=5.0)
    parameters = (mean, 1)
    sf = Sampling_Function(sampling_type='Gaussian', parameters=parameters, stationary=False,
                           non_stationary_params = (0,1))
    sampling_functions.append(sf)
    store_parameters.append(parameters)


""" Comparing 3 schemes of solving k-bandit problem"""
updation_scheme = Updation_Scheme(scheme='exponential_constant')
action_selection = Action_Selection(scheme='greedy')

rewards = np.zeros(shape=steps)
for i in range(runs):
    kab = K_Armed_Bandit(bandits=20, sampling_function=sampling_functions, 
                          updation_scheme=updation_scheme, action_selection=action_selection, 
                          steps=steps, QValues=0)
    rewards += np.array(kab)

plt.plot(range(steps), rewards/runs)
plt.show()


updation_scheme = Updation_Scheme(scheme='exponential_constant')
action_selection = Action_Selection(scheme='epsilon_greedy')

rewards = np.zeros(shape=steps)
for i in range(runs):
    kab = K_Armed_Bandit(bandits=20, sampling_function=sampling_functions, 
                          updation_scheme=updation_scheme, action_selection=action_selection, 
                          steps=steps, QValues=0)
    rewards += np.array(kab)

plt.plot(range(steps), rewards/runs)
plt.show()


updation_scheme = Updation_Scheme(scheme='average')
action_selection = Action_Selection(scheme='epsilon_greedy')

rewards = np.zeros(shape=steps)
for i in range(runs):
    kab = K_Armed_Bandit(bandits=20, sampling_function=sampling_functions, 
                          updation_scheme=updation_scheme, action_selection=action_selection, 
                          steps=steps, QValues=0)
    rewards += np.array(kab)

plt.plot(range(steps), rewards/runs)
plt.show()

# To check absolute performance
print(store_parameters)
    
