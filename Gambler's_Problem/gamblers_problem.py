#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 23:25:15 2019

@author: abhigyan
"""

"""Program for simulating the Gamblers Problem in RL and finding its solution
   using Value Iteration and Policy Iteration methods. NOTE: The best policy will
   not appear as the smooth contour given in the book. This is a well documented
   problem and is discussed in the following thread: http://www.incompleteideas.net/book/gamblers.html"""
   
   
import numpy as np

class Gamblers_Problem(object):
    
    def __init__(self, coinProbability, maxReward):
        #Probability of heads, and maximum possible reward in the problem
        #this is 100
        self.coinProbability = coinProbability
        self.maxReward = maxReward
        
    
    def _enumerate_states(self):
        #Enumerate all possible money satetes
        self.states = {}
        for i in range(0, self.maxReward + 1):
            self.states[i] = {}
            if(i == 0):
                self.states[i].update({'currentStateValue':0,
                                      'updatedStateValue':0})
            elif(i == self.maxReward):
                self.states[i].update({'currentStateValue':+1,
                                      'updatedStateValue':+1})
            else:
                self.states[i].update({'currentStateValue':0,
                                      'updatedStateValue':0})
                                      
    
    def _link_states(self):
        #Link between all possible money states
        for i in self.states:
            self.states[i]['nextStateAction'] = {}
            if(i == 0):
                continue 
            elif(i == self.maxReward):
                continue
            else:
                for j in range(min(i, self.maxReward-i) + 1):
                    self.states[i]['nextStateAction'][j] = j
        
        
    def _evaluate_state_value(self, policy, gamma):
        #Evaluate state values under a certain policy
        for i in self.states:
            if(i==0):
                continue
            elif(i==self.maxReward):
                continue
            else:
                statePolicy = policy[i]
                stateValue = 0
                for j in self.states[i]['nextStateAction']:
                    envValue = self.coinProbability*self.states[i+j]['currentStateValue'] + \
                               (1-self.coinProbability)*self.states[i-j]['currentStateValue']
                    stateValue += statePolicy[j] * gamma * envValue
                self.states[i]['updatedStateValue'] = stateValue
            
    
    def _update_state_values(self):
        #Update state values
        for i in self.states:
            self.states[i]['currentStateValue'] = self.states[i]['updatedStateValue']
            
            
    def value_iteration(self, gamma, theta):
        #Perform value iteration
        while(True):
            flag = True
            for i in self.states:
                if(i == 0):
                    continue
                if(i == self.maxReward):
                    continue
                maximum = -np.inf
                for j in self.states[i]['nextStateAction']:
                    envReward = gamma*(self.coinProbability*self.states[i+j]['currentStateValue'] + \
                                       (1-self.coinProbability)*self.states[i-j]['currentStateValue'])
                    if envReward > maximum:
                        maximum = envReward
                self.states[i]['updatedStateValue'] = maximum
                delta = np.abs(maximum - self.states[i]['currentStateValue'])
                if(delta > theta):
                    flag = False
            self._update_state_values()
            if(flag):
                break
            
            
    def _policy_evaluation(self, policy, gamma, theta):
        #Perform policy evaluation
        iterations = 0
        while(True):
            iterations += 1
            flag = True
            self._evaluate_state_value(policy, gamma)
            for i in self.states:
                delta = np.abs(self.states[i]['currentStateValue'] - self.states[i]['updatedStateValue'])
                if(delta>theta):
                    flag = False
            if(flag):
                break
            self._update_state_values()
        return iterations
            
            
    def _policy_improvement(self, gamma):
        #Perform policy improvement
        policy = {}
        for i in self.states:
            if(i == 0):
                continue
            elif(i == self.maxReward):
                continue
            else:
                maximum = -np.inf
                policy[i] = {}
                index = 0
                for j in self.states[i]['nextStateAction']:
                    envReward = gamma*(self.coinProbability*self.states[i+j]['currentStateValue'] + \
                                       (1-self.coinProbability)*self.states[i-j]['currentStateValue'])
                    if(envReward>maximum):
                        index = j
                    policy[i][j] = 0
                policy[i][index] = 1
        return policy
            
    
    def policy_iteration(self, gamma, theta):
        #Perform policy iteration
        while(True):
            policy = self._policy_improvement(gamma)
            iterations = self._policy_evaluation(policy, gamma, theta)
            if(iterations == 1):
                break
    
    
    def get_policy(self, gamma):
        #Get the best policy (deterministic)
        policy = []
        for i in self.states:
            maximum = -np.inf
            index = 0
            for j in self.states[i]['nextStateAction']:
                envReward = gamma*(self.coinProbability*self.states[i+j]['currentStateValue'] + \
                                       (1-self.coinProbability)*self.states[i-j]['currentStateValue'])
                if envReward > maximum:
                    maximum = envReward
                    index = j
                
            policy.append(index)
        return policy
    
    
    def get_state_dict(self):
        return self.states.copy()
    
    
"""
Examples:
    
import matplotlib.pyplot as plt

gamma = 1
p_h = 0.4
max_reward = 100
theta = 0.0000001
       
gp = Gamblers_Problem(p_h, max_reward)
gp._enumerate_states()
gp._link_states()     
gp.value_iteration(gamma, theta) 
y = gp.get_policy(gamma)

x = list(range(max_reward+1))
plt.plot(x,y)



gamma = 1
p_h = 0.4
max_reward = 200
theta = 0.0000001


gp1 = Gamblers_Problem(p_h, max_reward)
gp1._enumerate_states()
gp1._link_states() 
gp1.policy_iteration(gamma, theta)
y = gp.get_policy(gamma)
  
x = list(range(max_reward+1))
plt.plot(x,y)
"""       
        
        
                