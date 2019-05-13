#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 20:04:41 2019

@author: abhigyan
"""


import numpy as np

"""State class which treats each position of a Gridworld as a state and has methods
   to enumerate succcesive states, link succesive states, find expected rewards for 
   each move from a state, act greedily w.r.t a state, act w.r.t a policy, update
   state values, and return state values.
"""

class state(object):
    
    # Change the cost of a single step 
    actionValue = -1
    
    def __init__(self, position, terminal, max_x, max_y):
        #Initializes position of a step in Gridworld, whether its goal state,
        #assigns a random state value and assigns the boundaries
        self.position = position
        self.terminal = terminal
        self.stateValue = -np.random.randint(low=1, high=100)
        self.mins = 0,0
        self.maxs = max_x-1, max_y-1
        if(terminal):
            self.actionValue = 0
            self.stateValue = 0
            
        
    def enumerate_next_states(self):
        #Enumerates all possible next states and the wall hittings are considered
        #as transition to itself but with an associated step cost
        if(self.terminal):
            self.nextStates = np.array([self.position])
        else:
            possibleMoves = np.array([[-1,0], [0,1], [1,0], [0,-1]])
            theoreticalNextStates = possibleMoves + self.position
            nextStates = theoreticalNextStates[(theoreticalNextStates >= self.mins).all(axis = 1) &
                                                    (theoreticalNextStates <= self.maxs).all(axis = 1)]
            #Assigns the non-valid states as self transition states
            if(len(nextStates<4)):
                wallStates = np.repeat(self.position.reshape((-1,2)), [4-len(nextStates)], axis=0)
                self.nextStates = np.concatenate((nextStates, wallStates), axis = 0)

                
    def link_states(self, stateDict):
        #Links a state to all reacgable states from it (including self transition)
        self.nextStateList = []
        for i,state in enumerate(self.nextStates):
            self.nextStateList.append(stateDict[(state[0], state[1])])
                
    
    def get_rewards(self):
        #Returns the expected reward for each state to state conversion 
        # action_cost + v(s')
        rewards = np.array([])
        for i,state in enumerate(self.nextStateList):
            rewards = np.append(rewards, self.actionValue + state.get_state_value())
        return rewards
                
    
    def act_greedily_max(self, rewards):
        #Acts greedily w.r.t current state irrespective of policy
        #Used in VI and is basically a max operation among all possible conversions
        reward = -np.inf
        for i in rewards:
            if(i > reward):
                reward = i
        self.newStateValue = reward
        return np.abs(self.newStateValue-self.stateValue)
        
    
    def act_policy(self, policy):
        #Acts on the basis of a policy and returns the expected value for that
        #policy for a given state
        rewards = self.get_rewards()
        self.newStateValue = np.sum(policy * rewards)
        return np.abs(self.newStateValue-self.stateValue)
        
    
    def update_state_values(self):
        #Updates the state values with the newkly state value
        self.stateValue = self.newStateValue
            
            
    def get_state_value(self):
        #Returns the expected value v(s) of a state
        return self.stateValue
    


"""Gridworld class which initializes the entire Gridworld and also has the associated 
   operations of Value and Policy Iteration alongwith a display method to display the
   State Value Function
"""    
    
class Gridworld(object):
    
    def __init__(self, dimensions, terminalStates):
        #Initializes the entire Gridworld, creates the different positions as states
        #links them to the succesive states, assigns the state value function
        self.dimensions = dimensions
        self.terminalStates = terminalStates
        self.stateDict = {}
        for i in range (dimensions[0]):
            for j in range (dimensions[1]):
                position = np.array([i,j])
                terminal = np.any(np.equal(self.terminalStates, position).all(axis=1))
                s = state(position, terminal, dimensions[0], dimensions[1])
                self.stateDict[(i,j)] = s
                
        for i in self.stateDict:
            self.stateDict[i].enumerate_next_states()
            self.stateDict[i].link_states(self.stateDict)
            
     
    def value_iteration(self, threshhold):
        #Performs Value Iteration on the given Gridworld with threshold as acceptable error
        while(True):
            flag = True
            for _,value in self.stateDict.items():
                rewards = value.get_rewards()
                delta = value.act_greedily_max(rewards)
                if(delta>threshhold):
                    flag = False
            for _,value in self.stateDict.items():
                value.update_state_values()
            if(flag):
                break
            
            
    def policy_iteration(self, dimensions, threshhold, policy='random'):
        #Performs Policy Iteration on the given Gridworld with threshold as acceptable error
        pi = {}
        if(policy == 'random'):
            for i in range(0, dimensions[0]):
                for j in range(dimensions[1]):
                    pi[(i,j)] = np.random.random(4)
                    pi[(i,j)] = pi[(i,j)]/np.sum(pi[(i,j)])
        elif(policy == 'stochastic'):
            for i in range(0, dimensions[0]):
                for j in range(dimensions[1]):
                    pi[(i,j)] = np.ones((4,))/4
        
        while(True):
            iterations = 0 
            
            #Policy Evaluation
            while(True):
                evalFlag = True
                for key,value in self.stateDict.items():
                    delta = value.act_policy(pi[key])
                    if(delta>threshhold):
                        evalFlag = False
                for _,value in self.stateDict.items():
                    value.update_state_values()
                if(evalFlag):
                    break
                iterations += 1
            
            if(iterations == 0):
                break
            
            #Policy Improvement
            for key,value in self.stateDict.items():
                rewards = value.get_rewards()
                pi[key] = np.zeros((4,))
                pi[key][np.argmax(rewards)] = 1
            
        
    def display_gridworld(self):
        #Displays the final state value function
        values = np.zeros(self.dimensions)
        for key,val in self.stateDict.items():
            values[key] = val.get_state_value()
        print(values)
    

"""Examples:
   terminal = np.array([[0,0], [5,5]])  (always use 2D array)
   dimension = (7,7)
   g = Gridworld(dimension, terminal) 
   g.display_gridworld()                (initial random state)
   g.policy_iteration(dimension, 0.1)   
   g.display_gridworld()     
   
   terminal = np.array([[0,0]])  (always use 2D array)
   dimension = (5,5)
   g = Gridworld(dimension, terminal) 
   g.display_gridworld()                (initial random state)
   g.value_iteration(0.1)   
   g.display_gridworld() 
   
"""
         
    
