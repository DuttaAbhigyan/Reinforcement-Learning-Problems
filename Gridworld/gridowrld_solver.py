#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:33:56 2019

@author: abhigyan
"""
import numpy as np


"""Class state to initialize various states in a Gridworld (the various positions 
   at which a users might find themselves). It has attributes like position, state
   type (transient/recurrent as a boolean value for recurrent), its inital value,
   immediate action reward.
   This class is controlled by the Gridworld class
"""

class state(object):
    
    #Initializes a state with its position, type = transient/terminal, reward due to immediate
    #action, maximum rows and maximum columns a Gridworld board can have
    def __init__(self, position, terminal, value, actionReward, max_x, max_y):
        self.position = position
        self.terminal = terminal
        self.stateValue = value
        self.actionReward = actionReward
        self.mins = 0,0
        self.maxs = max_x, max_y              #max_x = maximum rows, max_y = maximum columns
        
    
    #Enumertates all the possible moves from a state if hitting a wall it returns to the same state (enumerated explicitly)    
    def enumerateNextStates(self, probability = 'stochastic'):
        self.possibleMoves = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        
        if(self.terminal):
            self.nextStates = np.array([self.position])
            self.actionReward = 0
        else:
            self.theoreticalNextStates = self.possibleMoves + self.position
            self.nextStates = self.theoreticalNextStates[(self.theoreticalNextStates >= self.mins).all(axis = 1) &
                                                         (self.theoreticalNextStates <= self.maxs).all(axis = 1)]
            if(len(self.nextStates) < 4 and len(self.nextStates) > 1):       #Adding the wall hits as return to the same state explicilty
                selfPosition = np.tile(self.position, (4-len(self.nextStates), 1))
                self.nextStates = np.concatenate((self.nextStates, selfPosition), axis = 0)
            
        
        self.numberOfNextStates = len(self.nextStates)       
        if(probability == 'random'):                                        #Assigining a random policy
            self.transitionProbabilities = np.random.randint(low = 0, high = 50, 
                                                             size = (self.numberOfNextStates))
            self.transitionProbabilities = self.transitionProbabilities / sum(self.transitionProbabilities)
        elif(probability == 'stochastic'):                                  #Assigining a uniform stochastic policy
            self.transitionProbabilities = 1 / self.numberOfNextStates * np.ones(self.numberOfNextStates)

        
    #Links the state classes to a state, including reching itself by hitting a wall
    def linkStates(self, states):
        self.nextStateList = []
        count = 0
        
        for i in self.nextStates:  
            entry = str(i[0]) + ',' + str(i[1])
            self.nextStateList.append([states[entry], self.transitionProbabilities[count]])
            count += 1
            
    
    #Acts greedily in a current step at a current iteration
    def actGreedily(self):
        self.newBestValue = -np.inf
        
        for i in self.nextStateList:
            nextValue = self.actionReward + i[0].getStateValue()
            if(nextValue > self.newBestValue):
                self.newBestValue = nextValue
                
        
    #Evaluates a given policy            
    def evaluatePolicy(self): 
        self.expectedValue = 0
        
        for i in self.nextStateList:
            self.expectedValue += i[1] * (self.actionReward + i[0].getStateValue())
            
    
    #Improves the policy by changing state transition probabilities
    def improvePolicy(self):
        self.newBestValue = -np.inf
        self.nextBestState = 0
        index = 0
        
        for i in self.nextStateList:
            nextValue = self.actionReward + i[0].getStateValue()
            if(nextValue > self.newBestValue):
                self.newBestValue = nextValue
                index = self.nextStateList.index(i)
                
        for i in range(0, len(self.nextStateList)):
            if(i == index):
                self.nextStateList[i][1] = 1
            else:
                self.nextStateList[i][1] = 0
        
        
        
    
    #Updates the state value function for greedy policy   
    def updateGreedyAct(self):
        self.stateValue = self.newBestValue
    
    #Updates the state value function for stochastic policy    
    def updatePolicyAct(self):
        self.stateValue = self.expectedValue
        
    
    #Getter methods        
    def getNextStates(self):
        return self.nextStateList
            
    def getStateValue(self):
        return self.stateValue
        
        
    
        
"""The Gridworld class contains all the states a user can begin in Gridworld and is
   responsible for finding the optimal value function v_* via 2 methods:
   -> Policy Iteration
   -> Value Iteration
"""

class Gridworld(object):
    
    #Initializes a Gridowrld board with its dimensions, winning states = terminal states,
    #an immediate action rewards which a user can get on traversing the board
    def __init__(self, dimensions, terminalStates, immediateRewards):        
        self.dimensions = dimensions
        self.terminalStates = terminalStates
        self.numberOfStates = dimensions[0] * dimensions[1]
        self.immediateRewards = immediateRewards
        self.valueFunction = np.zeros(self.dimensions)      #Value function representation for different states, initialized to 0
    
    #Creates the board by initializing various states and linking them to proper
    #reachable states    
    def createBoard(self):
        self.states = {}
        
        for i in range(0, self.dimensions[0]):
            for j in range(0, self.dimensions[1]):
                position = np.array([i, j])
                terminal = (np.any(np.equal(self.terminalStates, position).all(axis=1)))
                self.states[str(i)+','+str(j)] = state(position, terminal, 0, self.immediateRewards, 
                                                       self.dimensions[0]-1, self.dimensions[1]-1)
                self.states[str(i)+','+str(j)].enumerateNextStates()
        
        for i in self.states:
            self.states[i].linkStates(self.states)
            
    
    #Act Greedily in a current iteration and update the value function of the states        
    def actGreedily(self):
        for i in self.states:
            self.states[i].actGreedily()
        for i in self.states:
            self.states[i].updateGreedyAct()
        self.updateValueFunction()

    
    #Policy Evaluation
    def policyEvaluation(self):
        stable = False
        
        while(stable == False):
            oldValueFunction = np.copy(self.valueFunction)
            for i in self.states:
                self.states[i].evaluatePolicy()
            for i in self.states:
                self.states[i].updatePolicyAct()
            self.updateValueFunction()
            if(np.sum(np.abs(self.valueFunction - oldValueFunction)) <= 10 ** -10):
                stable = True
                
    #Improves a policy by acting greedily w.r.t current policy            
    def policyImprovement(self):
        for i in self.states:
            self.states[i].improvePolicy()

    
    #Updates the value function represenation of the board (making the values equal to the value function of the states)       
    def updateValueFunction(self):     
        for i in range(0, self.dimensions[0]):
            for j in range(0, self.dimensions[1]):
                self.valueFunction[i,j] = self.states[str(i) + ',' + str(j)].getStateValue()
            
    
    
    #Solving by Value Iteration      
    def valueIteration(self):
        stable = False
        
        while(stable == False):
            oldValueFunction = np.copy(self.valueFunction)
            self.actGreedily()
            self.updateValueFunction()
            if(np.sum(np.abs(self.valueFunction - oldValueFunction)) <= 10 ** -10):
                stable = True
            
    
    #Solving by Policy Iteration
    def policyIteration(self):
        stable = False
        
        while(stable == False):
            oldValueFunction = np.copy(self.valueFunction)
            self.policyEvaluation()
            self.policyImprovement()
            self.updateValueFunction()
            if(np.sum(np.abs(self.valueFunction - oldValueFunction)) <= 10 ** -10):
                stable = True
            
    
    #Getter methods
    def getValueFunction(self):
        return self.valueFunction
            
                
 
"""Examples:
   1) Board Size = 3,3 , Reward State at 1,1 , Each action has reward -1, solved by Policy Iteration
                    
g = Gridworld((3,3), np.array([[0,0]]), -1)
g.createBoard()  
g.policyIteration() 
print(g.getValueFunction())   


2) Board Size = 5,5 , Reward State at (1,1) and (3,3) , Each action has reward -1, solved by Value Iteration
                    
g = Gridworld((5,5), np.array([[0,0], [2,2]]), -1)
g.createBoard()  
g.valueIteration() 
print(g.getValueFunction())
"""       
            
        
