#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 23:56:11 2019

@author: abhigyan
"""

"""Program to find optimal State-Action pair value function and the optimal policy for
   an agent traversing a Windy Gridworld, with a per step penalty and a bonus on reaching
   a teminal desired state and penalty on reaching a terminal un-desired state"""

import numpy as np


"""Creates the Windy Gridworld Environment. The rguments required to create it are
   the m*n grid size, the winds passed as 2 lists, the target state and the false
   target state"""
class Windy_Gridworld(object):
    
    # Initializes the 2D size of the Gridworld
    # and the winds which is a list containing 2  
    # lists describing the winds along vertical
    # and horizontal from top and from left
    # respectively
    def __init__(self, gridSize, winds, target, falseTarget):
        self.target = target
        self.gridSize = gridSize
        self.falseTarget = falseTarget
        self.winds = {}
        self.terminated = 0
        for i in range(gridSize[0]):
            self.winds[i] = {}
            for j in range(gridSize[1]):
                self.winds[i][j] = np.array([winds[0][j], winds[1][i]])
                
    # Returns the new position as a result of action
    # taken by an agent. Returns the same position and 
    # 1 if the terminal target state is reached  
    # i.e target and false target, also returns the
    # same position if going out of the grid         
    def movement(self, initialPosition, agentMove):
        if(not self.terminated):
            finalPosition = initialPosition + self.winds[initialPosition[0]][initialPosition[1]] +\
                            + agentMove
           
            # Taking care if terminal state is reached
            if np.array_equal(self.target, finalPosition):
                self.terminated = 1
            elif np.array_equal(self.falseTarget, finalPosition):
                self.terminated = -1
            # Taking care the agent does not cross the wall
            if(finalPosition[0] < 0):
                finalPosition[0] = 0
            elif(finalPosition[0] >= self.gridSize[0]):
                finalPosition[0] = self.gridSize[0]-1
            if(finalPosition[1] < 0):
                finalPosition[1] = 0
            elif(finalPosition[1] >= self.gridSize[1]):
                finalPosition[1] = self.gridSize[1]-1
        else:
            finalPosition = initialPosition
        return finalPosition, self.terminated
    
    
        
    
"""Creates an agent with inherent properties of the world it is going to tarverse in,
   the Grid size of the worl, its starting state, the algorithm used for finding value
   function, step reward/penalty, positive reward on reaching desired terminal state,
   negaitive reward on reaching un-desired terminal state, exploration constant epsilon,
   and the initilaization algorithm of the state-action values"""
   
class Agent(object):
    
    def __init__(self, world, gridSize, initialPosition, samplingAlgorithm, stepReward,
                 positiveReward, negativeReward, epsilon, initialization = 'zeros'):
        self.world = world
        self.initialPosition = initialPosition
        self.samplingAlgorithm = samplingAlgorithm   # Monte Carlo or TD methods
        self.stepReward = stepReward                 # Reward/Penalty for each step
        self.positiveReward = positiveReward         # Reward if desired state us reached
        self.negativeReward = negativeReward         # Reward if non-desired state us reached
        self.epsilon = epsilon                       # Exploration constant
        self.movements = np.array([[0,1], [0,-1], [1,0], [-1,0], [0,0]])
        
        # Creating State action pair lookup tables
        self.stateActionPairs = {}
        for i in range(gridSize[0]):
            for j in range(gridSize[1]):
                self.stateActionPairs[(i,j)] = {}
                for k in self.movements:
                    self.stateActionPairs[(i,j)][tuple(k)] = {}
                    self.stateActionPairs[(i,j)][tuple(k)]['probability'] = 0.2
                    if(initialization == 'zeros'):
                        self.stateActionPairs[(i,j)][tuple(k)]['value'] = 0
                    elif(initialization == 'random'):
                        self.stateActionPairs[(i,j)][tuple(k)]['value'] = np.random.randint(10)
        #print(self.stateActionPairs)
                        
    
    """First Visit Monte Carlo Policy Evaluation Function Definition"""
    # Takes in Number of Episodes
    # Updates the Action-Value function of visited states
    def first_monte_carlo(self, numEpisodes):
        N = {}                                        # Maintain the total number of first visits in all episodes
        R = {}                                        # Maintain returns over all episodes for each state action pairs
        for i in self.stateActionPairs:
            N[i] = {}
            R[i] = {}
            for j in self.movements:
                N[i][tuple(j)] = 0
                R[i][tuple(j)] = 0
        
        # Collect returns and visits for each state-action for each episode
        for n in range(numEpisodes):
            self.world.terminated = 0                 # Set terminated to 0
            G = 0                                     # Accumulate returns over an episode
            start = np.copy(self.initialPosition)
            # Generate an episode
            episode = []
            while(True):
                pr = []
                # Get the policy from the table
                for i in self.movements:
                    pr.append(self.stateActionPairs[tuple(start)][tuple(i)]['probability'])
                # Select action, move to next state, append state-action to episode,
                # re-initialize start to current position, if terminal end episode
                action = np.random.choice(np.arange(0, 5), p=pr)
                finalPosition, terminated = self.world.movement(start, self.movements[action])
                episode.append([tuple(start), tuple(self.movements[action])])
                start = np.copy(finalPosition)
                if terminated:
                    episode.append([tuple(finalPosition), (0,0)])        # Adds the terminal state as state action pair
                    if(terminated == 1):
                        self.stateActionPairs[tuple(finalPosition)][(0,0)]['value'] = self.positiveReward
                    elif(terminated == -1):
                        self.stateActionPairs[tuple(finalPosition)][(0,0)]['value'] = self.negativeReward
                    break
                  
            # Calculate the Monte Carlo returns from the entire episode
            for j, e in reversed(list(enumerate(episode))):
                if(j == len(episode)-1):
                    G = self.stateActionPairs[e[0]][e[1]]['value']
                else:
                    G = self.gamma*G + self.stepReward
                if(e not in episode[:j]):
                    R[e[0]][e[1]] += G
                    N[e[0]][e[1]] += 1
            
        # Calculate and assign/update average expected rewards for each state action pair
        for q in self.stateActionPairs:
            for m in self.movements:
                if N[q][tuple(m)] != 0:
                    self.stateActionPairs[q][tuple(m)]['value'] = R[q][tuple(m)]/N[q][tuple(m)]
                    #print(self.stateActionPairs[q][tuple(m)]['value'])
                    #print()
            
    
    """Every Visit Monte Carlo Policy Evaluation Function Definition"""
    # Takes in Number of Episodes
    # Updates the Action-Value function of visited states
    def every_monte_carlo(self, numEpisodes):
        N = {}                                        # Maintain the total number of first visits in all episodes
        R = {}                                        # Maintain returns over all episodes for each state action pairs
        for i in self.stateActionPairs:
            N[i] = {}
            R[i] = {}
            for j in self.movements:
                N[i][tuple(j)] = 0
                R[i][tuple(j)] = 0
        
        # Collect returns and visits for each state-action for each episode
        for n in range(numEpisodes):
            self.world.terminated = 0                      # Set terminated to 0
            G = 0                                     # Accumulate returns over an episode
            start = np.copy(self.initialPosition)
            # Generate an episode
            episode = []
            while(True):
                pr = []
                # Get the policy from the table
                for i in self.movements:
                    pr.append(self.stateActionPairs[tuple(start)][tuple(i)]['probability'])
                # Select action, move to next state, append state-action to episode,
                # re-initialize start to current position, if terminal end episode
                action = np.random.choice(np.arange(0, 5), p=pr)
                finalPosition, terminated = self.world.movement(start, self.movements[action])
                episode.append([tuple(start), tuple(self.movements[action])])
                start = np.copy(finalPosition)
                if terminated:
                    episode.append([tuple(finalPosition), (0,0)])        # Adds the terminal state as state action pair
                    if(terminated == 1):
                        self.stateActionPairs[tuple(finalPosition)][(0,0)]['value'] = self.positiveReward
                    elif(terminated == -1):
                        self.stateActionPairs[tuple(finalPosition)][(0,0)]['value'] = self.negativeReward
                    break
            # Calculate the Monte Carlo returns from the entire episode
            for j, e in reversed(list(enumerate(episode))):
                if(j == len(episode)-1):
                    G = self.stateActionPairs[e[0]][e[1]]['value']
                else:
                    G = self.gamma*G + self.stepReward
                R[e[0]][e[1]] += G
                N[e[0]][e[1]] += 1
            
        # Calculate and assign/update average expected rewards for each state action pair
        for q in self.stateActionPairs:
            for m in self.movements:
                if N[q][tuple(m)] != 0:
                    self.stateActionPairs[q][tuple(m)]['value'] = R[q][tuple(m)]/N[q][tuple(m)]
    
   
   """TD(n) Policy Evaluation Function Definition"""
    def td_n(self, n, numEpisodes, alpha):         
       # Collect returns and visits for each state-action for each episode
        for j in range(numEpisodes):
            initialPosition = np.copy(self.initialPosition)
            while(True):
                self.world.terminated = 0                      # Set terminated to 0
                start = np.copy(initialPosition)
                
                # Generate a truncated episode
                truncEpisode = []
                for k in range(n+2): 
                    pr = []
                    # Get the policy from the table
                    for i in self.movements:
                        pr.append(self.stateActionPairs[tuple(start)][tuple(i)]['probability'])
                    # Select action, move to next state, append state-action to episode,
                    # re-initialize start to current position, if terminal end episode
                    action = np.random.choice(np.arange(0, 5), p=pr)
                    finalPosition, terminated = self.world.movement(start, self.movements[action])
                    truncEpisode.append([tuple(start), tuple(self.movements[action]), 0])
                    start = np.copy(finalPosition)
                    if terminated:
                        truncEpisode.append([tuple(finalPosition), (0,0), terminated])        # Adds the terminal state as state action pair
                        if(terminated == 1):
                            self.stateActionPairs[tuple(finalPosition)][(0,0)]['value'] = self.positiveReward
                        elif(terminated == -1):
                            self.stateActionPairs[tuple(finalPosition)][(0,0)]['value'] = self.negativeReward
                        break
                
                # Calculate the TD returns from the entire episode
                length = len(truncEpisode)
                newEstimate = (length-1)*self.stepReward + self.gamma*self.stateActionPairs[truncEpisode[-1][0]][truncEpisode[-1][1]]['value']
                oldEstimate = self.stateActionPairs[truncEpisode[0][0]][truncEpisode[0][1]]['value']
                self.stateActionPairs[truncEpisode[0][0]][truncEpisode[0][1]]['value'] += alpha*(newEstimate - oldEstimate)
                
                
                if(truncEpisode[0][1] == (0,0) and truncEpisode[1][2]):
                    #print(truncEpisode[0][0])
                    break
                initialPosition = np.copy(truncEpisode[1][0])
                            
    
    """Policy Improvement based on the GLIE technique"""        
    def glie_policy_improvement(self):
        greedyP = (1-self.epsilon) + self.epsilon/5
        exploreP = self.epsilon/5
        for i in self.stateActionPairs:
            value = -np.inf
            move = None
            for j in self.movements:
                if(self.stateActionPairs[i][tuple(j)]['value'] > value):
                    value = self.stateActionPairs[i][tuple(j)]['value']
                    move = j
            for j in self.movements:
                if(np.array_equal(j, move)):
                    self.stateActionPairs[i][tuple(j)]['probability'] = greedyP
                else:
                    self.stateActionPairs[i][tuple(j)]['probability'] = exploreP
        
    
    # Evaluate a policy using sampling methods like First Visit Monte Carlo
    # and Every Visit Monte Carlo                    
    def mc_policy_evaluation(self, numEpisodes):
        if(self.samplingAlgorithm == 'FMC'):
            self.first_monte_carlo(numEpisodes)
        elif(self.samplingAlgorithm == 'EMC'):
            self.every_monte_carlo(numEpisodes)
    
    
    # Find by Policy Iteration the value of State-Action pairs and also the optimal
    # policy using MC+GLIE methods
    def mc_policy_iteration(self, gamma, iterations, num_episodes):
        self.gamma = gamma
        for i in range(iterations):
            self.epsilon = self.epsilon/(i+1)
            self.policy_evaluation(num_episodes)
            self.glie_policy_improvement()
            
            
     # Evaluate a policy using TD sampling methods like 
    def td_policy_evaluation(self, n,  alpha, numEpisodes):
        self.td_n(n, numEpisodes, alpha)
            
            
    # Find by Policy Iteration the value of State-Action pairs and also the optimal
    # policy using TD+GLIE methods
    def td_policy_iteration(self, gamma, iterations, alpha, n, numEpisodes):
        self.gamma = gamma
        for i in range(iterations):
            print(i)
            print(self.epsilon)
            self.epsilon = self.epsilon/(i+1)
            self.td_policy_evaluation(n, alpha, numEpisodes)
            self.glie_policy_improvement() 
                
            
            
 
"""Method to create the Windy Gridworld"""       

def create_world(gridSize, winds, target, falseTarget):
    try:
        assert len(gridSize) == 2
    except AssertionError:
        print("Grid size is invalid !!!")
    
    try:
        assert len(winds) == 2
    except AssertionError:
        print("Grid size is invalid !!!")
        
    if not((np.array(gridSize) > target).all() and (np.array(gridSize) > falseTarget).all()):
        raise ValueError("Invalid targets !!!")
     
        
    longWinds = []
    for i in range(2):
        if(len(winds[i]) == 0):
            longWinds.append(gridSize[1-i]*[0])
        else:
            longWinds.append(winds[i])
    print(longWinds)
    world = Windy_Gridworld(gridSize, longWinds, target, falseTarget)
    
    return world
        

"""Method to create the Agent"""

def create_agent(world, gridSize, initialPosition, samplingAlgorithm, stepReward,
                 positiveReward, negativeReward, epsilon, initialization = 'zeros'):
    agent = Agent(world, gridSize, initialPosition, samplingAlgorithm, stepReward,
                 positiveReward, negativeReward, epsilon, initialization = 'zeros')
    return agent




# Examples:
# 1. Without Wind

"""target = np.array([1,4])
falseTarget = np.array([0,0])
gridSize = [6,6]
initialPosition = np.array([3,0])
reward = 15
penalty = -5
stepPenalty = -1
epsilon = 0.2
iterations = 10
episodes = 2000
gamma = 0.95
world = create_world(gridSize, [[], []], target, falseTarget)
agent = create_agent(world, gridSize, initialPosition, 'FMC', stepPenalty, 
                     reward, penalty, epsilon)
agent.policy_iteration(gamma, iterations, episodes)"""

# 2. With Wind

"""target = np.array([1,4])
falseTarget = np.array([0,0])
gridSize = [6,6]
initialPosition = np.array([3,0])
reward = 15
penalty = -5
stepPenalty = -1
epsilon = 0.2
iterations = 10
episodes = 2000
gamma = 0.95
world = create_world(gridSize, [[0, 1, 1, 0, 0, 0], []], target, falseTarget)
agent = create_agent(world, gridSize, initialPosition, 'EMC', stepPenalty, 
                     reward, penalty, epsilon)
agent.mc_policy_iteration(gamma, iterations, episodes)"""     


# Examples (TD):
#1. Without Wind

"""
target = np.array([1,4])
falseTarget = np.array([0,0])
gridSize = [6,6]
initialPosition = np.array([3,0])
reward = 15
penalty = -5
stepPenalty = -1
epsilon = 0.2
iterations = 10
numEpisodes = 500
gamma = 0.95
alpha = 0.4
n = 0
world = create_world(gridSize, [[], []], target, falseTarget)
agent = create_agent(world, gridSize, initialPosition, 'FMC', stepPenalty, 
                     reward, penalty, epsilon)
agent.td_policy_iteration(gamma, iterations, alpha, n, numEpisodes)"""

#2. With Wind and Td(n) where n=2

"""
target = np.array([1,4])
falseTarget = np.array([0,0])
gridSize = [6,6]
initialPosition = np.array([3,0])
reward = 15
penalty = -5
stepPenalty = -1
epsilon = 0.2
iterations = 10
numEpisodes = 200
gamma = 0.95
alpha = 0.4
n = 2
world = create_world(gridSize, [[0, 1, 1, 0, 0, 0], []], target, falseTarget)
agent = create_agent(world, gridSize, initialPosition, 'FMC', stepPenalty, 
                     reward, penalty, epsilon)
agent.td_policy_iteration(gamma, iterations, alpha, n, numEpisodes)"""
