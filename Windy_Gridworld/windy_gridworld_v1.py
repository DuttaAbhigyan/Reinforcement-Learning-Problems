#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:08:06 2019

@author: abhigyan
"""

"""Program to find optimal State-Action pair value function and the optimal policy for
   an agent traversing a StochasticWindy Gridworld, with a per step penalty and a 
   bonus on reaching a teminal desired state and penalty on reaching a terminal 
   un-desired state. The stochastcity is in the form of a wind i.e. the wind will be
   generated by a Stochastic Process whose properties depend on (x,y)"""
   
   
import numpy as np
from math import *

"""Creates a Wind Generator object with different Stochastic processes and
   different probability distributions. New processes and probability distributions
   can easily be added"""
class Wind_Generator(object):
    
    # Initializes the process to be used to generate the parameters of a PDF, and
    # the PDF to be used
    def __init__(self, process, pdf):
        self.process = getattr(Wind_Generator, process)
        self.pdf = getattr(Wind_Generator, pdf)
        
    
    """Generator Function to be used by the Windy Gridworld Environment"""
    # Generates a (2,) NumPy array containing the wind at a particular co-ordinate
    # in the form of (wind in verical, wind in horizontal)     
    def generate(self, position):
        params = self.process(self, position)
        wind = self.pdf(self, params)
        return wind
    
    
    """Processes to generate parameters for a PDF, can be 1 or 2 paramaters for 
       each direction (i.e vertical and horizontal"""
    # Sample function generating a 0 which when supplied to a PDF should produce 0
    def process0(self, position):
        return 0
    
    # Sample function generating a single value (lambda) for Poisson/Exponential PDF
    def process1(self, position):
        vertical = abs(position[0] - position[1])
        horizontal = abs(position[0] - position[1])
        return np.array([vertical,horizontal])
    
    # Sample function generating a double value (mu, varince) for Gaussian PDF
    def process2(self, position):
        vertical_mu = 1                  # Larger means might result in agent not reaching goal
        vertical_var = abs(position[0]**2 - position[1]**2)/abs(2*(position[0] - position[1]) + 1)**2
        horizontal_mu = 1
        horizontal_var = abs(position[1]**2 - position[0]**2)/abs(2*(position[0] - position[1]) + 1)**2
        return np.array([[vertical_mu, vertical_var], [horizontal_mu, horizontal_var]])
    
    # Add more processes here on..........
    
    
    """PDF's to generate values"""
    # Sampling values according to Poisson PDF
    def poisson(self, params):
        try:
            vertical_w = np.random.poisson(lam=params[0], size=(1,))
            horizontal_w = np.random.poisson(lam=params[1], size=(1,))
            winds = np.array([vertical_w, horizontal_w]).reshape((2,))
        except:
            winds = np.zeros((2,))
        return winds
    
    # Sampling values according to Gaussian PDF
    def gaussian(self, params):
        try:
            vertical_w = np.round(np.random.normal(loc=params[0][0], scale=params[0][1], size=(1,)))
            horizontal_w = np.round(np.random.normal(loc=params[1][0], scale=params[1][1],size=(1,)))
            winds = np.array([vertical_w, horizontal_w]).reshape((2,))
        except:
            winds = np.zeros((2,))
        return winds
    
    # Sampling values according to Exponential PDF
    def exponential(self, params):
        try:
            vertical_w = np.round(np.random.exponential(scale=params[0], size=(1,)))
            horizontal_w = np.round(np.random.exponential(scale=params[1], size=(1,)))
            winds = np.array([vertical_w, horizontal_w]).reshape((2,))
        except:
            winds = np.zeros((2,))
        return winds
    
    # Add more PDF's here on.............




"""Creates the Windy Gridworld Environment. The rguments required to create it are
   the m*n grid size, the wind is generated by a function which will be passed as an 
   argument."""
class Windy_Gridworld(object):
    
    # Initializes the 2D size of the Gridworld
    # and the winds which is a list containing 2  
    # lists describing the winds along vertical
    # and horizontal from top and from left
    # respectively
    def __init__(self, gridSize, windGenerator, realTarget, falseTarget):
        self.realTarget = np.array(realTarget)
        self.falseTarget = np.array(falseTarget)
        self.gridSize = gridSize
        self.windGenerator = windGenerator
        
                
    # Returns the new position as a result of action
    # taken by an agent. Returns the same position and 
    # 1 if the terminal target state is reached  
    # i.e target and false target, also returns the
    # same position if going out of the grid         
    def movement(self, initialPosition, agentMove):
        terminated = 0
        
        # Taking care if terminal state is reached
        if np.array_equal(self.realTarget, initialPosition):
            terminated = 1
        elif np.array_equal(self.falseTarget, initialPosition):
            terminated = -1
        if(terminated):
            return initialPosition, terminated        
        
        winds = self.windGenerator.generate(initialPosition)
        finalPosition = np.array(initialPosition) + np.array(agentMove) + winds
        
        # Taking care the agent does not cross the wall
        if(finalPosition[0] < 0):
            finalPosition[0] = 0
        elif(finalPosition[0] >= self.gridSize[0]):
            finalPosition[0] = self.gridSize[0]-1
        if(finalPosition[1] < 0):
            finalPosition[1] = 0
        elif(finalPosition[1] >= self.gridSize[1]):
            finalPosition[1] = self.gridSize[1]-1
       
        if np.array_equal(self.realTarget, finalPosition):
            terminated = 1
        elif np.array_equal(self.falseTarget, finalPosition):
            terminated = -1
         
        return tuple(finalPosition), terminated
    




"""Creates an agent with some inherent properties of the world. These are step 
   cost, reward on reaching final target state, reward on reaching final false
   state, initialization of Q(s,a) values (NOTE: We are using tabluar methods).
   Possible moves by the agent in each state are: Left, Right, Up, Down, Stay"""

class Agent(object):
    
    # Initializes the agent with related inherent properties (arbitrarily decided),
    # and Lookup Table for Q(s,a)
    def __init__(self, world, initialPosition, stepCost, positiveReward, negativeReward, 
                 gamma=0.95, initialization = 'zeros'):
        self.world = world
        gridSize = self.world.gridSize
        self.initialPosition = initialPosition
        self.stepCost = stepCost                     # Reward/Penalty for each step
        self.positiveReward = positiveReward         # Reward if desired state us reached
        self.negativeReward = negativeReward         # Reward if non-desired state us reached
        self.gamma = gamma                           # Patience of the agent
        self.movements = ((0,1), (0,-1), (1, 0), (-1,0), (0,0))
        
        # Creating State action pair lookup tables
        self.stateActionPairs = {}
        for i in range(gridSize[0]):
            for j in range(gridSize[1]):
                self.stateActionPairs[(i,j)] = {}
                for k in self.movements:
                    self.stateActionPairs[(i,j)][k] = {}
                    self.stateActionPairs[(i,j)][k]['probability'] = 0.2
                    if(initialization == 'zeros'):
                        self.stateActionPairs[(i,j)][k]['value'] = 0
                    elif(initialization == 'random'):
                        self.stateActionPairs[(i,j)][k]['value'] = np.random.randint(10)
                        
                        
    
    # Algoirthms for Generalized Policy Evaluation
    
    """First Visit Monte Carlo Policy Evaluation Function Definition"""
    # Takes in Number of Episodes
    # Updates the Action-Value function of visited states.
    # Updates using Incremental Mean
    def first_monte_carlo(self, numEpisodes, n = None, alpha=None, lam=None):
        self.N = {}                                        # Maintain the total number of first visits in all episodes
        for i in self.stateActionPairs:
            self.N[i] = {}
            for j in self.movements:
                self.N[i][j] = 0
        
        # Collect returns and visits for each state-action for each episode
        for n in range(numEpisodes):
            G = 0                                     # Accumulate returns over an episode
            start = self.initialPosition     # Set starting state
            # Generate an episode
            episode = []
            while(True):
                pr = []
                # Get the policy from the table
                for i in self.movements:
                    pr.append(self.stateActionPairs[start][i]['probability'])
                # Select action, move to next state, append (S,A) to episode,
                # re-initialize start to current position, if terminal end episode
                # else continue
                action = np.random.choice(np.arange(0, 5), p=pr)
                finalPosition, terminated = self.world.movement(start, self.movements[action])
                episode.append([start, self.movements[action]])
                if terminated:
                    episode.append([finalPosition, (0,0)])        # Adds the terminal state as state action pair
                    if(terminated == 1):
                        self.stateActionPairs[finalPosition][(0,0)]['value'] = self.positiveReward
                    elif(terminated == -1):
                        self.stateActionPairs[finalPosition][(0,0)]['value'] = self.negativeReward
                    break    
                start = finalPosition
    
            # Calculate the returns from the entire episode
            for j, e in reversed(list(enumerate(episode))):
                if(j == len(episode)-1):
                    G = self.stateActionPairs[e[0]][e[1]]['value']
                else:
                    G = self.gamma*G + self.stepCost
                if(e not in episode[:j]):
                    self.N[e[0]][e[1]] += 1
                    # Incremental update to mean
                    self.stateActionPairs[e[0]][e[1]]['value'] += (1/self.N[e[0]][e[1]])* \
                                                                  (G - self.stateActionPairs[e[0]][e[1]]['value'])
                     
    
   """Every Visit Monte Carlo Policy Evaluation Function Definition"""
    # Takes in Number of Episodes
    # Updates the Action-Value function of visited states.
    # Updates using Incremental Mean
    def every_monte_carlo(self, numEpisodes, n = None, alpha=None, lam=None):
        self.N = {}                                        # Maintain the total number of first visits in all episodes
        for i in self.stateActionPairs:
            self.N[i] = {}
            for j in self.movements:
                self.N[i][j] = 0
        
        # Collect returns and visits for each state-action for each episode
        for n in range(numEpisodes):
            G = 0                                     # Accumulate returns over an episode
            start = self.initialPosition     # Set starting state
            # Generate an episode
            episode = []
            while(True):
                pr = []
                # Get the policy from the table
                for i in self.movements:
                    pr.append(self.stateActionPairs[start][i]['probability'])
                # Select action, move to next state, append (S,A) to episode,
                # re-initialize start to current position, if terminal end episode
                # else continue
                action = np.random.choice(np.arange(0, 5), p=pr)
                finalPosition, terminated = self.world.movement(start, self.movements[action])
                episode.append([start, self.movements[action]])
                if terminated:
                    episode.append([finalPosition, (0,0)])        # Adds the terminal state as state action pair
                    if(terminated == 1):
                        self.stateActionPairs[finalPosition][(0,0)]['value'] = self.positiveReward
                    elif(terminated == -1):
                        self.stateActionPairs[finalPosition][(0,0)]['value'] = self.negativeReward
                    break    
                start = finalPosition
    
            # Calculate the returns from the entire episode
            for j, e in reversed(list(enumerate(episode))):
                if(j == len(episode)-1):
                    G = self.stateActionPairs[e[0]][e[1]]['value']
                else:
                    G = self.gamma*G + self.stepCost
                self.N[e[0]][e[1]] += 1
                # Incremental update to mean
                self.stateActionPairs[e[0]][e[1]]['value'] += (1/self.N[e[0]][e[1]])* \
                                                              (G - self.stateActionPairs[e[0]][e[1]]['value'])
               
    
   """TD(n) On-Policy Evaluation Function Definition"""
    # Takes in Number of Episodes
    # Updates the Action-Value function of visited states
    def td_n_on(self, numEpisodes, n, alpha, lam=None):
        self.N = {}                                        # Maintain the total number of first visits in all episodes
        for i in self.stateActionPairs:                    # for GLIE policy improvement
            self.N[i] = {}
            for j in self.movements:
                self.N[i][j] = 0 
                
        # Repeat TD(n) learning for each epsiode
        for j in range(numEpisodes):
            start = self.initialPosition
            pr = []
            # Get the policy from the table and take step S_t, A_t
            for i in self.movements:
                pr.append(self.stateActionPairs[start][i]['probability'])
            # Select action, move to next state
            action1 = self.movements[np.random.choice(np.arange(0, 5), p=pr)]
            self.N[start][action1] += 1
            while True:
                flag = 0
                truncEpisode = []
                truncEpisode.append([start, action1])
                for i in range(n+1):
                    finalPosition, terminated = self.world.movement(start, action1)
                    if terminated:
                        # Check whether the starting state is the terminal state itself
                        if start == finalPosition:
                            flag = 1 
                        truncEpisode.append([finalPosition, (0,0)])        # Adds the terminal state as state action pair
                        if(terminated == 1):
                            self.stateActionPairs[finalPosition][(0,0)]['value'] = self.positiveReward
                        elif(terminated == -1):
                            self.stateActionPairs[finalPosition][(0,0)]['value'] = self.negativeReward
                        break
                    else:
                        pr = []
                        for i in self.movements:
                            pr.append(self.stateActionPairs[finalPosition][i]['probability'])
                        # Select action, move to next state
                        action2 = self.movements[np.random.choice(np.arange(0, 5), p=pr)]
                        truncEpisode.append([finalPosition, action2])
                        start = finalPosition
                        action1 = action2
                
                # Trajectory complete
                if flag:
                    break

                # Calculating estimates for trajectory of length 'n'
                l = len(truncEpisode)
                gammaGP = (self.gamma**(l-1) - 1) / (self.gamma - 1)   # GP of gammas to be multiplied with step reward
                gammaN = self.gamma**(l-1)                             # final gamma
                oldEstimate = self.stateActionPairs[truncEpisode[0][0]][truncEpisode[0][1]]['value']
                newEstimate = self.stepCost*gammaGP + gammaN*self.stateActionPairs[truncEpisode[-1][0]][truncEpisode[-1][1]]['value']
                self.stateActionPairs[truncEpisode[0][0]][truncEpisode[0][1]]['value'] += alpha*(newEstimate - oldEstimate)
                start = truncEpisode[1][0]
                action1 = truncEpisode[1][1]
                self.N[start][action1] += 1
                  
    
   """SARSA(lambda) Policy Evaluation Function Definition"""
    # Takes in Number of Episodes
    # Updates the Action-Value function of visited states (backward view)            
    def sarsa_lambda(self, numEpisodes, n, alpha, lam):
        self.N = {}                                        # Maintain the total number of first visits in all episodes
        for i in self.stateActionPairs:                    # for GLIE policy improvement
            self.N[i] = {}
            for j in self.movements:
                self.N[i][j] = 0 
                
        e = {}           # Maintain eligibility traces
        
        # Collect errors, eligibility and returns for each state-action for each episode
        for j in range(numEpisodes):
            start = self.initialPosition
            pr = []
            # Get the policy from the table and take step S_t, A_t
            for i in self.movements:
                pr.append(self.stateActionPairs[start][i]['probability'])
            # Select action, move to next state
            action1 = self.movements[np.random.choice(np.arange(0, 5), p=pr)]
            while(True):
                finalPosition, terminated = self.world.movement(start, action1)
                # Record new states in the eligiility trace
                if (start, action1) not in e:
                    e[start, action1] = 1
                    self.N[start][action1] = 1
                else:
                    e[start, action1] += 1
                    self.N[start][action1] += 1
                # Get the policy from the table and take step S_t+1, A_t+1
                if terminated:
                    action2 = (0,0)
                    if terminated == 1:
                        self.stateActionPairs[finalPosition][action2]['value'] = self.positiveReward
                    elif terminated == -1:
                        self.stateActionPairs[finalPosition][action2]['value'] = self.negativeReward
                else:
                    pr = []
                    for i in self.movements:
                        pr.append(self.stateActionPairs[finalPosition][i]['probability'])
                    # Select action, move to next state
                    action2 = self.movements[np.random.choice(np.arange(0, 5), p=pr)]
                
                delta = self.stepCost + self.gamma*self.stateActionPairs[finalPosition][action2]['value'] - \
                        self.stateActionPairs[start][action1]['value']
                
                for i in e:
                    self.stateActionPairs[i[0]][i[1]]['value'] += alpha*delta*e[i]
                    e[i] = self.gamma*lam*e[i]
                
                if terminated:
                    break
                else:
                    start = finalPosition
                    action1 = action2

                  
    """TD(0) Off-Policy Evaluation Function Definition"""
    # Takes in Number of Episodes
    # Updates the Action-Value function of visited states
    def td_off(self, numEpisodes, n, alpha, lam=None):
        self.N = {}                                        # Maintain the total number of first visits in all episodes
        for i in self.stateActionPairs:                    # for GLIE policy improvement
            self.N[i] = {}
            for j in self.movements:
                self.N[i][j] = 0 
                
        # Repeat TD(0) learning for each epsiode
        for j in range(numEpisodes):
            start = self.initialPosition
            while True:
                pr = []
                # Get the policy from the table and take step S_t, A_t
                for i in self.movements:
                    pr.append(self.stateActionPairs[start][i]['probability'])
                # Select action, move to next state
                action1 = self.movements[np.random.choice(np.arange(0, 5), p=pr)]
                self.N[start][action1] += 1
                truncEpisode = []
                truncEpisode.append([start, action1])
                finalPosition, terminated = self.world.movement(start, action1)
                if terminated:
                    truncEpisode.append([finalPosition, (0,0)])        # Adds the terminal state as state action pair (greedy action)
                    if(terminated == 1):
                        self.stateActionPairs[finalPosition][(0,0)]['value'] = self.positiveReward
                        greedyValue = self.positiveReward
                    elif(terminated == -1):
                        self.stateActionPairs[finalPosition][(0,0)]['value'] = self.negativeReward
                        greedyValue = self.negativeReward
                else:
                    # Select greedy action of next state
                    greedyValue = -np.inf
                    for m in self.stateActionPairs[finalPosition]:
                        if(greedyValue < self.stateActionPairs[finalPosition][m]['value']):
                            greedyValue = self.stateActionPairs[finalPosition][m]['value']
                    
                oldValue = self.stateActionPairs[start][action1]['value'] 
                greedyValue = self.stepCost + self.gamma*greedyValue
                self.stateActionPairs[start][action1]['value']  += alpha*(greedyValue-oldValue)
                
                # Trajectory complete
                if terminated:
                    break
                start = finalPosition
            
                 
               
    
    """GLIE Policy Improvement"""
    # Takes in visits to epsilon and visits to state and updates the policy to 
    # an epsilon soft policy based on 1/k decay
    def glie_policy_improvement(self, epsilon):
        visits = {}
        for i in self.N:
            # Check if the state is visited
            if i not in visits:
                visits[i] = 1
            
        for i in visits:
            nVisit = -1
            value = -np.inf
            move = None
            for j in self.movements:
                if(self.stateActionPairs[i][j]['value'] > value):
                    value = self.stateActionPairs[i][j]['value']
                    move = j
                if(np.round(epsilon/self.stateActionPairs[i][j]['probability']) > nVisit):
                    # -3 So that in the first iteration the number of visits is counted as 2
                    # as initial value of 'probability' = 2
                    nVisit = np.round(epsilon/self.stateActionPairs[i][j]['probability']+1)  
                
            # Exploration constant epsilon/|A(s)|
            exploreP = epsilon/(nVisit)
            greedyP = (1 - 5*exploreP) + exploreP
            for j in self.movements:
                if(np.array_equal(j, move)):
                    self.stateActionPairs[i][j]['probability'] = greedyP
                else:
                    self.stateActionPairs[i][j]['probability'] = exploreP
        

    """Generalized Policy Iteration step, takes in the policy evaluation function,
       the policy improvement function, alpha if required, epsilon if required"""
    def generalized_policy_iteration(self, iterations, evaluationFunction, numEpisodes, 
                                     improvementFunction, n = None, alpha=None, epsilon=None,
                                     lam=None):
    
        evaluationFunction = getattr(Agent, evaluationFunction)
        improvementFunction = getattr(Agent, improvementFunction)
        
        for i in range(iterations):
            print('Iteration: ' + str(i+1))
            evaluationFunction(self, numEpisodes, n, alpha, lam)
            improvementFunction(self, epsilon)
    
