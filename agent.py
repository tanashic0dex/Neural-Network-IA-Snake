import numpy as np
from random import random, randint
from collections import deque

from q_network import QNetwork 
from memory import ReplayMemory

class DeepQ_agent:
    """
    Represents the DQN agent.
    """
    def __init__(self, env, hidden_units = None, network_LR=0.01, batch_size=1024, update_every=5, gamma=0.95):
        """
        Creates a DQN agent.

        :param env: game environment.
        :type env: Class Snake_Env().
        :param hidden_units: number of neurons in each layer.
        :type hidden_units: tupple with dimension (1, 3).
        :param network_LR: learning rate of the action-value neural network.
        :type network_LR: float.
        :param batch_size: size of the minibatch taken from the replay buffer.
        :type batch_size: int.
        :param update_every: number of iterations for updating the target qnetwork. 
        :type update_every: int
        :param gamma: discount factor.
        :type gamma: float.
        """
        self.env = env
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma          
        self.NETWORK_LR = network_LR
        self.MEMORY_CAPACITY = int(1e5)   
        self.ACTION_SIZE = env.ACTION_SPACE           
        self.HIDDEN_UNITS = hidden_units
        self.UPDATE_EVERY = update_every
       
        self.qnetwork_local = QNetwork(input_shape = self.env.STATE_SPACE,
                                        hidden_units = self.HIDDEN_UNITS,
                                        output_size = self.ACTION_SIZE,
                                        learning_rate = self.NETWORK_LR)
        
        self.qnetwork_target = QNetwork(input_shape = self.env.STATE_SPACE,
                                        hidden_units = self.HIDDEN_UNITS,
                                        output_size = self.ACTION_SIZE,
                                        learning_rate = self.NETWORK_LR)

        self.memory = ReplayMemory(self.MEMORY_CAPACITY, self.BATCH_SIZE) 

        #Temp variable
        self.t = 0


    def learn(self):
        """
        Learn from memorized experience.
        """
        if self.memory.__len__() > self.BATCH_SIZE:
            states, actions, rewards, next_states, dones = self.memory.sample(self.env.STATE_SPACE)
            
            #Calculating action-values using local network
            target = self.qnetwork_local.predict(states, self.BATCH_SIZE)
            
            #Future action-values using target network
            target_val = self.qnetwork_target.predict(next_states, self.BATCH_SIZE)
            
            #Future action-values using local network
            target_next = self.qnetwork_local.predict(next_states, self.BATCH_SIZE)
        
            max_action_values = np.argmax(target_next, axis=1)   #action selection
            
            for i in range(self.BATCH_SIZE):
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    target[i][actions[i]] = rewards[i] + self.GAMMA*target_val[i][max_action_values[i]]   #action evaluation
            
            self.qnetwork_local.train(states, target, batch_size = self.BATCH_SIZE)

            if self.t == self.UPDATE_EVERY:
                self.update_target_weights()
                self.t = 0
            else:
                self.t += 1


    def act(self, state, epsilon=0.0):
        """
        Chooses an action using an epsilon-greedy policy.
        
        :param state: current state.
        :type state: NumPy array with dimension (1, 18).
        :param epsilon: epsilon used in epsilon-greedy policy.
        :type epsilon: float
        :return action: action chosen by the agent.
        :rtype: int
        """    
        state = state.reshape((1,)+state.shape)
        action_values = self.qnetwork_local.predict(state)    #returns a vector of size = self.ACTION_SIZE
        if random() > epsilon:
            action = np.argmax(action_values)                 #choose best action - Exploitation
        else:
            action = randint(0, self.ACTION_SIZE-1)           #choose random action - Exploration
        return action


    def add_experience(self, state, action, reward, next_state, done):
        """
        Add experience to agent's memory.
        """
        self.memory.add(state, action, reward, next_state, done)

    
    def update_target_weights(self):
        """
        Updates values of the Target network.
        """
        self.qnetwork_target.model.set_weights(self.qnetwork_local.model.get_weights())