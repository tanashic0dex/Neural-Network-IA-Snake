import numpy as np
from collections import deque
from random import sample


class ReplayMemory:
    """
    Represents the memory of the agent. Stores info while training.
    """
    def __init__(self, buffer_size, batch_size):
        """
        Memory params.

        :param buffer_size: size of the experience replay buffer.
        :type buffer_size: int. 
        :param batch_size: size of the minibatch taken from the replay buffer.
        :type batch_size: int.
        """
        self.memory = deque(maxlen=buffer_size)
        self.BATCH_SIZE = batch_size
        
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.

        :param state: current state.
        :type state: NumPy array with dimension (1, 18).
        :param action: action chosen by the agent.
        :type action: int.
        :param reward: iteration reward.
        :type reward: float.
        :param next_state: next state.
        :type next_state: NumPy array with dimension (1, 18).
        :param done: if the snake died in this iteration.
        :type done: boolean.
        """
        e = tuple((state, action, reward, next_state, done))
        self.memory.append(e)
        
    
    def sample(self, state_shape):
        """
        Randomly sample a batch of experiences from memory.

        :param state_shape: state size.
        :type state_shape: int.
        """
        experiences = sample(self.memory, k=self.BATCH_SIZE)
        #Extracting the SARSA
        states, actions, rewards, next_states, dones = zip(*experiences)
 
        #Converting them to numpy arrays
        states = np.array(states).reshape(self.BATCH_SIZE, state_shape)
        actions = np.array(actions, dtype='int').reshape(self.BATCH_SIZE)
        rewards = np.array(rewards).reshape(self.BATCH_SIZE)
        next_states = np.array(next_states).reshape(self.BATCH_SIZE,state_shape)
        dones = np.array(dones).reshape(self.BATCH_SIZE)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):     
        """
        Returns the len of the memory stored.
        """
        return len(self.memory)
