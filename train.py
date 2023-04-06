import os
import matplotlib.pyplot as plt
import numpy as np

from agent import DeepQ_agent
from environment import Snake_Env
from collections import deque


fig_format = 'png'
#fig_format = 'svg'
RENDER = False      #if the Snake environment should be rendered


#Creating the environment
max_env_width, max_env_height = 27, 27       # max size of the environment
#env_width, env_height = 5, 5                 # initial size of the environment
env_width, env_height = 25, 25                 # initial size of the environment
display_width, display_height = 18 * max_env_width, 18 * max_env_height     # size of display
env = Snake_Env(max_env_width, max_env_height, env_width, env_height, display_width, display_height)


#Hyperparams
HIDDEN_UNITS = (32, 18, 10)
NETWORK_LR = 0.01   #0.01
BATCH_SIZE = 64     #64
UPDATE_EVERY = 5 #5
GAMMA = 0.95
epsilon, eps_min, eps_decay = 1.0, 0.05, 0.9996
epsilon = 0.05
NUM_EPISODES = 20000


#Initialises the DQN agent
agent = DeepQ_agent(env, hidden_units = HIDDEN_UNITS, network_LR = NETWORK_LR, batch_size = BATCH_SIZE, update_every = UPDATE_EVERY, gamma = GAMMA)


if os.path.exists('snake.h5'):
    print('Loading weights from previous learning session.')
    agent.qnetwork_target.load('snake.h5')
    agent.qnetwork_local.load('snake.h5')
else:
    print('No weights found from previous learning session.')


INCREASE_EVERY, SAVE_EVERY = 500, 200 
return_history = []
score_history = []


for i in range(1, NUM_EPISODES+1):

    
    epsilon = max(epsilon*eps_decay, eps_min)
    state = env.reset()
    action = agent.act(state, epsilon)

    #Render the environment for visualization   
    if RENDER: 
        env.render(action)

    cumulative_reward = 0.0

    while True:

        next_state, reward, done, info = env.step(action)

        #Add the experience to agent's memory
        agent.add_experience(state, action, reward, next_state, done)
        agent.learn()

        if RENDER:
            env.render(action)

        if done:
            break
        
        #Update state and action
        state = next_state
        action = agent.act(state, epsilon)
        cumulative_reward = agent.GAMMA * cumulative_reward + reward

    return_history.append(cumulative_reward)
    score_history.append(agent.env.SNAKE.LENGTH-4)
    print('iter: {}/{}, score: {}, cumulative reward: {:.3f}, eps: {:.3f}'.format(i, 
                                NUM_EPISODES, agent.env.SNAKE.LENGTH - 4, cumulative_reward, epsilon)) 


    if (i + 1)% SAVE_EVERY == 0:
        agent.qnetwork_local.save('snake.h5')
        plt.plot(return_history, 'tab:blue')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.savefig('dqn_training_reward.' + fig_format, fig_format=fig_format)
        
        plt.figure()
        plt.plot(score_history, 'tab:blue')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.savefig('dqn_training_score.' + fig_format, fig_format=fig_format)

    
    #Increase environment size
    if (i +1) % INCREASE_EVERY == 0:
        env.change_size(1, 1)
    
    
    #After 5000 episodes increases training time
    if (i+1) == 5000:
        INCREASE_EVERY = 600

#Save the agent's q-network weights for testing
agent.qnetwork_local.save('snake.h5')