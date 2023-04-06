import os
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

from agent import DeepQ_agent
from environment import Snake_Env


fig_format = 'png'
#fig_format = 'svg'

#Creating the environment
max_env_width, max_env_height = 27, 27
env_width, env_height = max_env_width-2, max_env_height-2
display_width, display_height = max_env_width * 18, max_env_height * 18
env = Snake_Env(max_env_width, max_env_height, env_width, env_height, display_width, display_height)

#Creating the DQN agent (with greedy policy, suited for evaluation)
agent = DeepQ_agent(env, hidden_units=(32, 18, 10))

#Checking if weights from previous learning session exists
if os.path.exists('snake.h5'):
    print('Loading weights from previous learning session.')
    agent.qnetwork_local.load('snake.h5')
else:
    print('No weights found from previous learning session. Unable to proceed.')
    exit(-1)

NUM_EPISODES = 30
score_history = []
return_history = []

#Testing the agent
for i in range(NUM_EPISODES):
    state = env.reset()
    env.render(0)
    cumulative_reward = 0.0

    while True:
        #Decide action for present state
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        #Rendering the environment
        env.render(action)
        cumulative_reward = cumulative_reward * agent.GAMMA + reward
        if done:
            sleep(1)
            break

    score_history.append(agent.env.SNAKE.LENGTH-4)
    return_history.append(cumulative_reward)
    print('iter: {}/{}, score: {}, cumulative reward: {:.3f}'.format(i+1, NUM_EPISODES, agent.env.SNAKE.LENGTH - 4, cumulative_reward))

#Prints mean reward and mean score
print('Mean reward: {:.3f}'.format(np.mean(return_history)))
print('Mean score: {:.3f}'.format(np.mean(score_history)))

#Plots return history
plt.plot(return_history)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig('dqn_evaluation_reward.' + fig_format, fig_format=fig_format)

#Plot score history
plt.figure()
plt.plot(score_history)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.savefig('dqn_evaluation_score.' + fig_format, fig_format=fig_format)
plt.show()

plt.show()