#Description: Main file to run the DDPG algorithm
import torch
import gym
from ddpg import DDPGAgent
import numpy as np
import os


#initialize environment
env = gym.make(id = "Pendulum-v1")   
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

#initialize agent
agent = DDPGAgent(STATE_DIM, ACTION_DIM)

#hyperparameters
MAX_EPISODES = 200
MAX_STEPS = 200
BATCH_SIZE = 64

#train agent
for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    for step in range(MAX_STEPS):
        action = agent.get_action(state) + np.random.normal(0, 0.1, ACTION_DIM)
        next_state, reward, done, _, _ = env.step(2 * action)
        agent.memory.push(state, action, reward, next_state, done)
        episode_reward += reward
        if len(agent.memory) > BATCH_SIZE:
            agent.update()
        state = next_state
        if done:
            break
    print("Episode: {}, Total Reward: {:.2f}".format(episode, episode_reward))

env.close()

#save model
save_dir = 'C:\\Users\\Administrator\\Desktop\\some_learning\\learning_again'
os.makedirs(save_dir, exist_ok=True)
torch.save(agent.actor.state_dict(), os.path.join(save_dir, 'actor.pth'))
torch.save(agent.critic.state_dict(), os.path.join(save_dir, 'critic.pth'))
