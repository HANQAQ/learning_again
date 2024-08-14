import torch
import os
os.add_dll_directory(os.environ['minGwPath'])

from build import flyBall as env
from ddpg import DDPGAgent
import numpy as np
import matplotlib.pyplot as plt

#initialize the environment
env.reset()
STATE_DIM = 7
ACTION_DIM = 1

#initialize agent   
agent = DDPGAgent(STATE_DIM, ACTION_DIM, hidden_dim=STATE_DIM**3)

#create model save directory
model_save_dir = os.environ['MODEL_PATH']
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

#hyperparameters
DEBUG_MODE = False
MAX_EPISODES = 2 if DEBUG_MODE else 2000
MAX_STEPS = 110
BATCH_SIZE = 64
model_save_period = 500

episode_rewards = []

#train agent
for episode in range(MAX_EPISODES):
    env.reset()
    episode_reward = 0
    state = env.state()
    for step in range(MAX_STEPS):
        action = agent.get_action(state) + np.random.normal(0, 0.1, ACTION_DIM)
        env.step(12*np.clip(action, -1.0, 1.0))
        next_state = env.state()
        reward = env.reward()
        done = env.done()
        agent.memory.push(state, action, reward, next_state, env.done())
        episode_reward += reward
        state = next_state
        if len(agent.memory) > BATCH_SIZE:
            agent.update()

        if DEBUG_MODE:
            print(f"Episode: {episode}, Step: {step}, State: {state[:4]}, Action: {action}, Reward: {reward}, Done: {done}")
        
        if done:
            break
    episode_rewards.append(episode_reward)
    print("Episode: {}, Total Reward: {:.2f}".format(episode, episode_reward))
    #save model
    if (episode+1) % model_save_period == 0:
        torch.save(agent.actor.state_dict(), os.path.join(model_save_dir, f"{episode+1}actor.pth")) 
        torch.save(agent.critic.state_dict(), os.path.join(model_save_dir, f"{episode+1}critic.pth"))

#plot total rewards
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

