import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
#acotr network
#2 Linear layers with ReLU activation ending with a tanh activation
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 64):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    def forward(self, state):
        return self.actor(state)

# critic network
# 2 Linear layers with ReLU activation ending with no activation
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 64):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state, action):
        return self.critic(torch.cat([state, action], dim = 1))

#replay buffer
#deque with 15,000 capacity
#sample, push, and __len__ methods
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace = False)
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in indices])
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    def __len__(self):
        return len(self.buffer)

#DDPG agent
#initialize, get_action, update methods
class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim = 64, actor_lr = 0.0001, critic_lr = 0.001, gamma = 0.99, tau = 0.005):
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = actor_lr)
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = critic_lr)
        
        self.memory = ReplayBuffer(15000)
        self.gamma = gamma
        self.tau = tau
    def get_action(self, state):
        state = torch.FloatTensor(state).to(device)
        actor = self.actor(state).cpu().data.numpy()
        return actor
    def update(self, batch_size = 64):
        state, action, reward, next_state, done = self.memory.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)
        #update critic
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()
        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        #update actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        #update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)