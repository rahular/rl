import random
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from itertools import count
from torch.distributions import Categorical

GAMMA = 0.999
NUM_EPISODES = 5000
TEST_INTERVAL = 500
device = 'gpu' if torch.cuda.is_available() else 'cpu'

env = gym.make('CartPole-v0')
seed = 42

random.seed(seed)
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class PGNet(nn.Module):
    def __init__(self, state_size, action_size):
        '''
        Params
        ------
        state_size (int): size of each observation
        action_size (int): size of the action space
        '''
        super(PGNet, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        self.policy_history = torch.Tensor()
        self.episode_reward = []

        self.l1 = nn.Linear(self.state_size, self.state_size)
        self.l2 = nn.Linear(self.state_size, self.action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        logits = self.l2(F.relu(self.l1(state)))
        return self.softmax(logits)
    
    def get_action(self, state, train=True):
        output = self.forward(state)
        c = Categorical(output)
        action = c.sample()

        if not train: return action
        
        if self.policy_history.dim() != 0:
            self.policy_history = torch.cat([self.policy_history, c.log_prob(action)])
        else:
            self.policy_history = (c.log_prob(action))
        return action
    
def get_state_size(env):
    size = 1
    for dim in env.observation_space.shape:
        size *= dim
    return size

def train():
    R = 0
    rewards = []
    
    for r in policy_net.episode_reward[::-1]:
        R = r + GAMMA * R
        rewards.insert(0, R)
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    loss = torch.sum(-torch.mul(policy_net.policy_history, rewards))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    policy_net.policy_history = torch.Tensor()
    policy_net.episode_reward = []

def test():
    state = env.reset()
    for t in count():
        env.render()
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = policy_net.get_action(state, train=False)
        state, _, done, _ = env.step(action.item())
        if done: 
            print('Test run lasted for {} steps'.format(t))
            break

if __name__ == '__main__':
    policy_net = PGNet(get_state_size(env), env.action_space.n).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters())
    running_reward = 10

    for i_episode in range(NUM_EPISODES):
        cum_reward = 0.0
        state = env.reset()
        for t in count():
            state = torch.from_numpy(state).float().unsqueeze(0)
            action = policy_net.get_action(state)
            state, reward, done, _ = env.step(action.item())
            policy_net.episode_reward.append(reward)
            cum_reward += reward
            if done: 
                running_reward = 0.05 * cum_reward + (1 - 0.05) * running_reward
                break
        train()
        if i_episode % TEST_INTERVAL == 0:
            print('Episode {} done with avg. reward {:.2f}'.format(i_episode, running_reward))
            test()
        if running_reward > env.spec.reward_threshold:
            print('Complete with avg. reward {:.2f}!'.format(running_reward))
            break
    for _ in range(10): test()
    env.close()