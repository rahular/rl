import random
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
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

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

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
        
        self.episode_reward = []
        self.saved_actions = []

        self.l1 = nn.Linear(self.state_size, self.state_size)
        self.action_head = nn.Linear(self.state_size, self.action_size)
        self.value_head = nn.Linear(self.state_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        action_logits = self.action_head(F.relu(self.l1(state)))
        value = self.value_head(F.relu(self.l1(state)))
        return self.softmax(action_logits), value
    
    def get_action(self, state, train=True):
        action, value = self.forward(state)
        c = Categorical(action)
        action = c.sample()

        if not train: return action
        
        self.saved_actions.append(SavedAction(c.log_prob(action), value))
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

    action_loss, value_loss = [], []
    for (log_prob, value), reward in zip(policy_net.saved_actions, rewards):
        advantage = reward - value.item()
        action_loss.append(log_prob * advantage)
        value_loss.append(F.smooth_l1_loss(value.squeeze(0), torch.Tensor([reward])))
    loss = -torch.stack(action_loss).sum() + torch.stack(value_loss).sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    policy_net.saved_actions = []
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