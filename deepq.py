'''
Initialize Environment E
Initialize replay Memory M with capacity N (= finite capacity)
Initialize the DQN weights w
for episode in max_episode:
    s = Environment state
    for steps in max_steps:
        Choose action a from state s using epsilon greedy.
        Take action a, get r (reward) and s' (next state)
        Store experience tuple <s, a, r, s'> in M
        s = s' (state = new_state)
        Get random minibatch of exp tuples from M
        Set Q_target = reward(s,a) +  γmaxQ(s')
        Update w =  α(Q_target - Q_value) *  ∇w Q_value
'''
import math
import gym
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from itertools import count
from collections import deque, namedtuple

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 100
NUM_EPISODES = 10000
TARGET_UPDATE = 100
device = 'gpu' if torch.cuda.is_available() else 'cpu'

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class DQN(nn.Module):
    def __init__(self, state_size, action_size, eps=1.0, decay=0.999):
        '''
        Params
        ------
        state_size (int): size of each observation
        action_size (int): size of the action space
        eps (float): initial value for greedy exploration (default=1.0)
        decay (float): the rate at which `eps` is decayed (default=0.999)
        '''
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.decay = decay
        self.eps = eps
        self._steps = 0

        self.l1 = nn.Linear(self.state_size, self.state_size)
        self.l2 = nn.Linear(self.state_size, self.action_size)

    def forward(self, state):
        return self.l2(F.relu(self.l1(state)))
    
    def get_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self._steps / EPS_DECAY)
        self._steps += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.forward(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

class Memory(object):
    def __init__(self, maxlen=1000):
        '''
        Params
        ------
        maxlen (int): Maximum length of the memory buffer
        '''
        self.buffer = deque(maxlen=maxlen)
    
    def append(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size=32):
        return random.choices(self.buffer, k=batch_size)
    
    def __len__(self):
        return len(self.buffer)


def get_state_size(env):
    size = 1
    for dim in env.observation_space.shape:
        size *= dim
    return size


def train():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = tuple(map(lambda s: s is not None, batch.next_state))
    non_final_mask = torch.tensor(non_final_mask, device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def test():
    state = env.reset()
    state = torch.from_numpy(state).float().unsqueeze(0)
    for t in count():
        env.render()
        action = target_net(state).max(1)[1].view(1, 1)
        next_state, _, done, _ = env.step(action.item())
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        state = next_state
        if done: 
            print('Test run lasted for {} steps'.format(t))
            break

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    policy_net = DQN(get_state_size(env), env.action_space.n).to(device)
    target_net = DQN(get_state_size(env), env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters())
    memory = Memory(maxlen=10**6)

    for i_episode in range(NUM_EPISODES):
        state = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0)
        for t in count():
            action = policy_net.get_action(state)
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            if done: next_state = None
            memory.append(state, action, reward, next_state)
            state = next_state

            train()
            if done: break
        if i_episode % TARGET_UPDATE == 0:
            print('Episode {} done with {} steps'.format(i_episode, t))
            target_net.load_state_dict(policy_net.state_dict())
            test()

    print('Complete')
    env.close()