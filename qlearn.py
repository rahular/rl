import random
import numpy as np

from collections import defaultdict
from gridworld import GridWorld

class qAgent(object):
    def __init__(self, lr=0.1, eps=1.0, discount=0.9):
        '''
        Params
        ------
        lr (float): learning rate (default=0.1)
        eps (float): exploration co-efficient; this will be decayed over time (default=1.0)
        discount (float): future reward discount co-efficient (default=0.9)
        '''
        self.lr = lr
        self.eps = eps
        self.discount = discount
        self.qtable = defaultdict(lambda: defaultdict(float))
    
    def set_grid(self, grid):
        self.g = grid

    def make_state(self, i, j):
        return str(i) + ' ' + str(j)
    
    def act(self, row, rand=False):
        if rand:
            action = random.choice(list(row.keys()))
            return row[action], action
        best_qval, best_action = -100, None
        for action, qval in row.items():
            if qval > best_qval:
                best_qval = qval
                best_action = action
        return best_qval, best_action
    
    def get_action(self, i, j):
        state = self.make_state(i, j)
        if not self.qtable[state]:
            for action in self.g.get_valid_actions(i, j):
                self.qtable[state][action] = 0
            
        if random.random() > self.eps:
            return self.act(self.qtable[state])[1]
        else:
            return self.act(self.qtable[state], rand=True)[1]

    def update_q(self, old_i, old_j, new_i, new_j, action, reward):
        old_state = self.make_state(old_i, old_j)
        new_state = self.make_state(new_i, new_j)
        qprime, _ = self.act(self.qtable[new_state])
        self.qtable[old_state][action] = self.qtable[old_state][action] + self.lr * (reward + self.discount*qprime - self.qtable[old_state][action])
        self.eps = max(0.2, self.eps*0.9)

def show_qtable(agent, size):
    for i in range(size):
        for j in range(size):
            state = agent.make_state(i, j)
            action = agent.qtable.get(state, 'NA')
            if action == 'NA':
                print(action, end='\t')
            else:
                # print('{0:.2f}'.format(agent.act(action)[0]), end='\t')
                print(agent.act(action)[1], end='\t')
        print()

if __name__ == '__main__':
    max_steps = 100
    max_iters = 1000
    seed = random.randint(0, 100)
    agent = qAgent()
    grid = GridWorld(size=8, force_fast=True, seed=seed)
    grid.show()
    print()
    for iter in range(max_iters):
        agent.set_grid(grid)
        i, j = 0, 0 # initial state
        cum_reward = 0
        for step in range(max_steps):
            action = agent.get_action(i, j)
            new_i, new_j = grid.move(i, j, action)
            reward, is_final = grid.get_reward(i, j)
            cum_reward += reward
            agent.update_q(i, j, new_i, new_j, action, reward)
            if is_final:
                break
            i = new_i
            j = new_j
        if iter%100 == 0:
            print('Episode {} finished after {} steps with cumulative reward of {}'.format(iter, step, cum_reward))
        grid = GridWorld(size=8, force_fast=True, seed=seed)
    print()
    show_qtable(agent, grid.size)