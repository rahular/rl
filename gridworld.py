'''
This is a simple implementation of grid world for a mouse.
It contains the following tiles:
    - an empty tile
    - cheese block
    - rat poison
    - exit (optional)
The goal of the mouse is to eat as many cheese blocks as possible
as quickly as possible and exit the world without eating poison.
'''
import random

class GridWorld(object):
    def __init__(self, size=8, cprob=0.1, creward=10, pprob=0.1, preward=-10, greward=100, force_fast=True, seed=42):
        '''
        Params
        ------
        size (int): size of the grid world (default=8)
        cprob (float): probability of placing a cheese on a tile (default=0.1)
        creward (int): reward for landing on a tile with cheese (default=10)
        pprob (float): probability of placing poison on a tile (default=0.1)
        preward (int): reward for landing on a tile with poison (default=-10)
        greward (int): reward for reaching the goal tile (default=100)
        force_fast (bool): force the agent to reach the goal fast by giving a small 
                           negative reward at each step (default=True)
        seed (int): a random seed (default=42)
        '''
        random.seed(seed)

        self.size = size
        self.cprob = cprob
        self.creward = creward
        self.pprob = pprob
        self.preward = preward
        self.greward = greward
        self.force_fast = force_fast

        # initialize the world with empty spaces
        self._empty = -1 if self.force_fast else 0
        self.grid = [[self._empty] * self.size for _ in range(self.size)]

        # place objects
        self._place('p', self.pprob)
        self._place('c', self.cprob)

    def _place(self, item, prob):
        if item not in ['p', 'c']:
            raise ValueError('Invalid item type.')
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == self._empty and random.random() < prob:
                    self.grid[i][j] = self.creward if item == 'c' else self.preward
        # reserve [0, 0] for start and [size, size] for goal
        self.grid[0][0] = 0
        self.grid[self.size-1][self.size-1] = self.greward
    
    def get_valid_actions(self, i, j):
        valid_actions = []
        if i > 0: valid_actions.append('up')
        if i < self.size-1: valid_actions.append('down')
        if j > 0: valid_actions.append('left')
        if j < self.size-1: valid_actions.append('right')
        return valid_actions
    
    def get_reward(self, i, j):
        is_final = True if i == self.size-1 and j == self.size-1 else False
        reward = self.grid[i][j]
        self.grid[i][j] = self._empty
        return reward, is_final

    def move(self, i, j, action):
        if action == 'up':
            i = i-1
        elif action == 'down':
            i = i+1
        elif action == 'right':
            j = j+1
        elif action == 'left':
            j = j-1
        return i, j

    def show(self):
        for i in range(self.size):
            for j in range(self.size):
                print(self.grid[i][j], end='\t')
            print()
