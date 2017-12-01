# -*- coding: utf-8 -*-

class State(object):
    """State is a class, which represents
    a coordinate in grid world.
    
    :param x: x-axis coordinate
    :param y: y-axis coordinate
    """
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        else:
            raise IndexError('state have no index {key}'.format(key=key))
        
    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        else:
            raise IndexError('state have no index {key}'.format(key=key))
    
    def __eq__(self, other):
        assert isinstance(other, State)
        if self.x == other.x and self.y == other.y:
            return True
        return False
    
    def __hash__(self):
        return hash((self.x, self.y))
        
    def __str__(self):
        return '({x}, {y})'.format(x=self.x, y=self.y)
    
    def __repr__(self):
        return 'State({x}, {y})'.format(x=self.x, y=self.y)
    
class GridWorld(object):
    """Grid world is a game with
    four actions:
    
        0  <--->  left
        1  <--->  right
        2  <--->  up
        3  <--->  down
    
    :param size: size * size grids
    :param gamma: discounted value
    """

    def __init__(self, size, gamma):
        self.size = size
        self.gamma = gamma
    
    @property
    def state_transition_table(self):
        table = {}
        for x in range(self.size):
            for y in range(self.size):
                for action in range(4):
                    if x > 0 and action == 0:
                        table[(State(x, y), action)] = (State(x-1, y), 1)
                    elif x < self.size-1 and action == 1:
                        table[(State(x, y), action)] = (State(x+1, y), 1)
                    elif y > 0 and action == 2:
                        table[(State(x, y), action)] = (State(x, y-1), 1)
                    elif y < self.size-1 and action == 3:
                        table[(State(x, y), action)] = (State(x, y+1), 1)
                    else:
                        table[(State(x, y), action)] = (State(x, y), 1)
        return table

    @property
    def reward_table(self):
        table = {}
        for x in range(self.size):
            for y in range(self.size):
                for action in range(4):
                    table[(State(x, y), action)] = -1
        return table
        
    def reset(self):
        self.state = State(0, 0)
        return self.state
    
    def step(self, action):
        if self.state[0] > 0 and action == 0:
            self.state[0] -= 1
        elif self.state[0] < self.size-1 and action == 1:
            self.state[0] += 1
        elif self.state[1] > 0 and action == 2:
            self.state[1] -=1
        elif self.state[1] < self.size-1 and action == 3:
            self.state[1] += 1
        
        if self.state == State(self.size-1, self.size-1):
            done = True
        else:
            done = False
            
        return self.state, -1, done
    
    def render(self):
        import time
        time.sleep(0.01)
        
        view = ''
        for y in range(self.size):
            for x in range(self.size):
                if State(x, y) == self.state:
                    view += 'Q'
                else:
                    view += '*'
            view += '\n'
        print view
        
if __name__ == '__main__':
    env = GridWorld(4, 1)
    total_reward = 0
    env.reset()
    while True:
        env.render()
        action = input('>>> ')
        _, reward, done = env.step(action)
        total_reward += reward
        if done:
            env.render()
            break
    print 'Gamve Over'
    print 'Your Reward:', total_reward