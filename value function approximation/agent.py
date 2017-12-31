# -*- coding: utf-8 -*-
import random

def _all_equal(actions):
    for action in actions:
        for action_ in actions:
            if actions[action] != actions[action_]:
                return False
    return True

def _dot_operation(x, y):
    assert len(x) == len(y)
    return sum([x[i] * y[i] for i in range(len(x))])

def _make_feature(state, action):
    return [state[0], state[1], action]
    
class SarsaAgent:

    def __init__(self, state_n, act_n, gamma=0.9, epsilon=0.1, learning_rate=0.01):
        self.state_n = state_n
        self.act_n = act_n
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.w = [0.0 for _ in range(self.state_n+1)]
    
    def choose(self, state):
        act_val = [_dot_operation(self.w, _make_feature(state, a)) 
                   for a in range(self.act_n)]
        if random.random() < self.epsilon or _all_equal(act_val):
            return random.choice(range(self.act_n))
        else:
            m = max(act_val.itervalues())
            for a in self.range(act_n):
                if act_val[a] == m:
                    return a

    def learn(self, state, action, reward, state_, action_, done):
        predict = self.value_q[state][action]
        target = reward if done else reward + self.gamma * self.value_q[state_][action_]
        error = target - predict
        features = _make_feature(state, action)
        for i in range(self.state_n+1):
            self.w[i] += self.learning_rate * error * features[i]