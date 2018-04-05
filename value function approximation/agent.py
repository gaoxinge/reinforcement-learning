# -*- coding: utf-8 -*-
import random


def _all_equal(actions):
    for action in actions:
        for action_ in actions:
            if actions[action] != actions[action_]:
                return False
    return True


class SarsaAgent(object):

    def __init__(self, act_n, linear_func, gamma=0.9, epsilon=0.1, learning_rate=0.01):
        self.act_n = act_n
        self.linear_func = linear_func
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def choose(self, state):
        actions = {a: self.linear_func(state, a) for a in range(self.act_n)}
        if random.random() < self.epsilon or _all_equal(actions):
            return random.choice(range(self.act_n))
        else:
            m = max(actions.values())
            for k, v in actions.items():
                if v == m:
                    return k

    def learn(self, state, action, reward, state_, action_, done):
        predict = self.linear_func(state, action)
        target = reward if done else reward + self.gamma * self.linear_func(state_, action_)
        error = target - predict
        self.linear_func.update(self.learning_rate * error)
