# -*- coding: utf-8 -*-
import math


def _dot_operation(x, y):
    return sum([t1 * t2 for t1, t2 in zip(x, y)])


class Model:

    def __init__(self, state_n, act_n, learning_rate):
        self.state_n = state_n
        self.act_n = act_n
        self.learning_rate = learning_rate
        self.theta = [0.0] * (state_n + 1)
        
    def _make_feature(self, state, action):
        return [s for s in state] + [action]
        
    def _logits(self, state, action):
        return _dot_operation(self._make_feature(state, action), self.theta)
        
    def _prob(self, state, action):
        t1 = math.exp(self._logits(state, action))
        t2 = sum([math.exp(self._logits(state, a)) for a in range(self.act_n)])
        return t1 / t2
        
    def predict(self, state):
        t1 = [math.exp(self._logits(state, a)) for a in range(self.act_n)]
        t2 = sum(t1)
        return [t / t2 for t in t1]
        
    def fit(self, state, action, total_reward):
        theta = [0.0] * len(self.theta)
        t1 = self._make_feature(state, action)
        t2 = [self._prob(state, a) for a in range(self.act_n)]
        t3 = [self._make_feature(state, a) for a in range(self.act_n)]
        for i in range(len(self.theta)):
            theta[i] = self.theta[i] + self.learning_rate * (t1[i] - _dot_operation(t2, [t[i] for t in t3])) * total_reward
        self.theta = theta