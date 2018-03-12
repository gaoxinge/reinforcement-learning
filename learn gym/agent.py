# -*- coding: utf-8 -*-
import random


def _all_equal(actions):
    for action in actions:
        for action_ in actions:
            if actions[action] != actions[action_]:
                return False
    return True


class SarsaLambdaAgent(object):

    def __init__(self, act_n, gamma=0.9, epsilon=0.1, lambda_=0.9):
        self.act_n = act_n
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.value_q = {}
        self.value_n = {}
        self.eligibility_trace = {}

    def _initialize_state(self, state):
        if state not in self.value_q:
            self.value_q[state] = {a: 0.0 for a in range(self.act_n)}
            self.value_n[state] = {a: 0.0 for a in range(self.act_n)}
            self.eligibility_trace[state] = {a: 0.0 for a in range(self.act_n)}

    def choose(self, state):
        self._initialize_state(state)
        actions = self.value_q[state]
        if random.random() < self.epsilon or _all_equal(actions):
            return random.choice(range(self.act_n))
        else:
            m = max(actions.values())
            for k, v in actions.items():
                if v == m:
                    return k

    def learn(self, state, action, reward, state_, action_, done):
        self.value_n[state][action] += 1
        predict = self.value_q[state][action]
        target = reward if done else reward + self.gamma * self.value_q[state_][action_]
        error = target - predict
        self.eligibility_trace[state][action] += 1
        for s in self.eligibility_trace:
            for a in self.eligibility_trace[s]:
                self.value_q[s][a] += self.eligibility_trace[s][a] * error / (self.value_n[s][a] + 1e-6)
                self.eligibility_trace[s][a] *= self.gamma * self.lambda_
