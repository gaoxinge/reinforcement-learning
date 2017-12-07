# -*- coding: utf-8 -*-
import random

class MonteCarloAgent(object):

    def __init__(self, act_n, gamma, epsilon):
        self.act_n = act_n
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_q = {}
        self.value_n = {}
        self.episode = []

    def _initialize_state(self, state):
        if state not in self.value_q:
            self.value_q[state] = {a: 0.0 for a in range(self.act_n)}
            self.value_n[state] = {a: 0.0 for a in range(self.act_n)}

    def choose(self, state):
        if random.random() < self.epsilon:
            return random.choise(range(self.act_n))
        else:
            actions = self.value_q[state]
            m = max(actions.itervalues())
            for k, v in actions.iteritems():
                if v == m:
                    return k

    def store(self, state, action, reward):
        self.episode.append((state, action, reward))

    def learn(self):
        total_reward = 0
        for state, action, reward in reversed(self.episode):
            total_reward = reward + self.gamma * total_reward
            self.value_n[state][action] += 1
            self.value_q[state][action] += (total_reward - self.value[state][action]) / self.value_n[state][action]
        self.episode = []

class SaraAgent(object):

    def __init__(self, act_n, gamma, epsilon):
        self.act_n = act_n
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_q = {}
        self.value_n = {}

    def _initialize_state(self, state):
        if state not in self.value_q:
            self.value_q[state] = {a: 0.0 for a in range(self.act_n)}
            self.value_n[state] = {a: 0.0 for a in range(self.act_n)}

    def choose(self, state):
        if random.random() < self.epsilon:
            return random.choise(range(self.act_n))
        else:
            actions = self.value_q[state]
            m = max(actions.itervalues())
            for k, v in actions.iteritems():
                if v == m:
                    return k

    def learn(self, state, action, reward, state_, action_, done):
        self.value_n[state][action] += 1
        predict = self.value_q[state][action]
        target = reward if done else reward + self.gamma * self.value_q[state_][action_]
        self.value_q[state][action] += (target - predict) / self.value_n[state][action]

class SaraLambdaAgent(object):

    def __init__(self, act_n, gamma, epsilon, lambda_):
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
        if random.random() < self.epsilon:
            return random.choise(range(self.act_n))
        else:
            actions = self.value_q[state]
            m = max(actions.itervalues())
            for k, v in actions.iteritems():
                if v == m:
                    return k

    def learn(self, state, action, reward, state_, action_, done):
        self.value_n[state][action] += 1
        self.eligibility_trace[state][action] += 1
        predict = self.value_q[state][action]
        target = reward if done else reward + self.gamma * self.value_q[state_, action_]
        error = target - predict
        self.value_q[state][action] += error / self.value_n[state][action]
        for state in self.eligibility_trace:
            for action in self.eligibility_trace[state]:
                self.value_q[state][action] += self.lr * error * self.eligibility_trace[state][action]
                self.eligibility_trace[state][action] *= self.gamma * self.lambda_
                
class QLearnAgent(object):

    def __init__(self, act_n, gamma, epsilon):
        self.act_n = act_n
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_q = {}
        self.value_n = {}

    def _initialize_state(self, state):
        if state not in self.value_q:
            self.value_q[state] = {a: 0.0 for a in range(self.act_n)}
            self.value_n[state] = {a: 0.0 for a in range(self.act_n)}

    def choose(self, state):
        if random.random() < self.epsilon:
            return random.choise(range(self.act_n))
        else:
            actions = self.value_q[state]
            m = max(actions.itervalues())
            for k, v in actions.iteritems():
                if v == m:
                    return k

    def learn(self, state, action, reward, state_, done):
        self.value_n[state][action] += 1
        predict = self.value_q[state][action]
        target = reward + self.gamma * max(self.value[state_].itervalues())
        self.value_q[state][action] += (target - predict) / self.value_n[state][action]