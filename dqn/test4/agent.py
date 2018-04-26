# -*- coding: utf-8 -*-
import random
import numpy as np
from model import Model


class QLearnAgent:

    def __init__(self, state_n, act_n, gamma=0.9, epsilon=0.1, learning_rate=0.01):
        self.state_n = state_n
        self.act_n = act_n
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.model = Model(self.state_n, self.act_n, self.learning_rate)

    def choose(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.act_n))
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values)

    def learn(self, state, action, reward, state_, done):
        target = reward if done else reward + self.gamma * np.amax(self.model.predict(state_))
        act_values = self.model.predict(state)
        act_values[0][action] = target
        self.model.fit(state, act_values)
