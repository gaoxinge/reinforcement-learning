# -*- coding: utf-8 -*-
import random
import numpy as np
from model import Model


class QLearnAgent:

    def __init__(self, state_n, act_n, gamma=0.9, epsilon=0.1):
        self.state_n = state_n
        self.act_n = act_n
        self.gamma = gamma
        self.epsilon = epsilon

        self._build_model()

    def _build_model(self):
        self.model = Model(self.state_n, self.act_n)

    def choose(self, state):
        act_values = self.model.predict(state)
        if random.random() < self.epsilon:
            return random.choice(range(self.act_n))
        else:
            np.argmax(act_values)

    def learn(self, state, action, reward, state_, done):
        target = reward if done else reward + self.gamma * np.amax(self.model.predict(state_))
        act_values = self.model.predict(state)
        act_values[action] = target
        self.model.fit(state, act_values)
