# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from model import Model


class OOAgent:

    def __init__(self, state_n, act_n):
        self.state_n = state_n
        self.act_n = act_n
        self.gamma = 0.95
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.memory_size = 10000
        self.batch_size = 500
        self.memory = deque(maxlen=self.memory_size)
        
        self.model = Model(self.state_n, self.act_n, self.learning_rate)

    def choose(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.act_n))
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values)
            
    def store(self, state, action, reward, state_, done):
        self.memory.append((state, action, reward, state_, done))

    def online_learn(self, state, action, reward, state_, done):
        target = reward if done else reward + self.gamma * np.amax(self.model.predict(state_))
        act_values = self.model.predict(state)
        act_values[0][action] = target
        self.model.fit(state, act_values)
        
    def offline_learn(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, state_, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(state_))
            act_values = self.model.predict(state)
            act_values[0][action] = target
            self.model.fit(state, act_values)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
