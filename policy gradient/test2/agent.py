# -*- coding: utf-8 -*-
import random
from model import Model


def _all_equal(probs):
    for prob in probs:
        for prob_ in probs:
            if prob != prob_:
                return False
    return True


class MonteCarloPolicyGradientAgent:

    def __init__(self, state_n, act_n, gamma=0.9, learning_rate=0.01):
        self.state_n = state_n
        self.act_n = act_n
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        self.episode = []
        
        self.model = Model(state_n, act_n, learning_rate)

    def choose(self, state):
        probs = self.model.predict(state)
        if _all_equal(probs):
            return random.choice(range(self.act_n))
        m = max(probs)
        return probs.index(m)

    def store(self, state, action, reward):
        self.episode.append((state, action, reward))

    def learn(self):
        total_reward = 0
        for state, action, reward in reversed(self.episode):
            total_reward = reward + self.gamma * total_reward
            self.model.fit(state, action, total_reward)
        self.episode = []