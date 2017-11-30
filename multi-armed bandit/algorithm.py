# -*- coding: utf-8 -*-
import math
import random

class ABTest(object):

    def __init__(self, arm_n):
        self.arm_n = arm_n
        self.name = 'ab test'

    def pull(self):
        return random.choice(range(self.arm_n))

    def update(self, arm, reward):
        pass

class EpsilonGreedy(object):
    
    def __init__(self, arm_n, epsilon):
        self.arm_n = arm_n
        self.epsilon = epsilon
        self.values = [0.0 for _ in range(self.arm_n)]
        self.counts = [0.0 for _ in range(self.arm_n)]
        self.name = 'epsilon greedy: %s' % self.epsilon
        
    def pull(self):
        if random.random() < self.epsilon:
            return random.choice(range(self.arm_n))
        else:
            m = max(self.values)
            return self.values.index(m)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

class Softmax(object):
    
    def __init__(self, arm_n, temperature):
        self.arm_n = arm_n
        self.temperature = temperature
        self.values = [0.0 for _ in range(self.arm_n)]
        self.counts = [0.0 for _ in range(self.arm_n)]
        self.name = 'softmax: %s' % self.temperature
        
    def pull(self):
        s = sum([math.exp(v / self.temperature) for v in self.values])
        probs = [math.exp(v / self.temperature) / s for v in self.values]
        t, prob = random.random(), 0.0
        for arm in range(self.arm_n):
            prob += probs[arm]
            if prob > t:
                return arm
        return self.arm_n - 1
        
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        
class UCB1(object):
    
    def __init__(self, arm_n):
        self.arm_n = arm_n
        self.values = [0.0 for _ in range(self.arm_n)]
        self.counts = [0.0 for _ in range(self.arm_n)]
        self.name = 'UCB1'
        
    def pull(self):
        for arm in range(self.arm_n):
            if self.counts[arm] == 0:
                return arm
        total_counts = sum(self.counts)
        ucb_values = [self.values[arm] + math.sqrt((2 * math.log(total_counts)) / self.counts[arm])
                      for arm in range(self.arm_n)]
        m = max(ucb_values)
        return ucb_values.index(m)
        
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]