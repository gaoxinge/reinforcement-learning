import random


class BernoulliArm:
    
    def __init__(self, p):
        self.p = p
        
    def step(self):
        return 1.0 if random.random() < self.p else 0.0
