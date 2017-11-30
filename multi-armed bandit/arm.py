# -*- coding: utf-8 -*-
import random

class BernoulliArm(object):
    
    def __init__(self, p):
        self.p = p
        
    def step(self):
        if random.random() < self.p:
            return 1.0
        else:
            return 0.0