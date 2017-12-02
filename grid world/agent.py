# -*- coding: utf-8 -*-
import math

def _initialize(states, actions):
    policy, value_p, value_q = {}, {}, {}
    for state in states:
        value_p[state] = 0.0
        for action in actions:
            policy[(state, action)] = 1.0 / actions
            value_q[(state, action)] = 0.0
    return policy, value_p, value_q

def _norm(value_p, new_value_p):
    square = [(value_p[i] - new_value_p[i])**2 for i in range(len(value_p))]
    return math.sqrt(sum(square))

def _argmax(value_q, state):
    k, m = None, None
    for sa in value_q:
        if sa[0] == state:
            if m is None:
                m = value_q[sa]
            if value_q[sa] > m:
                k = sa
                m = value_q[sa]
    return k[1] if k
    
def _all_equal(policy, new_policy):
    for sa in policy:
        if policy[sa] == new_policy[sa]:
            return False
    return True
    
class Control(object):

    def __init__(self, states, actions, state_transition_table, reward_table, gamma):
        self.states = states
        self.actions = actions
        self.table = state_transition_table
        self.reward = reward_table
        self.gamma = gamma
        self.policy, self.value_p, self.value_q = \
        _initialize(self.states, self.actions)
        
    def policy_evaluation(self, iteration_num=None):
        iteration = 0
        while True:
            iteration += 1
            new_value_p = self.value_p.copy()
            for state in self.states:
                sum_s = 0
                for state_ in self.states:
                    sum_a = 0
                    for action in self.actions:
                        sum_a += self.policy[(state, action)] * self.table[(state, action)][state_] * (self.reward[(state, action)] + self.gamma * new_value_p[state_])
                    sum_s += sum_a
                new_value_p[state] = sum_s
            diff = _norm(self.value_p, new_value_p)
            if diff < 1e-6 or (iteration_num and iteration == iteration_num):
                break
            else:
                self.value_p = new_value_p

    def policy_improvement(self):
        for state in self.states:
            for action in self.actions:
                sa = (state, action)
                sum_s = 0.0
                for state_ in self.states:
                    sum_s += self.table[sa][state_] * (self.reward[sa] + self.gamma * self.value_p[state_])
                self.value_q[sa] = sum_s
            k = _argmax(self.value_q, state)
            for action in self.actions:
                sa = (state, action)
                self.policy[sa] = 1.0 if sa == (state, k) else 0.0
        if _all_equal(self.policy, new_policy):
            return True
        else:
            self.policy = new_policy
            return False
            
    def policy_iteration(self, iteration_num=None):
        iteration = 0
        while True:
            iteration += 1
            self.policy_evaluation(iteration_num)
            done = self.policy_improvement()
            if done:
                break
    
    def value_iteration(self, iteration_num=None):
        iteration = 0
        while True:
            iteration += 1
            for state in self.states:
                list_a = []
                for action in self.actions:
                    sa = (state, action)
                    sum_s = []
                    for state_ in self.states:
                        sum_s.append(self.table[sa][state_] * (self.reward[sa] + self.gamma * new_value_p[state_]))
                    list_a.append(sum_s)
                new_value_p[state] = max(list_a)
            diff = _norm(self.value_p, new_value_p)
            if diff < 1e-6 or (iteration_num and iteration == iteration_num):
                break
            else:
                self.value_pi = new_value_pi
        
        for state in self.states:
            for action in self.actions:
                sa = (state, action)
                sum_s = 0.0
                for state_ in self.states:
                    sum_s += self.table[sa][state_] * (self.reward[sa] + self.gamma * self.value_p[state_])
                self.value_q[sa] = sum_s
            k = _argmax(self.value_q, state)
            for action in self.actions:
                sa = (state, action)
                self.policy[sa] = 1.0 if sa == (state, k) else 0.0
                
    def generalized_policy_iteration(self, value_iter_num, policy_iter_num):
        self.value_iteration(value_iter_num)
        self.policy_iteration(policy_iter_num)