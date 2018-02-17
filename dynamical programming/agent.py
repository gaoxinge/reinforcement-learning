# -*- coding: utf-8 -*-
import math

####################
# utility function #
####################
def _initialize(states, actions):
    policy, value_p, value_q = {}, {}, {}
    for state in states:
        value_p[state] = 0.0
        for action in actions:
            sa = (state, action)
            policy[sa] = 1.0 / len(actions)
            value_q[sa] = 0.0
    return policy, value_p, value_q


def _norm_diff(value_p, new_value_p):
    square = [(value_p[s] - new_value_p[s])**2 for s in value_p]
    return math.sqrt(sum(square))


def _arg_max(value_q, state):
    max_sa, max_value_q = None, None
    for sa in value_q:
        if sa[0] == state:
            if max_sa is None or value_q[sa] > max_value_q:
                max_sa = sa
                max_value_q = value_q[sa]
    return max_sa


def _all_equal(policy, new_policy):
    for sa in policy:
        if policy[sa] != new_policy[sa]:
            return False
    return True


#########################
# dynamical programming #
#########################
class DP(object):
    """
    :param states: state space
    :param actions: action space
    :param state_transition_table: state transition matrix
    :param reward_table: reward function
    :gamma: discount factor
    """

    def __init__(self, states, actions, state_transition_table, reward_table, gamma):
        self.states = states
        self.actions = actions
        self.table = state_transition_table
        self.reward = reward_table
        self.gamma = gamma
    
    def reset(self):
        self.policy, self.value_p, self.value_q = _initialize(self.states, self.actions)
    
    # below is the core methods in
    # dynamical programming:
    #
    # 1 policy_evalution
    # 2 policy_improvement
    # 3 policy_iteration
    # 4 value_iteration
    # 5 generalized_policy_iteration
    def policy_evaluation(self, iteration_num=None):
        iteration = 0
        
        while True:
            new_value_p = self.value_p.copy()
            
            for state in self.states:
                s = 0.0
                for action in self.actions:
                    for state_ in self.states:
                        sa = (state, action)
                        target = self.reward[sa] + self.gamma * self.value_p[state_]
                        s += self.policy[sa] * self.table[sa][state_] * target
                new_value_p[state] = s
            
            # check whether
            # 1 policy evaluation converges or
            # 2 iteration == iteration_num
            diff = _norm_diff(self.value_p, new_value_p)
            iteration += 1
            if diff < 1e-6 or (iteration_num and iteration == iteration_num):
                break
            else:
                self.value_p = new_value_p

    def policy_improvement(self, epsilon):
        new_policy = self.policy.copy()
        
        for state in self.states:
            for action in self.actions:
                s, sa = 0.0, (state, action)
                for state_ in self.states:
                    target = self.reward[sa] + self.gamma * self.value_p[state_]
                    s += self.table[sa][state_] * target
                self.value_q[sa] = s
            
            # update policy
            max_sa = _arg_max(self.value_q, state)
            for action in self.actions:
                sa = (state, action)
                if sa == max_sa:
                    new_policy[sa] = 1.0 - epsilon + epsilon / len(self.actions) 
                else:
                    new_policy[sa] = epsilon / len(self.actions)
        
        return new_policy
            
    def policy_iteration(self, iteration_num=None, epsilon=0):
        while True:
            self.policy_evaluation(iteration_num)
            new_policy = self.policy_improvement(epsilon)
            if _all_equal(self.policy, new_policy):
                break
            else:
                self.policy = new_policy
    
    def value_iteration(self, iteration_num=None, epsilon=0):
        iteration = 0
        
        while True:
            new_value_p = self.value_p.copy()
            
            for state in self.states:
                l = []
                for action in self.actions:
                    s = 0.0
                    for state_ in self.states:
                        sa = (state, action)
                        target = self.reward[sa] + self.gamma * self.value_p[state_]
                        s += self.table[sa][state_] * target
                    l.append(s)
                new_value_p[state] = max(l)
            
            # check whether
            # 1 value iteration converges or
            # 2 iteration == iteration_num
            diff = _norm_diff(self.value_p, new_value_p)
            iteration += 1
            if diff < 1e-6 or (iteration_num and iteration == iteration_num):
                break
            else:
                self.value_p = new_value_p
                
        self.policy = self.policy_improvement(epsilon)
                
    def generalized_policy_iteration(self, value_iter_num=None, policy_iter_num=None, 
                                     value_epsilon=0, policy_epsilon=0):
        self.value_iteration(value_iter_num, value_epsilon)
        self.policy_iteration(policy_iter_num, policy_epsilon)