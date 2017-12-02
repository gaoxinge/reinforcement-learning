# -*- coding: utf-8 -*-
from mdp import GridWorld
from algorithm import DP

size = 4
gamma = 1

gw = GridWorld(size, gamma)

args = [
    gw.states,
    gw.actions,
    gw.state_transition_table,
    gw.reward_table,
    gw.gamma
]

correspond = {
    0:  u'左',
    1:  u'右',
    2:  u'上',
    3:  u'下'
}

dp = DP(*args)

def show_value_p():
    view = ''
    for y in range(size):
        for x in range(size):
            value_p = dp.value_p[(x, y)]
            view += str(int(round(value_p, 0))) + ' '
        view += '\n\n'
    print view

def show_policy():
    view = u''
    for y in range(size):
        for x in range(size):
            max_sa, max_policy = None, None
            for action in range(4):
                sa = ((x, y), action)
                if max_sa is None or dp.policy[sa] > max_policy:
                    max_sa = sa
                    max_policy = dp.policy[sa]
            view += correspond[max_sa[1]] + u' '
        view += u'\n\n'
    print view

def grid_world_policy_evaluation_demo():
    dp.reset()
    dp.policy_evaluation()
    show_value_p()
    
def grid_world_policy_iteration_demo():
    dp.reset()
    dp.policy_iteration(epsilon=0.01)
    show_policy()
    
def grid_world_value_iteration_demo():
    dp.reset()
    dp.value_iteration(epsilon=0.02)
    show_policy()
    
def grid_world_generalized_policy_iteration_demo():
    dp.reset()
    dp.generalized_policy_iteration(10, 10, 0.02, 0.01)
    show_policy()
    
grid_world_policy_evaluation_demo()
grid_world_policy_iteration_demo()
grid_world_value_iteration_demo()
grid_world_generalized_policy_iteration_demo()