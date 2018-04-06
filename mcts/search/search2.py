# -*- coding: utf-8 -*-
"""use custom policy as default policy"""
import math
import time
import random
from reversi import *


class TreeNode:

    def __init__(self, parentnode, state, parene_action, player):
        self.state = state
        self.action = parene_action
        self.parent = parentnode
        self.wins = 0.0
        self.visits = 0.0
        self.unreach = feasible_actions(state, player)
        self.children = []
        self.activeplayer = player

    def addchild(self, state, parent_action, child_player):
        node = TreeNode(self, state, parent_action, child_player)
        self.unreach.remove(parent_action)
        self.children.append(node)
        return node

    def selectchild(self):
        ordered_children = sorted(self.children, key=lambda child: child.wins / child.visits + math.sqrt(2 * math.log(self.visits) / child.visits))
        return ordered_children[0]

    def update(self, result):
        self.visits += 1.0
        self.wins += 2 * int(result == self.activeplayer) - 1
        if self.parent is not None:
            self.parent.update(result)

    def mostVisitedchild(self):
        mostVisited = self.children[0]
        for i in range(len(self.children)):
            if self.children[i].visits > mostVisited.visits:
                mostVisited = self.children[i]
        return mostVisited


def custom_default_policy(actions):
    pos_weight = np.array([
        [7, 2, 5, 4, 4, 5, 2, 7],
        [2, 1, 3, 3, 3, 3, 1, 2],
        [5, 3, 6, 5, 5, 5, 3, 5],
        [4, 3, 5, 6, 6, 5, 3, 4],
        [4, 3, 5, 6, 6, 5, 3, 4],
        [5, 3, 6, 5, 5, 5, 3, 5],
        [2, 1, 3, 3, 3, 3, 1, 2],
        [7, 2, 5, 4, 4, 5, 2, 7]
    ])
    pos_pair = [pos_weight[action[0]][action[1]] for action in actions]
    sum_pair = float(sum(pos_pair))
    pair = map(lambda x: x / sum_pair, pos_pair)
    pairs = zip(pair, actions)
    q = random.random()
    accumulate = 0.0
    for prob, action in pairs:
        if q < accumulate:
            accumulate += prob
        else:
            return action


def UCT_MCTS(state, max_time, turn, op_chosen_node):
    blocksize = 50
    node_visited = 0.0
    if op_chosen_node:
        root = op_chosen_node
        root.parent = None
        root.action = None
    else:
        root = TreeNode(None, state, None, turn)

    current_time = time.time()
    while time.time() < current_time + max_time:
        node = root
        for block in range(blocksize):
            # selection
            if len(node.unreach) == 0 and len(node.children) > 0:
                node = node.selectchild()
            # expansion
            if len(node.unreach) > 0:
                action = random.choice(node.unreach)
                temp_state = update_state(node.state.copy(), action[0], action[1], node.activeplayer)
                node = node.addchild(temp_state, action, -1*node.activeplayer)
            # simulation
            now_turn = node.activeplayer
            actions = feasible_actions(node.state, now_turn)
            temp_state = node.state.copy()
            while True:
                if len(actions) > 0:
                    action = custom_default_policy(actions)
                    temp_state = update_state(temp_state, action[0], action[1], now_turn)
                    node_visited += 1
                    now_turn = -1 * now_turn
                    actions = feasible_actions(temp_state, now_turn)
                else:
                    now_turn = -1 * now_turn
                    actions = feasible_actions(temp_state, now_turn)
                    if len(actions) == 0:
                        break
            # backpropagation
            result = game_winner(temp_state)
            node.update(result)
            node = node.parent
    return root.mostVisitedchild()
