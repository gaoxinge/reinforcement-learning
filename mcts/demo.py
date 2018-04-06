# -*- coding: utf-8 -*-
from reversi import *
from search.search1 import UCT_MCTS as UCT_MCTS1
from search.search2 import UCT_MCTS as UCT_MCTS2


def demo(ai):
    Reversi = reversi()
    op_chosen_node = None
    op_chosen_node2 = None
    ai_pos = None
    ai_pos2 = None
    while Reversi.feasible_move_show():
        if Reversi.turn == ai:
            if not op_chosen_node:
                ai_pos = UCT_MCTS1(Reversi.state.copy(), 5, ai, op_chosen_node)
            else:
                ai_pos = UCT_MCTS1(None, 5, ai, op_chosen_node)
            print Reversi.turn, ai_pos.action[0], ai_pos.action[1]
            Reversi.move(ai_pos.action[0], ai_pos.action[1])
            print Reversi.state
            if ai_pos2:
                for node in ai_pos2.children:
                    if node.action == (ai_pos.action[0], ai_pos.action[1]):
                        op_chosen_node2 = node
        else:
            if not op_chosen_node2:
                ai_pos2 = UCT_MCTS2(Reversi.state.copy(), 5, -1 * ai, op_chosen_node2)
            else:
                ai_pos2 = UCT_MCTS2(None, 5, -1 * ai, op_chosen_node2)
            print Reversi.turn, ai_pos2.action[0], ai_pos2.action[1]
            Reversi.move(ai_pos2.action[0], ai_pos2.action[1])
            print Reversi.state
            if ai_pos:
                for node in ai_pos.children:
                    if node.action == (ai_pos2.action[0], ai_pos2.action[1]):
                        op_chosen_node = node
        Reversi.turn = -1 * Reversi.turn
    return Reversi.winorlose()


def main():
    uct_mcts1_first = 5
    uct_mcts2_first = 5
    uct_mcts1_win = 0
    uct_mcts2_win = 0
    double_win = 0

    for i in range(uct_mcts1_first):
        result = demo(-1)
        if result == -1:
            uct_mcts1_win += 1
        elif result == 1:
            uct_mcts2_win += 1
        else:
            double_win += 1

    for i in range(uct_mcts2_first):
        result = demo(1)
        if result == -1:
            uct_mcts2_win += 1
        elif result == 1:
            uct_mcts1_win += 1
        else:
            double_win += 1

    print 'uct_mcts1: %s, uct_mcts2: %s, double win: %s' % (uct_mcts1_win, uct_mcts2_win, double_win)


if __name__ == '__main__':
    main()
