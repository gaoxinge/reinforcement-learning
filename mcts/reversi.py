# -*- coding: utf-8 -*-
import numpy as np


def game_not_end(temp_state):
    if feasible_actions(temp_state, 1) or feasible_actions(temp_state, -1):
        return True
    else:
        return False


def game_winner(state):
    black = (state == -1).sum()
    white = (state == 1).sum()
    if black < white:
        return 1
    elif white < black:
        return -1
    else:
        return 0


def feasible_actions(state, turn):
    feasible_actions = []
    for x in range(8):
        for y in range(8):
            if state[x][y] == 0 and quick_check_state(state, x, y, turn):
                feasible_actions.append((x, y))
    return feasible_actions


def quick_check_state(state, x, y, turn):
    # right
    changer = []
    for i in range(y + 1, 8, 1):
        if state[x][i] == -1 * turn:
            changer.append(i)
            if i == 7:
                changer = []
                break
        elif state[x][i] == turn:
            break
        else:
            changer = []
            break
    if len(changer) != 0:
        return True

    # left
    changel = []
    for i in range(y - 1, -1, -1):
        if state[x][i] == -1 * turn:
            changel.append(i)
            if i == 0:
                changel = []
                break
        elif state[x][i] == turn:
            break
        else:
            changel = []
            break
    if len(changel) != 0:
        return True

    # down
    changed = []
    for i in range(x + 1, 8, 1):
        if state[i][y] == -1 * turn:
            changed.append(i)
            if i == 7:
                changed = []
                break
        elif state[i][y] == turn:
            break
        else:
            changed = []
            break
    if len(changed) != 0:
        return True

    # up
    changeu = []
    for i in range(x - 1, -1, -1):
        if state[i][y] == -1 * turn:
            changeu.append(i)
            if i == 0:
                changeu = []
                break
        elif state[i][y] == turn:
            break
        else:
            changeu = []
            break
    if len(changeu) != 0:
        return True

    # lu
    changelu = []
    for i in zip(range(x - 1, -1, -1), range(y - 1, -1, -1)):
        if state[i[0]][i[1]] == -1 * turn:
            changelu.append(i)
            if i[0] == 0 or i[1] == 0:
                changelu = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changelu = []
            break
    if len(changelu) != 0:
        return True

    # ld
    changeld = []
    for i in zip(range(x + 1, 8, 1), range(y - 1, -1, -1)):
        if state[i[0]][i[1]] == -1 * turn:
            changeld.append(i)
            if i[0] == 7 or i[1] == 0:
                changeld = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changeld = []
            break
    if len(changeld) != 0:
        return True

    # ru
    changeru = []
    for i in zip(range(x - 1, -1, -1), range(y + 1, 8, 1)):
        if state[i[0]][i[1]] == -1 * turn:
            changeru.append(i)
            if i[0] == 0 or i[1] == 7:
                changeru = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changeru = []
            break
    if len(changeru) != 0:
        return True

    # rd
    changerd = []
    for i in zip(range(x + 1, 8, 1), range(y + 1, 8, 1)):
        if state[i[0]][i[1]] == -1 * turn:
            changerd.append(i)
            if i[0] == 7 or i[1] == 7:
                changerd = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changerd = []
            break
    if len(changerd) != 0:
        return True

    return False


def update_state(state, x, y, turn):
    state[x][y] = turn

    # right
    changer = []
    for i in range(y + 1, 8, 1):
        if state[x][i] == -1 * turn:
            changer.append(i)
            if i == 7:
                changer = []
                break
        elif state[x][i] == turn:
            break
        else:
            changer = []
            break
    for i in changer:
        state[x][i] = turn

    # left
    changel = []
    for i in range(y - 1, -1, -1):
        if state[x][i] == -1 * turn:
            changel.append(i)
            if i == 0:
                changel = []
                break
        elif state[x][i] == turn:
            break
        else:
            changel = []
            break
    for i in changel:
        state[x][i] = turn

    # down
    changed = []
    for i in range(x + 1, 8, 1):
        if state[i][y] == -1 * turn:
            changed.append(i)
            if i == 7:
                changed = []
                break
        elif state[i][y] == turn:
            break
        else:
            changed = []
            break
    for i in changed:
        state[i][y] = turn

    # up
    changeu = []
    for i in range(x - 1, -1, -1):
        if state[i][y] == -1 * turn:
            changeu.append(i)
            if i == 0:
                changeu = []
                break
        elif state[i][y] == turn:
            break
        else:
            changeu = []
            break
    for i in changeu:
        state[i][y] = turn

    # lu
    changelu = []
    for i in zip(range(x - 1, -1, -1), range(y - 1, -1, -1)):
        if state[i[0]][i[1]] == -1 * turn:
            changelu.append(i)
            if i[0] == 0 or i[1] == 0:
                changelu = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changelu = []
            break
    for i in changelu:
        state[i[0]][i[1]] = turn

    # ld
    changeld = []
    for i in zip(range(x + 1, 8, 1), range(y - 1, -1, -1)):
        if state[i[0]][i[1]] == -1 * turn:
            changeld.append(i)
            if i[0] == 7 or i[1] == 0:
                changeld = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changeld = []
            break
    for i in changeld:
        state[i[0]][i[1]] = turn

    # ru
    changeru = []
    for i in zip(range(x - 1, -1, -1), range(y + 1, 8, 1)):
        if state[i[0]][i[1]] == -1 * turn:
            changeru.append(i)
            if i[0] == 0 or i[1] == 7:
                changeru = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changeru = []
            break
    for i in changeru:
        state[i[0]][i[1]] = turn

    # rd
    changerd = []
    for i in zip(range(x + 1, 8, 1), range(y + 1, 8, 1)):
        if state[i[0]][i[1]] == -1 * turn:
            changerd.append(i)
            if i[0] == 7 or i[1] == 7:
                changerd = []
                break
        elif state[i[0]][i[1]] == turn:
            break
        else:
            changerd = []
            break
    for i in changerd:
        state[i[0]][i[1]] = turn

    return state


class reversi(object):

    def __init__(self):
        self.turn = -1
        self.state = np.zeros([8, 8], int)
        self.state[3][3] = 1
        self.state[3][4] = -1
        self.state[4][3] = -1
        self.state[4][4] = 1

    def move(self, x, y):
        return self.update_state(x, y)

    def feasible_move_show(self):
        feasible_move = []
        for x in range(8):
            for y in range(8):
                if self.state[x][y] == 0 and quick_check_state(self.state, x, y, self.turn):
                    feasible_move.append((x, y))
        return feasible_move

    def update_state(self, x, y):
        if (x, y) not in self.feasible_move_show():
            return False
        self.state[x][y] = self.turn  # update the change
        self.state = self.complete_update_state(self.state, x, y, self.turn)
        return True

    def complete_update_state(self, state, x, y, turn):
        # right
        changer = []
        for i in range(y+1, 8, 1):
            if state[x][i] == -1*turn:
                changer.append(i)
                if i == 7:
                    changer = []
                    break
            elif state[x][i] == turn:
                break
            else:
                changer = []
                break
        for i in changer:
            state[x][i] = turn

        # left
        changel = []
        for i in range(y-1, -1, -1):
            if state[x][i] == -1*turn:
                changel.append(i)
                if i == 0:
                    changel = []
                    break
            elif state[x][i] == turn:
                break
            else:
                changel = []
                break
        for i in changel:
            state[x][i] = turn

        # down
        changed = []
        for i in range(x+1, 8, 1):
            if state[i][y] == -1*turn:
                changed.append(i)
                if i == 7:
                    changed = []
                    break
            elif state[i][y] == turn:
                break
            else:
                changed = []
                break
        for i in changed:
            state[i][y] = turn

        # up
        changeu = []
        for i in range(x-1, -1, -1):
            if state[i][y] == -1*turn:
                changeu.append(i)
                if i == 0:
                    changeu = []
                    break
            elif state[i][y] == turn:
                break
            else:
                changeu = []
                break
        for i in changeu:
            state[i][y] = turn

        # lu
        changelu = []
        for i in zip(range(x-1, -1, -1), range(y-1, -1, -1)):
            if state[i[0]][i[1]] == -1*turn:
                changelu.append(i)
                if i[0] == 0 or i[1] == 0:
                    changelu = []
                    break
            elif state[i[0]][i[1]] == turn:
                break
            else:
                changelu = []
                break
        for i in changelu:
            state[i[0]][i[1]] = turn

        # ld
        changeld = []
        for i in zip(range(x+1, 8, 1), range(y-1, -1, -1)):
            if state[i[0]][i[1]] == -1*turn:
                changeld.append(i)
                if i[0] == 7 or i[1] == 0:
                    changeld = []
                    break
            elif state[i[0]][i[1]] == turn:
                break
            else:
                changeld = []
                break
        for i in changeld:
            state[i[0]][i[1]] = turn

        # ru
        changeru = []
        for i in zip(range(x-1, -1, -1), range(y+1, 8, 1)):
            if state[i[0]][i[1]] == -1*turn:
                changeru.append(i)
                if i[0] == 0 or i[1] == 7:
                    changeru = []
                    break
            elif state[i[0]][i[1]] == turn:
                break
            else:
                changeru = []
                break
        for i in changeru:
            state[i[0]][i[1]] = turn

        # rd
        changerd = []
        for i in zip(range(x+1, 8, 1), range(y+1, 8, 1)):
            if state[i[0]][i[1]] == -1*turn:
                changerd.append(i)
                if i[0] == 7 or i[1] == 7:
                    changerd = []
                    break
            elif state[i[0]][i[1]] == turn:
                break
            else:
                changerd = []
                break
        for i in changerd:
            state[i[0]][i[1]] = turn

        if len(changel + changer + changed + changeu + changeld + changelu + changerd + changeru) == 0:
            return False
        else:
            return state

    def winorlose(self):
        Black = (self.state == -1).sum()
        White = (self.state == 1).sum()
        if Black > White:
            print 'Black %d , White %d' % (Black, White)
            print 'Black win'
            return -1
        elif Black < White:
            print 'Black %d , White %d' % (Black, White)
            print 'White win'
            return 1
        elif Black == White:
            print 'Black %d , White %d' % (Black, White)
            print 'Double win'
            return 0
