# -*- coding: utf-8 -*-
import random
import math

__all__ = [
    "SaraLambdaAgentWithEpsilonGreedy",
    "SarsaLambdaAgentWithSoftmax",
    "SarsaLambdaAgentWithUCB1"
]

####################
# utility function #
####################
def _all_equal(actions):
    for action in actions:
        for action_ in actions:
            if actions[action] != actions[action_]:
                return False
    return True


#########
# agent #
#########
class SarsaLambdaAgentWithEpsilonGreedy(object):

    def __init__(self, act_n, gamma=0.9, epsilon=0.1, lambda_=0.9):
        self.act_n = act_n
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_
        self.value_q = {}
        self.value_n = {}
        self.eligibility_trace = {}

    def _initialize_state(self, state):
        if state not in self.value_q:
            self.value_q[state] = {a: 0.0 for a in range(self.act_n)}
            self.value_n[state] = {a: 0.0 for a in range(self.act_n)}
            self.eligibility_trace[state] = {a: 0.0 for a in range(self.act_n)}

    def choose(self, state):
        self._initialize_state(state)
        actions = self.value_q[state]
        if random.random() < self.epsilon or _all_equal(actions):
            return random.choice(range(self.act_n))
        else:
            m = max(actions.itervalues())
            for k, v in actions.iteritems():
                if v == m:
                    return k

    def learn(self, state, action, reward, state_, action_, done):
        self.value_n[state][action] += 1
        predict = self.value_q[state][action]
        target = reward if done else reward + self.gamma * self.value_q[state_][action_]
        error = target - predict
        self.eligibility_trace[state][action] += 1
        for s in self.eligibility_trace:
            for a in self.eligibility_trace[s]:
                self.value_q[s][a] += self.eligibility_trace[s][a] * error / (self.value_n[s][a] + 1e-6)
                self.eligibility_trace[s][a] *= self.gamma * self.lambda_


class SarsaLambdaAgentWithSoftmax(object):

    def __init__(self, act_n, gamma=0.9, temperature=0.1, lambda_=0.9):
        self.act_n = act_n
        self.gamma = gamma
        self.temperature = temperature
        self.lambda_ = lambda_
        self.value_q = {}
        self.value_n = {}
        self.eligibility_trace = {}

    def _initialize_state(self, state):
        if state not in self.value_q:
            self.value_q[state] = {a: 0.0 for a in range(self.act_n)}
            self.value_n[state] = {a: 0.0 for a in range(self.act_n)}
            self.eligibility_trace[state] = {a: 0.0 for a in range(self.act_n)}

    def choose(self, state):
        self._initialize_state(state)
        actions_n = self.value_n[state]
        actions_q = self.value_q[state]
        for a in range(self.act_n):
            if actions_n[a] == 0.0:
                return a
        s = 0.0
        for v in actions_q.itervalues():
            if v / self.temperature > 700.0:
                s += math.exp(700.0)
            else:
                s += math.exp(v / self.temperature)
        actions_prob = {}
        for a in actions_q:
            if actions_q[a] / self.temperature > 700.0:
                actions_prob[a] = math.exp(700.0)
            else:
                actions_prob[a] = math.exp(actions_q[a] / self.temperature)
        m = max(actions_prob.itervalues())
        for k, v in actions_prob.iteritems():
            if v == m:
                return k

    def learn(self, state, action, reward, state_, action_, done):
        self.value_n[state][action] += 1
        predict = self.value_q[state][action]
        target = reward if done else reward + self.gamma * self.value_q[state_][action_]
        error = target - predict
        self.eligibility_trace[state][action] += 1
        for s in self.eligibility_trace:
            for a in self.eligibility_trace[s]:
                self.value_q[s][a] += self.eligibility_trace[s][a] * error / (self.value_n[s][a] + 1e-6)
                self.eligibility_trace[s][a] *= self.gamma * self.lambda_


class SarsaLambdaAgentWithUCB1(object):

    def __init__(self, act_n, gamma=0.9, lambda_=0.9):
        self.act_n = act_n
        self.gamma = gamma
        self.lambda_ = lambda_
        self.value_q = {}
        self.value_n = {}
        self.eligibility_trace = {}

    def _initialize_state(self, state):
        if state not in self.value_q:
            self.value_q[state] = {a: 0.0 for a in range(self.act_n)}
            self.value_n[state] = {a: 0.0 for a in range(self.act_n)}
            self.eligibility_trace[state] = {a: 0.0 for a in range(self.act_n)}

    def choose(self, state):
        self._initialize_state(state)
        actions_n = self.value_n[state]
        actions_q = self.value_q[state]
        for a in range(self.act_n):
            if actions_n[a] == 0.0:
                return a
        t = sum(actions_n.itervalues())
        actions_ucb = {a: actions_q[a] + math.sqrt((2 * math.log(t)) / actions_n[a])
                       for a in range(self.act_n)}
        m = max(actions_ucb.itervalues())
        for k, v in actions_ucb.iteritems():
            if v == m:
                return k

    def learn(self, state, action, reward, state_, action_, done):
        self.value_n[state][action] += 1
        predict = self.value_q[state][action]
        target = reward if done else reward + self.gamma * self.value_q[state_][action_]
        error = target - predict
        self.eligibility_trace[state][action] += 1
        for s in self.eligibility_trace:
            for a in self.eligibility_trace[s]:
                self.value_q[s][a] += self.eligibility_trace[s][a] * error / (self.value_n[s][a] + 1e-6)
                self.eligibility_trace[s][a] *= self.gamma * self.lambda_