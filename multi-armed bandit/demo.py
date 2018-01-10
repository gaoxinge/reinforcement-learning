# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from arm import BernoulliArm
from algorithm import ABTest, EpsilonGreedy, Softmax, UCB1


arms = [BernoulliArm(0.1), BernoulliArm(0.1), BernoulliArm(0.1), 
        BernoulliArm(0.1), BernoulliArm(0.9)]

def algorithm_demo(arms, algorithm, episodes):
    r, s = 0.0, []
    for episode in range(episodes):
        arm = algorithm.pull()
        reward = arms[arm].step()
        algorithm.update(arm, reward)
        r += reward
        s.append(r / (episode + 1))
    return s


########
# demo #
########
def ab_test_demo():
    ab_test = ABTest(5)
    s = algorithm_demo(arms, ab_test, 1000)
    plt.plot(range(1000), s), plt.grid(), plt.show()


def epsilon_greedy_demo():
    algorithms = [ABTest(5), EpsilonGreedy(5, 0.1), EpsilonGreedy(5, 0.25), 
                  EpsilonGreedy(5, 0.75), EpsilonGreedy(5, 1)]
    ss = [algorithm_demo(arms, algorithm, 1000) for algorithm in algorithms]
    [plt.plot(range(1000), ss[i], label=str(algorithms[i])) for i in range(5)]
    plt.legend(), plt.grid(), plt.show()


def all_demo():
    algorithms = [ABTest(5), EpsilonGreedy(5, 0.1), Softmax(5, 0.1), UCB1(5)]
    ss = [algorithm_demo(arms, algorithm, 1000) for algorithm in algorithms]
    [plt.plot(range(1000), ss[i], label=str(algorithms[i])) for i in range(4)]
    plt.legend(), plt.grid(), plt.show()