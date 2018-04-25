# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import gym
from agent import QLearnAgent
from model import Model


def gym_demo(env, agent, episodes):
    r, rs = 0, []
    for episode in range(episodes):
        total_reward = 0
        state = env.reset()
        while True:
            env.render()
            action = agent.choose(state)
            state_, reward, done, prob = env.step(action)
            agent.learn(state, action, reward, state_, done)
            state = state_
            total_reward += reward
            if done:
                env.render()
                print('episode: {episode}, total reward: {total_reward}'.format(
                    episode=episode,
                    total_reward=total_reward
                ))
                r += total_reward
                rs.append(r / (episode + 1))
                break
    return rs
    


def test_gym():
    env = gym.make('MountainCar-v0').env
    linear_func = LinearModel(feat_n, get_feature)
    agent = QLearnAgent(state_n=2, act_n=3)
    rs = gym_demo(env, agent, 200)
    plt.plot(range(200), rs), plt.grid(), plt.show()


test_gym()