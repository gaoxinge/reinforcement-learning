# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import gym
from agent import SarsaLambdaAgent


def gym_demo(env, agent, episodes):
    r, rs = 0, []
    for episode in range(episodes):
        total_reward = 0
        state = env.reset()
        action = agent.choose(state)
        while True:
            state_, reward, done, prob = env.step(action)
            action_ = agent.choose(state_)
            agent.learn(state, action, reward, state_, action_, done)
            state, action = state_, action_
            total_reward += reward
            if done:
                print('episode: {episode}, total reward: {total_reward}'.format(
                    episode=episode,
                    total_reward=total_reward
                ))
                r += total_reward
                rs.append(r / (episode + 1))
                break
    return rs


def test_gym():
    env = gym.make('FrozenLake-v0')
    agent = SarsaLambdaAgent(act_n=4)
    rs = gym_demo(env, agent, 100000)
    plt.plot(range(100000), rs), plt.grid(), plt.show()


test_gym()
