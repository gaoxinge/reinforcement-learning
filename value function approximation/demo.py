# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import gym
from agent import SarsaAgent
from tile_coding import IHT, tiles
from linear_model import LinearModel


feat_n = 2048
tile_n = 8
iht = IHT(feat_n)


max_position = 0.6
min_position = -1.2
max_velocity = 0.07
min_velocity = -0.07


def gym_demo(env, agent, episodes):
    r, rs = 0, []
    for episode in range(episodes):
        total_reward = 0
        state = env.reset()
        action = agent.choose(state)
        while True:
            env.render()
            state_, reward, done, prob = env.step(action)
            action_ = agent.choose(state_)
            agent.learn(state, action, reward, state_, action_, done)
            state, action = state_, action_
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


def get_feature(state, action):
    hash_table = iht
    num_tilings = tile_n
    position_scale = num_tilings / (max_position - min_position)
    velocity_scale = num_tilings / (max_velocity - min_velocity)
    position, velocity = state
    # print(position, velocity, action, num_tilings, position_scale, velocity_scale)

    indexs = tiles(
        hash_table,
        num_tilings,
        [position_scale * position, velocity_scale * velocity],
        [action]
    )
    feature = [0 for _ in range(feat_n)]
    for index in indexs:
        feature[index] = 1
    return feature


def test_gym():
    env = gym.make('MountainCar-v0')
    linear_func = LinearModel(feat_n, get_feature)
    agent = SarsaAgent(act_n=3, linear_func=linear_func)
    rs = gym_demo(env, agent, 500)
    plt.plot(range(500), rs), plt.grid(), plt.show()


test_gym()
