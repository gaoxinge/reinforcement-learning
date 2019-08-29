# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import gym
from agent import MonteCarloPolicyGradientAgent


def gym_demo(env, agent, episodes):
    r, rs = 0, []
    for episode in range(episodes):
        total_reward = 0
        raw_state = env.reset()
        state = [raw_state]
       	for _ in range(500):
            # env.render()
            action = agent.choose(state)
            raw_state_, reward, done, prob = env.step(action)
            state_ = [raw_state_]
            agent.store(state, action, reward)
            state = state_
            total_reward += reward
            if done:
                # env.render()
                print('episode: {episode}, total reward: {total_reward}'.format(
                    episode=episode,
                    total_reward=total_reward
                ))
                r += total_reward
                rs.append(r / (episode+1))
                agent.learn()
                break
        else:
            # env.render()
            print('episode: {episode}, total reward: {total_reward}'.format(
                episode=episode,
                total_reward=total_reward
            ))
            r += total_reward
            rs.append(r / (episode+1))
            agent.learn()
    return rs

    
def test_gym():
    env = gym.make('MountainCar-v0').env
    agent = MonteCarloPolicyGradientAgent(2, 3)
    rs = gym_demo(env, agent, 10000)
    # plt.plot(range(10000), rs), plt.grid(), plt.show()
    plt.plot(range(10000), rs), plt.grid(), plt.savefig('../image/3.jpg')

test_gym()
