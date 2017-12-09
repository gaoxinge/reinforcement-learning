# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from environment import Maze
from agent import MonteCarloAgent, SaraAgent, SaraLambdaAgent, QLearnAgent

def monte_carlo_demo(env, agent, episodes):
    r, rs = 0, []
    for episode in range(episodes):
        total_reward = 0
        state = env.reset()
        while True:
            env.render()
            action = agent.choose(state)
            state_, reward, done = env.step(action)
            agent.store(state, action, reward)
            state = state_
            total_reward += reward
            if done:
                print 'episode: {episode}, total reward: {total_reward}'.format(episode=episode, total_reward=total_reward)
                r += total_reward
                rs.append(r / (episode+1))
                env.render()
                agent.learn()
                break
    return rs
                
def sara_demo(env, agent, episodes):
    r, rs = 0, []
    for episode in range(episodes):
        total_reward = 0
        state = env.reset()
        action = agent.choose(state)
        while True:
            env.render()
            state_, reward, done = env.step(action)
            action_ = agent.choose(state_)
            agent.learn(state, action, reward, state_, action_, done)
            state, action = state_, action_
            total_reward += reward
            if done:
                print 'episode: {episode}, total reward: {total_reward}'.format(episode=episode, total_reward=total_reward)
                r += total_reward
                rs.append(r / (episode+1))
                env.render()
                break
    return rs
    
def q_learn_demo(env, agent, episodes):
    r, rs = 0, []
    for episode in range(episodes):
        total_reward = 0
        state = env.reset()
        while True:
            env.render()
            action = agent.choose(state)
            state_, reward, done = env.step(action)
            agent.learn(state, action, reward, state_, done)
            state = state_
            total_reward += reward
            if done:
                print 'episode: {episode}, total reward: {total_reward}'.format(episode=episode, total_reward=total_reward)
                r += total_reward
                rs.append(r / (episode+1))
                env.render()
                break
    return rs

def test_monte_carlo():
    env = Maze()
    agent = MonteCarloAgent(act_n=4)
    rs = monte_carlo_demo(env, agent, 2000)
    plt.plot(range(2000), rs), plt.grid(), plt.show()

def test_sara():
    env = Maze()
    agent = SaraAgent(act_n=4)
    rs = sara_demo(env, agent, 2000)
    plt.plot(range(2000), rs), plt.grid(), plt.show()
    
def test_sara_lambda():
    env = Maze()
    agent = SaraLambdaAgent(act_n=4)
    rs = sara_demo(env, agent, 2000)
    plt.plot(range(2000), rs), plt.grid(), plt.show()

def test_q_learn():
    env = Maze()
    agent = QLearnAgent(act_n=4)
    rs = q_learn_demo(env, agent, 2000)
    plt.plot(range(2000), rs), plt.grid(), plt.show()