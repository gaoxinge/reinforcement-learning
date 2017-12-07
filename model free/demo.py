# -*- coding: utf-8 -*-
from environment import MazeEnvironment
from agent import MonteCarloPolicyGradientAgent

def play(env, agent, episodes):
    for episode in range(episodes):
        r = 0
        state = env.reset()
        while True:
            env.render()
            action = agent.choose(state)
            state_, reward, done = env.step(action)
            agent.store(state, action, reward)
            r += reward
            state = state_
            if done:
                print 'episode: {episode}, total reward: {r}'.format(episode=episode, r=r)
                env.render()
                agent.learn()
                break

env = MazeEnvironment()
agent = MonteCarloPolicyGradientAgent(act_n=4)
play(env, agent, 300)