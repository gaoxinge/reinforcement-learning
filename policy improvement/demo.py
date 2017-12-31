# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from environment import Maze
from agent import SarsaLambdaAgentWithEpsilonGreedy, SarsaLambdaAgentWithUCB1

                
def sarsa_demo(env, agent, episodes):
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
    

def test_sarsa_lambda_with_epsilon_greedy():
    env = Maze()
    agent = SarsaLambdaAgentWithEpsilonGreedy(act_n=4)
    rs = sarsa_demo(env, agent, 2000)
    plt.plot(range(2000), rs), plt.grid(), plt.show()
    
def test_sarsa_lambda_with_ucb1():
    env = Maze()
    agent = SarsaLambdaAgentWithUCB1(act_n=4)
    rs = sarsa_demo(env, agent, 2000)
    plt.plot(range(2000), rs), plt.grid(), plt.show()