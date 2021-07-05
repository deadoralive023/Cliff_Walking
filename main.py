import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from agent import Agent
import gym
from collections import deque

plt.style.use('default')

env = gym.make('CliffWalking-v0')
agent = Agent(no_of_states=env.observation_space.n, no_of_actions=env.action_space.n, off_policy=False)

fig, ax = plt.subplots(2)


ax[0].set_xlabel("Episodes")
ax[0].set_ylabel("Mean Score")
# ax[0].set_ylim(-200, 100)
ax[1].set_xlabel("Episodes")
ax[1].set_ylabel("Epsilon")


# ax[1].set_ylim(0.00, 1.00)


def animate(i, ep, x_values, y_values_score, y_values_eps):
    if len(x_values) > 20:
        x_values.pop(0)
        y_values_score.pop(0)
        y_values_eps.pop(0)
        ax[0].cla()
        ax[1].cla()

    ax[0].plot(x_values, y_values_score, linestyle='-')
    ax[1].plot(x_values, y_values_eps, linestyle='-')
    pass


def cal_graph_vals(ep, scores_list, x_values, y_values_score, y_values_eps):
    mean = np.mean(scores_list)
    scores_list.clear()
    x_values.append(ep)
    y_values_score.append(mean)
    y_values_eps.append(agent.eps)


def test_agent():
    score = 0
    state = env.reset()
    done = False
    steps = 0
    while not done and steps < 50:
        steps += 1
        action = agent.max_action(state)
        state_, reward, done, _ = env.step(action)
        print(state, action, state_, reward)
        state = state_
        score += reward
    print('\n\n\n')
    return np.mean(score)


def QLearning(episodes=50000, plot_every=10):
    fig.suptitle('QLearning')
    x_values = []
    y_values_score = []
    y_values_eps = []
    scores_list = deque(maxlen=plot_every)
    count = 0
    for ep in range(1, episodes + 1):
        score = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, state_)
            state = state_
            score += reward
        scores_list.append(score)
        if (ep % plot_every) == 0:
            cal_graph_vals(ep, scores_list, x_values, y_values_score, y_values_eps)
            plt.pause(0.0001)
            plt.tight_layout()
            ani = FuncAnimation(plt.gcf(), animate, fargs=(ep, x_values, y_values_score, y_values_eps))
    plt.show()


def Sarsa(episodes=50000, plot_every=10):
    fig.suptitle('Sarsa')
    x_values = []
    y_values_score = []
    y_values_eps = []
    scores_list = deque(maxlen=100)
    for ep in range(1, episodes + 1):
        score = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, _ = env.step(action)
            action_ = agent.choose_action(state_)
            agent.learn(state, action, reward, state_, action_)
            state = state_
            score += reward
        scores_list.append(score)
        if (ep % plot_every) == 0:
            cal_graph_vals(ep, scores_list, x_values, y_values_score, y_values_eps)
            plt.pause(0.0001)
            plt.tight_layout()
            ani = FuncAnimation(plt.gcf(), animate, fargs=(ep, x_values, y_values_score, y_values_eps))
    plt.show()


Sarsa()
#QLearning()
