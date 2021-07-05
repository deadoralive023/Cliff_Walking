import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from agent import Agent
import gym
from collections import deque

plt.style.use('default')

env = gym.make('CliffWalking-v0')
agent = Agent(no_of_states=env.observation_space.n, no_of_actions=env.action_space.n)

fig, ax = plt.subplots(2)
fig.suptitle('QLearning')

ax[0].set_xlabel("Episodes")
ax[0].set_ylabel("Mean Score")
# ax[0].set_ylim(-200, 100)
ax[1].set_xlabel("Episodes")
ax[1].set_ylabel("Epsilon")


# ax[1].set_ylim(0.00, 1.00)


def animate(i, x_values, y_values_score, y_values_eps):
    ax[0].plot(x_values, y_values_score, linestyle='-')
    ax[1].plot(x_values, y_values_eps, linestyle='-')
    pass


def plot_graphs(ep, scores_list, x_values, y_values_score, y_values_eps):
    mean = np.mean(scores_list)
    scores_list.clear()
    x_values.append(ep)
    y_values_score.append(mean)
    y_values_eps.append(agent.eps)


def QLearning(episodes=50000, plot_every=10):
    x_values = deque(maxlen=episodes)
    y_values_score = deque(maxlen=episodes)
    y_values_eps = deque(maxlen=episodes)
    scores_list = deque(maxlen=100)
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
            plot_graphs(ep, scores_list, x_values, y_values_score, y_values_eps)
            plt.pause(0.0001)
            plt.tight_layout()
            ani = FuncAnimation(plt.gcf(), animate, fargs=(x_values, y_values_score, y_values_eps))
    plt.show()


def Sarsa(episodes=50000, plot_every=10):
    x_values = deque(maxlen=episodes)
    y_values_score = deque(maxlen=episodes)
    y_values_eps = deque(maxlen=episodes)
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
            plot_graphs(ep, scores_list, x_values, y_values_score, y_values_eps)
            plt.pause(0.0001)
            plt.tight_layout()
            ani = FuncAnimation(plt.gcf(), animate, fargs=(x_values, y_values_score, y_values_eps))
    plt.show()


Sarsa()
#QLearning()
