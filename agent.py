import random
import numpy as np
from collections import defaultdict

random.seed(10)


class Agent:
    def __init__(self, no_of_states, no_of_actions, alpha=0.01, gamma=1.0, eps=1.0, eps_min=0.001,
                 eps_decay=0.9999):
        self.no_of_states = no_of_states
        self.no_of_actions = no_of_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.Q = defaultdict(lambda: np.zeros(no_of_actions))

    def choose_action(self, state):
        if random.random() < self.eps:
            return random.choice(np.arange(self.no_of_actions))
        else:
            return np.argmax(self.Q[state])

    # Q[s][a] = Q[s][a] + alpha * (reward + gamma * maxAV(s_) - Q[s][a])
    def learn(self, s, a, r, s_):
        self.Q[s][a] = self.Q[s][a] + (self.alpha * ((r + self.gamma * np.max(self.Q[s_])) - self.Q[s][a]))
        self.decay_eps()

    def decay_eps(self):
        self.eps = self.eps * self.eps_decay if self.eps > self.eps_min else self.eps_min
