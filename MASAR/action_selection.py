import numpy as np


def greedy(q):
    return np.random.choice(np.flatnonzero(q == np.max(q)))


def eps_greedy(q, actions, epsilon=0.05):
    if np.random.random() < epsilon:
        idx = np.random.randint(len(actions))
    else:
        idx = greedy(q)
    return idx


def ucb(q, c, step, N):
    ucb_eq = q + c * np.sqrt(np.log(step) / N)
    return greedy(ucb_eq)

def Boltzmann(q, t=0.4):
    return np.exp(q / t) / np.sum(np.exp(q / t))