from gym import spaces
import numpy as np

class Quadratic:
    def __init__(self):
        self.observation_space = spaces.box.Box(-25 * np.ones(4), 25 * np.ones(4))
        self.action_space = spaces.box.Box(-5 * np.ones(4), 5 * np.ones(4))
        self.H = np.array([[-1, 0.5, 0, 0], [0.5, -1, 0, 0], [0, 0, -1, 0.5], [0, 0, 0.5, -1]])
        self.difficulty = 0.001

    def step(self, action):
        state = np.random.randn(4) * self.difficulty
        reward = action.dot(self.H).dot(action) + action.sum() + np.random.randn() * self.difficulty
        done = False
        info = {'name': Quadratic}
        return state, reward, done, info

    def reset(self):
        state = np.random.randn(4) * self.difficulty
        return state

    def render(self):
        raise NotImplementedError

    def seed(self, seed_value):
        self.seed = seed_value
