from gym import spaces
import numpy as np

class Quadratic:
    def __init__(self, m, K):
        self.synthetic = True
        self.observation_space = spaces.box.Box(-25 * np.ones(m), 25 * np.ones(m))
        self.action_space = spaces.box.Box(-5 * np.ones(m), 5 * np.ones(m))
        self.m = m
        self.K = K
        self.H = []
        z = np.array([1])
        while True:
            self.partition = np.insert(np.sort(np.random.choice(np.arange(1, m), K-1)), 0, 0)
            z = np.append(self.partition, m)[1:] - self.partition
            if z.min() > 1:
                break
        for d in z:
            #A = np.random.rand(d, d)
            #self.H.append(100 * A.dot(A.T))
            self.H.append(100 * np.ones((d,d)).astype(np.float64))
        #self.H = np.array([[-1, 0.5, 0, 0], [0.5, -1, 0, 0], [0, 0, -1, 0.5], [0, 0, 0.5, -1]])
        self.difficulty = 0.001

    def step(self, action):
        state = np.random.randn(self.m) * self.difficulty
        pp = np.append(self.partition, self.m)
        reward = 0
        for idx, ss in [(i, slice(pp[i],pp[i+1])) for i in range(self.K)]:
            reward -= action[ss].dot(self.H[idx]).dot(action[ss])
        reward += np.random.randn() * self.difficulty
        #reward += action.dot(self.H).dot(action) + action.sum() + np.random.randn() * self.difficulty
        done = False
        info = {'name': Quadratic}
        return state, reward, done, info

    def reset(self):
        state = np.random.randn(self.m) * self.difficulty
        return state

    def render(self):
        raise NotImplementedError

    def seed(self, seed_value):
        self.seed = seed_value

#Quadratic(4, 2)
