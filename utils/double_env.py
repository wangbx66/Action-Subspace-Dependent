import gym
import numpy as np

class Double:
    def __init__(self, env_name, replicate=2):
        self.envs = []
        self.replicate = replicate
        for i in range(self.replicate):
            env = gym.make(env_name)
            self.envs.append(env)
        self.observation_space = gym.spaces.box.Box(np.tile(self.envs[-1].observation_space.low, 2), np.tile(self.envs[-1].observation_space.high, 2))
        self.action_space = gym.spaces.box.Box(np.tile(self.envs[-1].action_space.low, 2), np.tile(self.envs[-1].action_space.high, 2))

    def step(self, action):
        action_dim = int(action.shape[0] / self.replicate)
        states = []
        reward = 0
        done = False
        for i in range(self.replicate):
            action_this = action[i*action_dim:(i+1)*action_dim]
            #import pdb;pdb.set_trace()
            state_this, reward_this, done_this, info = self.envs[i].step(action_this)
            states.append(state_this)
            reward += reward_this
            done |= done_this
        states = np.concatenate(states)
        return states, reward, done, info

    def reset(self):
        states = []
        for i in range(self.replicate):
            state_this = self.envs[i].reset()
            states.append(state_this)
        states = np.concatenate(states)
        return states

    def render(self):
        self.envs[-1].render()

    def seed(self, seed_value):
        for i in range(self.replicate):
            self.envs[i].seed(seed_value)

    def take_action(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

