import os
import numpy as np
from gym import make
import torch
from torch.distributions.categorical import Categorical


class Agent:
    def __init__(self):
        path = os.path.join(os.getcwd(), 'agent.pkl')
        self.model = torch.load(path)
        self.model.eval()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float()
            logits = self.model(state)
            m = Categorical(logits=logits)
            return m.sample().item()

    def reset(self):
        pass


EPISODES = 20
RENDER = False

env = make('LunarLander-v2')
agent = Agent()
rewards = np.zeros(EPISODES)
for i in range(EPISODES):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        if RENDER:
            env.render()
        state, reward, done, _ = env.step(action)
        rewards[i] += reward
    print(rewards[i])

print('Rewards:')
print(f'  - mean: {rewards.mean()}')
print(f'  - max: {rewards.max()}')
print(f'  - min: {rewards.min()}')
