import pybullet_envs
from gym import make
import numpy as np
from agent import Agent


EPISODES = 50
RENDER = False

env = make('Walker2DBulletEnv-v0')
if RENDER:
    env.render()
env.reset()
agent = Agent()
print(agent.sigma)
rewards = np.zeros(EPISODES)
for i in range(EPISODES):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        rewards[i] += reward
#    print(rewards[i])
env.close()

print('Reward:')
print(f'  max = {rewards.max():.2f}')
print(f'  mean = {rewards.mean():.2f}')
print(f'  std = {rewards.std():.2f}')

