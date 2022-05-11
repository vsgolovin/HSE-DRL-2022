from gym import make
import torch
from agent import Agent

device = torch.device('cpu')
agent = Agent()
agent.model = agent.model.to(device)

env = make('LunarLander-v2')
iterations = 20
avg_reward = 0.0
for _ in range(iterations):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        env.render()
        state, reward, done, _ = env.step(agent.act(state))
        total_reward += reward
    print(f'reward = {total_reward}', end='\r')
    avg_reward += total_reward

avg_reward /= iterations
print()
print(avg_reward)
