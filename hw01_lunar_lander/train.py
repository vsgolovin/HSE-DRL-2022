from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
BUFFER_SIZE = 2048


class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0  # Do not change
        self.model = create_model(state_dim, action_dim)
        self.num_actions = action_dim
        self.target_model = create_model(state_dim, action_dim)
        self.update_target_network()
        self.target_model.eval()
        self.buffer = deque([], maxlen=BUFFER_SIZE)
        self.optimizer = Adam(params=self.model.parameters(),
                              lr=LEARNING_RATE)

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen.
        # It will remove old experience automatically.
        self.buffer.append(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor.
        #    This will work faster.
        inds = np.random.randint(0, len(self.buffer), size=BATCH_SIZE)
        return [self.buffer[ind] for ind in inds]

    def train_step(self, batch):
        # Use batch to update DQN's network.
        s, a, s1, r, d = map(
            lambda t: torch.tensor(np.asarray(t)),
            zip(*batch)
        )
        Q_sa = self.model(s).gather(1, a.view(-1, 1))
        Q_s1a1 = self.target_model(s1).max(1)[0]
        with torch.no_grad():
            target = r.to(torch.float32)
            target[~d] += GAMMA * Q_s1a1[~d]
        loss = F.mse_loss(Q_sa, target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here.
        # You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor
        # and then turn an action to a numpy array.
        state = np.array(state)
        state = torch.tensor(state)
        with torch.no_grad():
            action = self.model(state).max(0)[1]
        return action.cpu().numpy()

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


def create_model(state_dim, action_dim):
    return nn.Sequential(
        nn.Linear(state_dim, state_dim),
        nn.ReLU(),
        nn.Linear(state_dim, action_dim),
    )


if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0],
              action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()

    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()


    for i in range(TRANSITIONS):
        #Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()
