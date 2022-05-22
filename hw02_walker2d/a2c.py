"""
Weird version of the A2C algorithm for a discrete action set.
Samples all the generated trajectories once, i.e., does not use all the data.
"""
import gym
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
import numpy as np


class Trajectory:
    def __init__(self, state_dim: int, max_length: int = 1000):
        self.states = np.zeros((max_length, state_dim))
        self.actions = np.zeros(max_length, dtype='int')
        self.rewards = np.zeros(max_length)
        self.max_length = max_length
        self.pos = 0

    def __len__(self) -> int:
        return self.pos

    @property
    def is_full(self) -> bool:
        return self.pos == self.max_length

    def append(self, state: np.ndarray, action: int, reward: float):
        assert not self.is_full
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.pos += 1

    def get_SAR(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.states[:self.pos], self.actions[:self.pos],
                self.rewards[:self.pos])

    def compute_lambda_returns(self, values: np.ndarray,
                               lam: float = 0.95,
                               gamma: float = 0.99) -> np.ndarray:
        G = self.rewards[:self.pos].copy()
        i = self.pos
        G_prev = 0.0
        v_prev = 0.0
        while i > 0:
            G[i - 1] += gamma * ((1 - lam) * v_prev + lam * G_prev)
            i -= 1
            v_prev = values[i]
            G_prev = G[i]
        return G


class Actor(nn.Module):
    def __init__(self, state_dim: int, num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            # nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, state: torch.tensor) -> torch.tensor:
        return self.layers(state)

    @torch.no_grad()
    def act(self, state: torch.tensor) -> int:
        logits = self.layers(state)
        m = Categorical(logits=logits)
        return int(m.sample().cpu().numpy())

    def evaluate_episode(self, env: gym.Env,
                         max_length: int = 1000,
                         render: bool = False) -> Trajectory:
        state = env.reset()
        done = False
        trajectory = Trajectory(state_dim=env.observation_space.shape[0],
                                max_length=max_length)
        while not done and not trajectory.is_full:
            if render:
                env.render()
            action = self.act(torch.tensor(state))
            next_state, reward, done, _ = env.step(action)
            trajectory.append(state, action, reward)
            state = next_state
        return trajectory


class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state: torch.tensor) -> torch.tensor:
        return self.layers(state)

    @torch.no_grad()
    def get_value(self, state: torch.tensor) -> float:
        return float(self.layers(state).cpu().item())


class A2C:
    def __init__(self, state_dim: int, num_actions: int,
                 lr_actor: float, lr_critic: float):
        self.actor = Actor(state_dim, num_actions)
        self.actor_optimizer = torch.optim.RMSprop(
            params=self.actor.parameters(),
            lr=lr_actor
        )
        self.critic = Critic(state_dim)
        self.critic_optimizer = torch.optim.RMSprop(
            params=self.critic.parameters(),
            lr=lr_critic
        )
        self.state_dim = state_dim
        self.num_actions = num_actions

    def update(self, env: gym.Env, batch_size: int,
               num_episodes: int = 20,
               max_episode_length: int = 1000,
               lam: float = 0.95,
               gamma: float = 0.99,
               beta: float = 0.01) -> tuple[np.ndarray, float, float]:
        self.actor.train()
        self.critic.train()

        # compute trajectories
        trajectories = [self.actor.evaluate_episode(
            env, max_episode_length, False) for _ in range(num_episodes)]
        num_transitions = sum(len(t) for t in trajectories)
        while num_transitions < batch_size:
            t = self.actor.evaluate_episode(env, max_episode_length, False)
            trajectories.append(t)
            num_transitions += len(t)

        # concatenate S, A and R arrays, compute advantage
        states = np.zeros((num_transitions, self.state_dim))
        actions = np.zeros(num_transitions, dtype='int')
        rewards = np.zeros(num_transitions)
        adv = np.zeros_like(rewards)
        returns = np.zeros(len(trajectories))  # total rewards for each episode
        i1 = 0
        for j, t in enumerate(trajectories):
            i2 = min(i1 + len(t), num_transitions)
            states[i1:i2], actions[i1:i2], rewards[i1:i2] = t.get_SAR()
            returns[j] = np.sum(rewards[i1:i2])
            with torch.no_grad():
                values = self.critic(
                    torch.tensor(states[i1:i2], dtype=torch.float32)
                    ).cpu().ravel().numpy()
            G = t.compute_lambda_returns(values, lam, gamma)
            adv[i1:i2] = G - values
            i1 = i2

        # sample a batch
        idx = np.random.randint(0, num_transitions, size=batch_size)
        states = torch.tensor(states[idx], dtype=torch.float32)
        actions = torch.tensor(actions[idx])
        adv = torch.tensor(adv[idx], dtype=torch.float32)

        # update actor
        output = self.actor(states)
        log_probs = F.log_softmax(output, dim=1)
        log_prob = torch.gather(
            input=log_probs,
            dim=1,
            index=actions.view(-1, 1)
        )
        if beta > 0:
            entropy = -(log_probs * F.softmax(output, dim=1)).sum(1, keepdim=True)
        else:
            entropy = 0
        actor_loss = torch.mean(-log_prob.ravel() * adv - beta * entropy)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update critic
        values = self.critic(states)
        critic_loss = F.smooth_l1_loss(values.ravel(), adv)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor.eval()
        self.critic.eval()
        return returns, actor_loss.item(), critic_loss.item()

    def save(self):
        torch.save(self.actor.layers, "agent.pkl")


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    LR_ACTOR = 5e-3
    LR_CRITIC = 5e-4
    BATCH_SIZE = 512
    EPISODES_PER_UPDATE = 4
    ITERATIONS = 1000
    MAX_STEPS = 1000
    GAMMA = 0.99
    LAMBDA = 0.95
    BETA = 0.01

    env = gym.make('LunarLander-v2')
    agent = A2C(env.observation_space.shape[0], env.action_space.n,
                lr_actor=LR_ACTOR, lr_critic=LR_CRITIC)
    mean_returns = np.zeros(ITERATIONS)
    best_returns = np.zeros(ITERATIONS)
    actor_loss = np.zeros(ITERATIONS)
    critic_loss = np.zeros(ITERATIONS)

    for i in range(1, ITERATIONS + 1):
        returns, actor_loss[i - 1], critic_loss[i - 1] =\
            agent.update(
                env=env,
                batch_size=BATCH_SIZE,
                num_episodes=EPISODES_PER_UPDATE,
                max_episode_length=MAX_STEPS,
                lam=LAMBDA,
                gamma=GAMMA,
                beta=BETA
            )
        mean_returns[i - 1] = np.mean(returns)
        best_returns[i - 1] = np.max(returns)

        if i % 10 == 0:
            print(f'Iteration {i}, average reward {mean_returns[i-10:i].mean()}')
            agent.save()

    t = np.arange(1, ITERATIONS + 1)
    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, sharex=True)
    fig.set_size_inches(6.0, 9.0)
    ax1.plot(t, mean_returns, label='avg.')
    ax1.plot(t, best_returns, label='best')
    ax1.legend()
    ax1.set_ylabel('Reward')
    ax2.plot(t, actor_loss)
    ax2.set_ylabel('Actor loss')
    ax3.plot(t, critic_loss)
    ax3.set_ylabel('Critic loss')
    ax3.set_xlabel('Epoch')
    plt.tight_layout()
    plt.show()
