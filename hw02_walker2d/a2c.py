from warnings import warn
import gym
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
import numpy as np

LR_ACTOR = 2e-4
LR_CRITIC = 1e-4


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
        G = self.rewards[:self.pos]
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
    def __init__(self, state_dim: int, num_actions: int,
                 hidden_shape: tuple[int] = (64, 32)):
        super().__init__()
        self.num_actions = num_actions
        layers = []
        dim_out = state_dim
        for d in hidden_shape:
            dim_in = dim_out
            dim_out = d
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim_out, num_actions))
        self.layers = nn.Sequential(*layers)

    def forward(self, state: torch.tensor) -> torch.tensor:
        return self.layers(state)

    @torch.no_grad()
    def act(self, state: torch.tensor, eps: float = 0.1) -> int:
        if np.random.random() < eps:
            return np.random.randint(0, self.num_actions)
        logits = self.layers(state)
        m = Categorical(logits=logits)
        return int(m.sample().cpu().numpy())

    def evaluate_episode(self, env: gym.Env, eps: float = 0.0,
                         max_length: int = 1000,
                         render: bool = False) -> Trajectory:
        state = env.reset()
        done = False
        trajectory = Trajectory(state_dim=env.observation_space.shape[0],
                                max_length=max_length)
        while not done and not trajectory.is_full:
            if render:
                env.render()
            action = self.act(torch.tensor(state), eps)
            next_state, reward, done, _ = env.step(action)
            trajectory.append(state, action, reward)
            state = next_state
        return trajectory


class Critic(nn.Module):
    def __init__(self, state_dim: int,
                 hidden_shape: tuple[int] = (64, 32)):
        super().__init__()
        layers = []
        dim_out = state_dim
        for d in hidden_shape:
            dim_in = dim_out
            dim_out = d
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim_out, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, state: torch.tensor) -> torch.tensor:
        return self.layers(state)

    @torch.no_grad()
    def get_value(self, state: torch.tensor) -> float:
        return float(self.layers(state).cpu().item())


class A2C:
    def __init__(self, state_dim: int, num_actions: int,
                 lr_actor: float, hidden_actor: tuple[int],
                 lr_critic: float, hidden_critic: tuple[int]):
        self.actor = Actor(state_dim, num_actions, hidden_actor)
        self.actor_optimizer = torch.optim.Adam(
            params=self.actor.parameters(),
            lr=lr_actor
        )
        self.critic = Critic(state_dim, hidden_critic)
        self.critic_optimizer = torch.optim.Adam(
            params=self.critic.parameters(),
            lr=lr_critic
        )
        self.state_dim = state_dim
        self.num_actions = num_actions

    def update(self, env: gym.Env,
               batch_size: int,
               eps: float = 0.1,
               num_episodes: int = 20,
               max_episode_length: int = 1000,
               lam: float = 0.95,
               gamma: float = 0.99):
        # compute trajectories
        trajectories = [self.actor.evaluate_episode(
            env, eps, max_episode_length, False) for _ in range(num_episodes)]
        num_transitions = sum(len(t) for t in trajectories)
        while num_transitions < batch_size:
            warn(f'Not enough transitions ({num_transitions}) for a batch')
            t = self.actor.evaluate_episode(env, eps, max_episode_length, False)
            trajectories.append(t)
            num_transitions += len(t)

        # concatenate S, A and R arrays, compute advantage
        states = np.zeros((num_transitions, self.state_dim))
        actions = np.zeros(num_transitions, dtype='int')
        rewards = np.zeros(num_transitions)
        adv = np.zeros_like(rewards)
        returns = np.zeros(len(trajectories))
        i1 = 0
        for j, t in enumerate(trajectories):
            i2 = i1 + len(t)
            states[i1:i2], actions[i1:i2], rewards[i1:i2] = t.get_SAR()
            returns[j] = np.sum(rewards[i1:i2])
            with torch.no_grad():
                values = self.critic(
                    torch.tensor(states[i1:i2], dtype=torch.float32)
                    ).cpu().ravel().numpy()
            G = t.compute_lambda_returns(values, lam, gamma)
            adv[i1:i2] = G - values

        # sample a batch
        idx = np.random.randint(0, num_transitions, size=batch_size)
        states = torch.tensor(states[idx], dtype=torch.float32)
        actions = torch.tensor(actions[idx])
        adv = torch.tensor(adv[idx], dtype=torch.float32)

        # compute actor loss
        output = self.actor(states)
        log_probs = torch.gather(
            input=F.log_softmax(output, dim=1),
            dim=1,
            index=actions.view(-1, 1)
        )
        actor_loss = torch.mean(log_probs * adv)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # compute critic loss
        values = self.critic(states)
        critic_loss = F.mse_loss(values.ravel(), adv)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return returns.mean(), returns.std()


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = A2C(env.observation_space.shape[0], env.action_space.n,
                lr_actor=LR_ACTOR, hidden_actor=(64, 32),
                lr_critic=LR_CRITIC, hidden_critic=(64, 32))
    for i in range(1000):
        mu, sigma = agent.update(env, 32, 0.1, 10)
        if i % 10 == 0:
            print(i, mu, sigma)
