from gym import make
import torch
from torch import nn
from torch.distributions.categorical import Categorical
import numpy as np

LR_ACTOR = 2e-4
LR_CRITIC = 1e-4
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Trajectory:
    def __init__(self, state_dim: int, max_length: int = 10000):
        self.states = np.zeros(max_length, state_dim)
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

    def get_states(self) -> np.ndarray:
        return self.states[:self.pos]

    def compute_lambda_returns(self, values: np.ndarray,
                               lam: float = 0.95,
                               gamma: float = 0.99) -> np.ndarray:
        G = self.rewards[:self.pos]
        i = self.pos
        while i > 0:
            G[i - 1] += gamma * ((1 - lam) * values[i] + lam * G[i])
            i -= 1
        return G


def sample_trajectories(trajectories: list[Trajectory]):
    raise NotImplementedError


class Actor(nn.Module):
    def __init__(self, state_dim: int, num_actions: int,
                 hidden_shape: tuple[int] = (64, 32)):
        super().__init__()
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
    def act(self, state: torch.tensor) -> np.ndarray:
        logits = self.layers(state)
        m = Categorical(logits=logits)
        return m.sample().cpu().numpy()


class Critic(nn.Module):
    def __init__(self, state_dim: int, num_actions: int,
                 hidden_shape: tuple[int] = (64, 32)):
        super().__init__()
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
        self.critic = Critic(state_dim, num_actions, hidden_critic)
        self.critic_optimizer = torch.optim.Adam(
            pararams=self.critic.parameters(),
            lr=lr_critic
        )


if __name__ == '__main__':
    env = make('LunarLander-v2')
    state = env.reset()
    actor = Actor(state_dim=env.observation_space.shape[0],
                  num_actions=env.action_space.n)
    state = env.reset()
    action = actor.act(torch.tensor(state))
    rv = env.step(action)
    print(rv)
