import numpy as np
import torch
from torch.distributions import Normal


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float()
            output = self.model(state)
            action_dim = output.size(-1) // 2
            mu = output[..., :action_dim]
            sigma = torch.exp(output[..., action_dim:])
            dist = Normal(mu, sigma)
            return torch.tanh(dist.sample().squeeze()).cpu().numpy()

    def reset(self):
        pass
