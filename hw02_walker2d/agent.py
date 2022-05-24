import numpy as np
import torch
from torch.distributions import Normal


class Agent:
    def __init__(self):
        self.model, log_sigma = torch.load(__file__[:-8] + "/agent.pkl")
        self.sigma = torch.nn.Parameter(torch.exp(log_sigma.detach()))

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float()
            mu = self.model(state)
            dist = Normal(mu, self.sigma)
            return torch.tanh(dist.sample().squeeze()).cpu().numpy()

    def reset(self):
        pass
