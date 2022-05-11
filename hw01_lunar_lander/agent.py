import numpy as np
import torch


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")

    def act(self, state):
        state = torch.tensor(state)
        self.model.eval()
        with torch.no_grad():
            output = self.model(state).detach().numpy()
        return np.argmax(output)
