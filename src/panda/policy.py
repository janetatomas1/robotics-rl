
from stable_baselines3.td3.policies import TD3Policy
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, n_joints=3):
        super().__init__()

        self.n = n_joints

        self.l1 = nn.Linear(in_features=2 * self.n, out_features=2 * self.n)
        self.l2 = nn.Linear(in_features=2 * self.n, out_features=self.n)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = F.softmax(x)
        return x


class Critic(nn.Module):
    def __init__(self, n_joints=3):
        super().__init__()

        self.n = n_joints

        self.l1 = nn.Linear(in_features=2 * self.n, out_features=2 * self.n)
        self.l2 = nn.Linear(in_features=2 * self.n, out_features=self.n)
        self.l3 = nn.Linear(in_features=self.n, out_features=1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = F.softmax(x)
        return x


# class CustomPolicy(TD3Policy):
#     def __init__(self, n_joints):
#         self.


