import asyncio
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poke_env import RandomPlayer, Player
from poke_env.battle.abstract_battle import AbstractBattle

# ---- Neural network ----
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)