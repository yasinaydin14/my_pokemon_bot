import asyncio
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poke_env import RandomPlayer, Player
from poke_env.battle.abstract_battle import AbstractBattle
from train import DQN
from train import RLPlayer


# Load model
model = DQN(4, 4)
model.load_state_dict(torch.load("poke_dqn.pt"))

# Create RLPlayer
rl_player = RLPlayer(model=model, epsilon=0.0, battle_format="gen9randombattle")

# Battle a random player
async def play():
    rand_player = RandomPlayer(battle_format="gen9randombattle")
    await rl_player.battle_against(rand_player, n_battles=10)
    print(f"Wins: {rl_player.n_won_battles} / {rl_player.n_finished_battles}")

asyncio.get_event_loop().run_until_complete(play())