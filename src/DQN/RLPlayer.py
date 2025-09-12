import asyncio
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poke_env import RandomPlayer, Player
from poke_env.battle.abstract_battle import AbstractBattle
# ---- RL‐Player ----
class RLPlayer(Player):
    def __init__(self, model, epsilon=0.1, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.epsilon = epsilon
        self.replay_buffer = []
        self.max_buffer = 5000

    def embed_battle(self, battle: AbstractBattle):
        """
        Construct a simple state: e.g. our active Pokémon HP fraction,
        opponent active Pokémon HP fraction, plus maybe statuses.
        """
        my_pokemon = battle.active_pokemon
        opp_pokemon = battle.opponent_active_pokemon

        my_hp_frac = my_pokemon.current_hp_fraction if my_pokemon is not None else 0.0
        opp_hp_frac = opp_pokemon.current_hp_fraction if opp_pokemon is not None else 0.0

        # statuses as dummy 0/1
        my_status = 0.0
        if my_pokemon is not None and my_pokemon.status is not None:
            my_status = 1.0
        opp_status = 0.0
        if opp_pokemon is not None and opp_pokemon.status is not None:
            opp_status = 1.0

        # Example state vector
        return np.array([my_hp_frac, opp_hp_frac, my_status, opp_status], dtype=np.float32)

    def choose_move(self, battle: AbstractBattle):
        state = self.embed_battle(battle)
        legal_moves = battle.available_moves

        # If no legal moves, pick random switch or move
        if not legal_moves:
            return self.choose_random_move(battle)

        # Exploration
        if random.random() < self.epsilon:
            move = random.choice(legal_moves)
            return self.create_order(move)

        # Exploitation: model predicts Q‐values
        state_tensor = torch.tensor(state).unsqueeze(0)  # shape [1, state_dim]
        with torch.no_grad():
            q_values = self.model(state_tensor)  # shape [1, output_dim]

        # Need to map moves to output indices
        # Here we assume output_dim == max number of moves; pad if fewer moves
        # Simpler: pick the move with highest base power among legal moves,
        # but weighted by Q values if full action set known.
        # For demo: we pick argmax over legal moves via random index mapping:
        # Create list of (idx, move) pairs
        # For simplicity, assume output_dim >= len(legal_moves)
        q_vals_moves = []
        for i, move in enumerate(legal_moves):
            q_vals_moves.append((q_values[0, i].item(), move))
        _, best_move = max(q_vals_moves, key=lambda x: x[0])
        return self.create_order(best_move)

    def reward_computing(self, battle: AbstractBattle) -> float:
        # Called after the battle ends; you can compute final reward
        # You may also provide intermediate rewards via battle attributes
        # Simple reward: +1 if won, -1 if lost
        return 1.0 if battle.won else -1.0