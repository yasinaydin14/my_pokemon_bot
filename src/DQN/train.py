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

async def train():
    # Setup
    state_dim = 4
    max_moves = 4  # maximum number of legal moves considered for Q outputs

    model = DQN(input_dim=state_dim, output_dim=max_moves)
    target_model = DQN(input_dim=state_dim, output_dim=max_moves)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    gamma = 0.99
    batch_size = 32

    # Players
    rl_player = RLPlayer(
        model=model,
        epsilon=0.2,
        battle_format="gen9randombattle"
    )

    rand_player = RandomPlayer(battle_format="gen9randombattle")

    n_episodes = 2000
    target_update_freq = 20

    for ep in range(1, n_episodes + 1):
        # Run one battle
        battle = await rl_player.battle_against(rand_player, n_battles=1)
        battle = list(rl_player.battles.values())[-1]
        # Extract transitions (this is illustrative; you might need to modify to collect during battle)
        # poke-env does not directly provide step-by-step transitions, so you may need to hook into choose_move, etc.
        # For demo: use only final reward
        state = np.zeros(state_dim, dtype=np.float32)   # placeholder
        action = 0  # placeholder
        reward = rl_player.reward_computing(battle)
        next_state = np.zeros(state_dim, dtype=np.float32)
        done = True

        rl_player.replay_buffer.append((state, action, reward, next_state, done))
        if len(rl_player.replay_buffer) > rl_player.max_buffer:
            rl_player.replay_buffer.pop(0)

        # Training step
        if len(rl_player.replay_buffer) >= batch_size:
            batch = random.sample(rl_player.replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.tensor(np.vstack(states), dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(np.vstack(next_states), dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Q(s,a)
            q_pred = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            # target Q
            with torch.no_grad():
                max_next_q = target_model(next_states).max(1)[0]
                q_target = rewards + gamma * max_next_q * (1 - dones)

            loss = loss_fn(q_pred, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network
        if ep % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())
            print(f"[Episode {ep}] target network updated.")

        # Maybe decay epsilon
        rl_player.epsilon = max(0.01, rl_player.epsilon * 0.995)

        print(f"Episode {ep} completed. Reward: {reward:.3f}")

    # After training
    print("Training done.")
    print(
        f"RLPlayer won {rl_player.n_won_battles} / {rl_player.n_finished_battles} battles "
    )
    torch.save(model.state_dict(), "poke_dqn.pt")
    print("Model saved as poke_dqn.pt")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(train())