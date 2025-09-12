import asyncio
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poke_env import RandomPlayer, Player
from poke_env.battle.abstract_battle import AbstractBattle
from DQN import DQN
from RLPlayer import RLPlayer

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