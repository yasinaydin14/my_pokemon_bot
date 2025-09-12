# Pokémon Bot Project

This project allows you to **train and use a DQN-based Pokémon bot** for battles using the [Pokémon Showdown](https://github.com/smogon/pokemon-showdown.git) platform.

---

## Features
- Train your own reinforcement learning Pokémon bot
- Connect and battle via Pokémon Showdown
- Modular structure for easy extension and experimentation

---

## Setup

### 1. Clone Pokémon Showdown
```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security

**The server will run at http://localhost:8000. Keep this terminal open while training or testing your bot.**
