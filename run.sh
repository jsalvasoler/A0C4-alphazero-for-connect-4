#!/usr/bin/env bash
set -e

# ── A0C4 — AlphaZero for Connect 4 ──────────────────────────────────
# Quick-reference commands. Run any of these from the project root.
# Make sure to run `uv sync` first to install dependencies.

# ── Play ─────────────────────────────────────────────────────────────
# Play against the random agent (moves randomly)
uv run python -m src.cli play --agent random

# Play against the optimal agent (uses online solver, never loses)
uv run python -m src.cli play --agent optimal

# Play against the trained AlphaZero agent
uv run python -m src.cli play --agent alpha

# ── Train ────────────────────────────────────────────────────────────
# Train the AlphaZero neural network (uses config/cfg.yaml for parameters)
uv run python -m src.cli train --config config/cfg_test.yaml

# ── Test / Evaluate ──────────────────────────────────────────────────
# Pit AlphaZero against the random agent (100 games)
uv run python -m src.cli test --agent1 alpha --agent2 random --games 100

# Pit AlphaZero against the optimal agent (accuracy computed automatically)
uv run python -m src.cli test --agent1 alpha --agent2 optimal --games 50

# Pit random vs random (sanity check — should be ~50/50)
uv run python -m src.cli test --agent1 random --agent2 random --games 200

# ── Default: play against optimal ────────────────────────────────────
uv run python -m src.cli play --agent optimal
