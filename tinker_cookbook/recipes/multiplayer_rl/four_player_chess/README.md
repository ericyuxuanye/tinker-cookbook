# Four-Player Chess RL Training

This recipe implements self-play reinforcement learning for four-player chess using the [4-player-chess-jax](https://github.com/ericyuxuanye/4-player-chess-jax) environment.

## Overview

Four-player chess is played on a 14×14 cross-shaped board with four players (Red, Blue, Yellow, Green) starting from each side. The game follows standard chess rules adapted for four players, with scoring based on captures and eliminations.

## Environment Details

- **Players**: 4 players (Red, Blue, Yellow, Green)
- **Board**: 14×14 cross-shaped board with 160 valid squares
- **Action Space**: Text-based moves in format `[(src_row, src_col) -> (dest_row, dest_col)]`
- **Rewards**:
  - Capture points (Pawn: 1, Knight/Bishop: 3, Rook: 5, Queen: 9)
  - Checkmate/elimination bonuses: 20 points
  - Illegal move penalty: -5 points

## Architecture

The implementation follows the multiplayer RL pattern from `text_arena`:

1. **FourPlayerCoordinator**: Manages the shared JAX game state and synchronizes player turns
2. **FourPlayerChessEnv**: Wraps the environment for each player, converting between:
   - JAX board states → Text observations
   - LLM text actions → JAX action indices
3. **FourPlayerChessEnvGroupBuilder**: Creates groups of 4 environments (one per player) sharing the same game
4. **Self-play training**: All 4 players train simultaneously on the same policy

## Usage

### Basic Training Run

```bash
python -m tinker_cookbook.recipes.multiplayer_rl.four_player_chess.train \
  model_name=meta-llama/Llama-3.2-1B \
  log_path=/tmp/tinker-examples/chess-selfplay
```

### Configuration Options

```bash
python -m tinker_cookbook.recipes.multiplayer_rl.four_player_chess.train \
  model_name=Qwen/Qwen3-4B-Instruct-2507 \
  renderer_name=qwen3 \
  batch_size=512 \
  num_train_datapoints=131072 \
  learning_rate=3e-5 \
  max_tokens=128 \
  eval_every=5 \
  save_every=20 \
  log_path=/tmp/my-chess-run
```

### Key Parameters

- `model_name`: Base model to fine-tune (default: Qwen/Qwen3-4B-Instruct-2507)
- `batch_size`: Training batch size (must be divisible by 4)
- `num_train_datapoints`: Total training datapoints (must be divisible by 4)
- `learning_rate`: Learning rate (default: 3e-5)
- `max_tokens`: Max tokens per move (default: 128)

## Implementation Details

### Text Observations

The LLM receives observations like:

```
You are 🔴 Red.
Move 5

Scores:
  🔴 Red: 3 points (Active)
  🔵 Blue: 1 points (Active)
  🟡 Yellow: 0 points (Active)
  🟢 Green: 5 points (Active)

Board (14x14 cross-shaped):
     0  1  2  3  4  5  6  7  8  9 10 11 12 13
 0:                rR rN rB rQ rK rB rN rR
 1:                rP rP rP rP rP rP rP rP
 2:                 .  .  .  .  .  .  .  .
...

Your turn! Make a move in the format: [(source_row, source_col) -> (dest_row, dest_col)]
For pawn promotion, add =Q (Queen), =R (Rook), =B (Bishop), or =N (Knight)
Example: [(12, 6) -> (10, 6)]
```

### Move Format

LLM outputs moves as:
- Basic move: `[(12, 6) -> (10, 6)]`
- Pawn promotion: `[(3, 7) -> (0, 7)=Q]`

The wrapper parses these and converts them to action indices for the JAX environment.

### Coordination

The `FourPlayerCoordinator` ensures proper turn-taking:
- Players wait asynchronously for their turn using `asyncio.Condition`
- In self-play mode, all 4 players share the same coordinator
- Invalid moves trigger retry logic:
  - Players get up to 3 retries to make a valid move
  - After each invalid move, the LLM receives feedback and the current board state
  - Exceeding the retry limit ends the game with a -5 penalty

## Reward Structure

Players receive rewards for:
- **Captures**: Points based on piece value (Pawn=1, Knight/Bishop=3, Rook=5, Queen=9)
- **Eliminations**: 20 points for checkmating/stalemating an opponent
- **Invalid moves**:
  - First 3 invalid moves: 0 reward, but player can retry
  - Exceeding retry limit: -5 penalty and game ends

The total reward is the sum of per-timestep rewards from the JAX environment.

## Logging and Monitoring

The environment provides comprehensive logging to track training progress:

### Per-Step Metrics

Each move logs:
- `move`: The move text (e.g., `[(12, 6) -> (10, 6)]`)
- `player`: Player color (Red, Blue, Yellow, Green)
- `move_number`: Current move number in the game
- `reward`: Immediate reward received
- `score`: Current cumulative score
- `retry_count`: Number of invalid move retries (if any)

### Trajectory-Level Metrics

At the end of each game:
- `game_length`: Total number of moves
- `final_score`: Final score for this player
- `won_game`: 1 if this player won, 0 otherwise
- `survived`: 1 if player still active, 0 if eliminated
- `winner_player`: Name of the winning player

### Game Summaries

After each game, a detailed summary is logged showing:
- Final scores for all players
- Winner and their score
- Move-by-move history (first 20 moves)
- Player eliminations and status

Example log output:
```
============================================================
GAME SUMMARY
============================================================
Total moves: 42
Winner: 🔴 Red (score: 15)

Final Scores:
  🔴 Red: 15 points (Active)
  🔵 Blue: 8 points (Eliminated)
  🟡 Yellow: 12 points (Active)
  🟢 Green: 5 points (Eliminated)

Move History:
  Move 1: 🔴 Red plays [(12, 6) -> (10, 6)] (reward: 0.0, score: 0)
  Move 2: 🔵 Blue plays [(1, 7) -> (3, 7)] (reward: 0.0, score: 0)
  ...
============================================================
```

### Metrics in Training Logs

All metrics are automatically aggregated and displayed during training via `ml_log` and saved to `metrics.jsonl`. You can also enable Weights & Biases logging with `wandb_project` parameter.

## Training Tips

1. **Batch size**: Use multiples of 4 (one env per player). Recommended: 512 or higher.
2. **Learning rate**: Start with 3e-5. Four-player chess requires careful exploration.
3. **Max tokens**: 128 tokens is usually sufficient for move generation.
4. **Eval frequency**: Evaluate every 5-10 batches to monitor progress.
5. **Monitor logs**: Check game summaries to see if the model is learning valid moves and strategy.

## Files

- `env.py`: Environment wrapper, coordinator, dataset builders
- `train.py`: Training script with CLI configuration
- `README.md`: This file

## Dependencies

Requires the `four-player-chess-jax` library:

```bash
pip install git+https://github.com/ericyuxuanye/4-player-chess-jax
```

Also requires JAX, which is installed as a dependency.

## Related Examples

- `tinker_cookbook/recipes/multiplayer_rl/text_arena`: Two-player TicTacToe (simpler example)
- `tinker_cookbook/recipes/multiplayer_rl/twenty_questions`: Cooperative multi-agent
- `tinker_cookbook/recipes/rl_basic`: Single-agent RL basics
