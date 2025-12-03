# Single-Step Move Refactoring

## Overview

The four-player chess environment has been completely refactored to use a **single-step move selection process** where all legal moves are enumerated at once. Instead of having the model output free-form chess notation, it simply picks a number from a list. This dramatically reduces invalid moves and makes the task much easier for the model.

## How It Works

### Single Step: Move Selection

The model is shown:
1. A numbered list of ALL legal moves
2. Each move shows the piece, source position, and destination
3. The model simply outputs a number

Example prompt:
```
You are 🔴 Red.
Move 0

Scores:
  🔴 Red: 0 points (Active)
  🔵 Blue: 0 points (Active)
  🟡 Yellow: 0 points (Active)
  🟢 Green: 0 points (Active)

YOUR LEGAL MOVES:
  1. P (Pawn) from (12, 3) to (11, 3)
  2. P (Pawn) from (12, 3) to (10, 3)
  3. P (Pawn) from (12, 4) to (11, 4)
  4. P (Pawn) from (12, 4) to (10, 4)
  5. P (Pawn) from (12, 5) to (11, 5)
  6. P (Pawn) from (12, 5) to (10, 5)
  7. N (Knight) from (13, 4) to (11, 3)
  8. N (Knight) from (13, 4) to (11, 5)
  ...

Choose your move.
Output ONLY a number from the list above.
Example: 1
```

Model output: `5` (moves pawn from (12, 5) forward one square)

## Key Benefits

### 1. **No Invalid Moves**
- The model can only choose from moves that are actually legal
- Impossible to select non-existent squares or illegal moves
- No need for complex validation or retry loops

### 2. **Much Simpler Task**
- No need to understand chess notation
- No need to remember which columns/rows are valid
- No need to understand piece movement rules
- Just pick from a list!

### 3. **Single Interaction Per Turn**
- One prompt, one response, done
- No multi-step state management
- Faster execution (half as many model calls)

## Technical Implementation

### Key Functions

**`get_player_pieces(state, player_id, valid_mask)`**
- Scans the board for all pieces owned by the player
- Returns list of (row, col, piece_name) tuples

**`get_legal_moves_for_piece(state, row, col, player_id, valid_mask)`**
- Uses JAX environment's `get_pseudo_legal_moves` function
- Filters to only fully legal moves (doesn't leave king in check)
- Returns list of (row, col) destination tuples

**`get_all_legal_moves(state, player_id, valid_mask)`**
- Combines the above two functions
- Returns list of (source_row, source_col, dest_row, dest_col, piece_name) tuples
- This is the complete set of legal moves for the player

**`parse_move_selection(text, num_moves)`**
- Extracts a number from model output
- Validates it's in range 1 to num_moves
- Returns 0-based index or None

### Environment State

The `FourPlayerChessEnv` maintains:
- `retry_count`: int (track retries for invalid selections)
- `has_had_initial_turn`: bool (track if first turn)

No more state management for multi-step selection!

### Error Handling

- If model outputs invalid selection: retry with error message
- If too many retries: end episode with penalty

## Migration

The refactored code is in `env_refactored.py`. To use it:

1. The training script (`train.py`) now imports from `env_refactored`
2. All the same config options work
3. The single-step interaction is completely transparent to the training loop

## Performance Expectations

With this refactoring, we should see:
- **~0% invalid moves** (vs ~90% with free-form notation)
- **Longer games** (actual gameplay instead of retry loops)
- **All 4 players participating** (games don't end on first player's turn)
- **Faster learning** (model can focus on strategy instead of notation)
- **2x faster execution** (one model call per turn instead of two)

## Example Game Flow

```
Turn 1 - Red:
  Prompt: Shows 30+ legal moves (pawns, knights, etc.)
  Model: "5"
  ✓ Red pawn moves (12,5) → (10,5)

Turn 2 - Blue:
  Prompt: Shows 30+ legal moves
  Model: "12"
  ✓ Blue knight moves (4,13) → (5,11)

... game continues naturally with all 4 players
```

## Future Improvements

Possible enhancements:
1. Show board state in the prompts
2. Add piece values/strategic hints
3. Multi-turn trajectory for model to see game history
4. Temperature tuning for exploration vs exploitation
