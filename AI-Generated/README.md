# Dynamic Connect-3 Game with Minimax AI

This implementation provides a complete game engine for Dynamic Connect-3 with an AI agent using minimax search with alpha-beta pruning.

## Game Rules

**Dynamic Connect-3** is played on a 5×4 grid with the following rules:

1. **Initial Setup**: 
   - Leftmost column: White-Black-White-Black (top to bottom)
   - Rightmost column: Black-White-Black-White (top to bottom)
   - Middle columns start empty

2. **Gameplay**:
   - White moves first
   - Players alternate turns moving one piece of their color
   - Pieces move one square horizontally or vertically to an empty square
   - Goal: Get three pieces in a row (horizontal, vertical, or diagonal)

3. **End Conditions**:
   - First player to get 3 in a row wins
   - Threefold repetition results in a draw
   - No available moves results in a draw

## Implementation Features

### Core Game Engine (`dynamic_connect3.py`)

- **GameState Class**: Manages board state, move validation, and game logic
- **Move Validation**: Ensures moves are legal (one square, to empty space, own piece)
- **Win Detection**: Checks for 3-in-a-row in all directions
- **Threefold Repetition**: Tracks position history to detect draws
- **Heuristic Evaluation**: H(n) = runs_of_2_white - runs_of_2_black

### AI Agent

- **Minimax Algorithm**: Depth-limited search with alpha-beta pruning
- **Configurable Depth**: Adjustable search depth for performance tuning
- **Node Counting**: Tracks nodes evaluated for performance analysis

### Heuristic Function

The evaluation function counts "runs of 2" for each player:
- A run of 2 is exactly 2 consecutive pieces of the same color
- Heuristic value = (White runs of 2) - (Black runs of 2)
- Terminal states: +1000 (White wins), -1000 (Black wins), 0 (draw)

## Files

1. **`dynamic_connect3.py`**: Core game implementation
2. **`interactive_game.py`**: Interactive gameplay modes
3. **`test_game.py`**: Comprehensive test suite
4. **`README.md`**: This documentation

## Usage

### Running the Interactive Game

```bash
python interactive_game.py
```

Options:
1. Human vs AI (you play as White)
2. AI vs AI (different depth levels)
3. Test heuristic function
4. Quit

### Running Tests

```bash
python test_game.py
```

This runs:
- Unit tests for game logic
- Performance benchmarks
- Game progression demo

### Using the Game Engine Directly

```python
from dynamic_connect3 import GameState, MinimaxAgent

# Create a new game
game = GameState()

# Create AI agent
ai = MinimaxAgent(max_depth=6)

# Get AI's best move
best_move = ai.get_best_move(game)

# Make the move
if best_move:
    from_pos, to_pos = best_move
    game.make_move(from_pos, to_pos)
```

## Move Input Formats

When playing interactively, you can use either format:

1. **Letter-Number**: `a1 b1` (column a-e, row 1-4)
2. **Coordinates**: `0,0 1,0` (column 0-4, row 0-3)

Example moves:
- `a1 a2` - Move piece from column a, row 1 to column a, row 2
- `0,0 0,1` - Same move using coordinates

## Algorithm Details

### Minimax with Alpha-Beta Pruning

The AI uses minimax search with the following optimizations:

1. **Alpha-Beta Pruning**: Eliminates branches that won't affect the final decision
2. **Depth Limiting**: Configurable maximum search depth
3. **Move Ordering**: Natural move generation order
4. **Transposition Detection**: Game state hashing for repetition detection

### Performance Characteristics

Typical performance on modern hardware:
- Depth 3: ~100-500 nodes evaluated, <0.1s
- Depth 4: ~500-2000 nodes evaluated, <0.5s  
- Depth 5: ~2000-8000 nodes evaluated, 1-3s
- Depth 6: ~8000-30000 nodes evaluated, 3-10s

## Example Game Flow

```
Initial position:
  0 1 2 3 4
0 W . . B
1 B . . W  
2 W . . B
3 B . . W
Current player: White

White's move: (0,0) -> (1,0)
  0 1 2 3 4
0 . W . B
1 B . . W
2 W . . B  
3 B . . W
Current player: Black
```

## Extending the Implementation

The modular design allows for easy extensions:

1. **Enhanced Heuristics**: Modify `heuristic_evaluation()` method
2. **Different Search Algorithms**: Replace minimax with other algorithms
3. **UI Improvements**: Add graphical interface
4. **Opening Books**: Add pre-computed opening moves
5. **Endgame Tables**: Add perfect play for small piece counts

## Testing

The test suite covers:
- Game rule enforcement
- Move validation
- Win condition detection
- Heuristic function correctness
- AI decision making
- Performance benchmarking

Run tests to verify the implementation works correctly on your system.
