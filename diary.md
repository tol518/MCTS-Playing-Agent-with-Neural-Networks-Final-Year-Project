# Development Diary - Go-Playing AI with MCTS

## Project Overview
A 9x9 Go game implementation with an AI opponent powered by Monte Carlo Tree Search (MCTS). Built in Python with both GUI and terminal interfaces.

---

## Core Components

### 1. Go Engine (`go_engine.py`)
The game engine handling all Go rules and logic.

**Key features implemented:**
- Board representation using 2D list with `Player` enum (BLACK, WHITE, EMPTY)
- Move validation including:
  - Basic legality (empty position, on board)
  - Suicide rule prevention
  - Ko rule enforcement
- Liberty counting with performance-optimized caching system
- Group detection using flood-fill algorithm
- Stone capture mechanics
- Territory scoring with komi (6.5 points for White)
- Board cloning for MCTS simulations
- Undo/redo functionality with full state snapshots

**Performance optimizations:**
- Liberty caching using group IDs to avoid recalculating liberties
- `get_adjacent_points_fast()` generator to avoid list allocations
- Efficient board cloning that preserves cache state

### 2. MCTS Agent (`mcts_agent.py`)
AI implementation using Monte Carlo Tree Search.

**MCTS phases implemented:**
1. **Selection** - UCB1 formula for balancing exploration vs exploitation
2. **Expansion** - Adding new nodes for unexplored moves
3. **Simulation** - Random playouts with heuristic guidance
4. **Backpropagation** - Updating win/visit counts up the tree

**Advanced features:**
- Tree reuse between turns (caches subtree for opponent's likely responses)
- Adaptive heuristic weights that learn during play:
  - Center preference
  - Liberty awareness
  - Connection bonus
  - Capture pressure
  - Edge penalties
- Pattern table that learns from simulation outcomes
- Move filtering to reduce search space (focuses on relevant areas)
- Early termination when position is clearly decided
- Conditioning on tree statistics (avoids moves that performed poorly)

**Evaluation heuristics:**
- Stone count with positional weighting (center stones worth more)
- Territory estimation from adjacent stone influence
- Capture count
- Fast single-pass evaluation for simulation speed

### 3. GUI (`game_gui.py`)
Modern graphical interface using Tkinter.

**Features:**
- Visual board with wood texture effect
- Stone rendering with shadows
- Last move highlight marker
- Hover preview showing legal moves
- Dark and light theme support (auto-detects system preference)
- Player indicator and capture counters
- Game modes: Human vs AI (as Black or White), Human vs Human
- Configurable AI thinking time (0.5 - 60 seconds)
- Undo/redo buttons for move history navigation
- Pass, New Game, and Quit controls

### 4. Terminal UI (`game_ui.py`)
Text-based interface for playing without GUI.
- ASCII board display
- Coordinate input (e.g., "D4", "pass")
- Same game modes as GUI

---

## Testing

### `test_go_engine.py`
Unit tests for core game logic:
- Board initialization
- Stone placement
- Capture mechanics
- Liberty counting
- Group detection
- Suicide rule
- Legal move generation
- Pass and game over
- Score calculation
- Board cloning

### `test_mcts_agent.py`
Tests for AI agent:
- Returns legal moves
- Doesn't mutate original board during search

### `test_ai_speed.py`
Performance benchmarking for AI.

---

## File Structure
```
product/
├── main.py           # Entry point (GUI/terminal selection)
├── go_engine.py      # Core game logic
├── mcts_agent.py     # AI implementation
├── game_gui.py       # Graphical interface
├── game_ui.py        # Terminal interface
├── play_gui.py       # Quick GUI launcher
├── test_go_engine.py # Engine unit tests
├── test_mcts_agent.py# AI unit tests
└── test_ai_speed.py  # Performance tests
```

---

## Recent Changes

### Undo/Redo System
Added full undo/redo support:
- `_undo_stack` and `_redo_stack` store complete board snapshots
- `undo()` restores previous state and pushes current to redo stack
- `redo()` restores undone state
- New moves clear the redo stack
- GUI buttons: Undo, Redo

### GUI Button Layout
- Reorganized button bar to fit 5 buttons
- Auto-sizing buttons using padding instead of fixed width
- Shortened labels for better fit

---

## How to Run
```bash
cd product
python main.py
```

Select:
1. Graphical Interface (recommended)
2. Terminal Interface

---

## Technical Notes

### Why Liberty Caching?
Go requires frequent liberty counting (checking if groups are captured). Naive counting is O(n) per query. The caching system:
- Assigns unique group IDs to connected stones
- Maintains `group_liberties[gid]` = set of liberty positions
- Updates incrementally when stones are placed/removed
- Reduces repeated calculations during MCTS simulations

### Why Tree Reuse?
MCTS explores many positions. When the opponent plays a move we already explored, we can reuse that subtree instead of starting from scratch, saving thousands of simulations.

### Why Move Filtering?
On a 9x9 board there are 81 possible moves. Many are irrelevant (far from existing stones). Filtering to ~20-30 relevant moves makes simulations 3-4x faster without losing quality.

---

## Future Improvements
- [ ] Neural network evaluation (AlphaGo-style)
- [ ] Opening book for common patterns
- [ ] Game record save/load (SGF format)
- [ ] Online multiplayer
- [ ] Larger board support (13x13, 19x19)
