# Final Year Project

This repository has been created to store your final year project.

You may edit it as you like, but please do not remove the default topics or the project members list. These need to stay as currently defined in order for your supervisor to be able to find your project.

---

## Project Overview
A 9x9 Go game implementation with an AI opponent powered by Monte Carlo Tree Search (MCTS). Built in Python with both GUI and terminal interfaces.

## How to Run
From the `PROJECT` directory:

```bash
pip install numpy
cd product
python main.py  # Choose GUI or terminal
```

## Core Components

### Go Engine (`product/go_engine.py`)
The game engine handling Go rules and gameplay logic.

- **Key features**:
  - Board representation using a 2D list with a `Player` enum (BLACK, WHITE, EMPTY)
  - Move validation (on-board, empty position, suicide prevention, ko enforcement)
  - Group detection (flood-fill)
  - Liberty counting with caching
  - Capture mechanics
  - Territory scoring with komi (6.5 for White)
  - Board cloning for MCTS simulations
  - Undo/redo via full state snapshots
- **Performance**:
  - Liberty caching using group IDs to reduce repeated recalculation
  - `get_adjacent_points_fast()` generator to reduce allocations
  - Efficient board cloning that preserves cache state

### MCTS Agent (`product/mcts_agent.py`)
AI implementation using Monte Carlo Tree Search.

- **Phases implemented**:
  1. Selection (UCB1)
  2. Expansion
  3. Simulation (random playouts with heuristics)
  4. Backpropagation
- **Advanced features**:
  - Tree reuse between turns
  - Adaptive heuristic weights (center preference, liberty awareness, connection bonus, capture pressure, edge penalties)
  - Pattern table learning from simulation outcomes
  - Move filtering to reduce the branching factor
  - Early termination when the position is clearly decided
  - Conditioning on tree statistics (avoid historically weak moves)
- **Evaluation heuristics**:
  - Stone count with positional weighting
  - Territory estimation via adjacent influence
  - Capture count
  - Fast single-pass evaluation for simulation speed

### GUI (`product/game_gui.py`)
Tkinter graphical interface.

- **Features**:
  - Board rendering (wood texture effect), stones with shadows
  - Last move highlight and hover preview of legal moves
  - Dark/light theme support
  - Capture counters + player indicator
  - Human vs AI / Human vs Human
  - Configurable AI thinking time (0.5–60s)
  - Undo/redo, pass, new game, quit

### Terminal UI (`product/game_ui.py`)
Text-based interface.

- ASCII board display
- Coordinate input (e.g. `D4`, `pass`)
- Same game modes as GUI

## Testing

- **`product/test_go_engine.py`**: engine logic (captures, liberties, legality, ko/suicide, scoring, cloning, etc.)
- **`product/test_mcts_agent.py`**: AI agent sanity checks (legal move selection, board not mutated during search)
- **`product/test_ai_speed.py`**: performance benchmarking

## File Structure

```text
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

## Technical Notes

### Why Liberty Caching?
Go requires frequent liberty counting. Naive counting can be O(n) per query; caching maintains group IDs and tracked liberties to reduce repeated recalculation during gameplay and MCTS simulations.

### Why Tree Reuse?
When the opponent plays a move already explored by MCTS, the corresponding subtree can be reused instead of starting from scratch, saving many simulations.

### Why Move Filtering?
On a 9x9 board there are 81 possible moves; many are irrelevant far from existing stones. Filtering to a smaller relevant subset increases simulation throughput without significantly degrading play quality.

## Future Improvements
- [ ] Neural network evaluation (AlphaGo-style)
- [ ] Game record save/load (SGF format)