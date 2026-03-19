# Final Year Project

This repository has been created to store your final year project.

You may edit it as you like, but please do not remove the default topics or the project members list. These need to stay as currently defined in order for your supervisor to be able to find your project.

---

## Project Overview

A 9×9 Go game implementation with a progressively stronger AI opponent. The system started with a hand-crafted Monte Carlo Tree Search (MCTS) agent and has evolved into an **AlphaGo-style Neural Network-guided MCTS** agent trained via supervised learning on professional games and refined through a Self-Play Reinforcement Learning pipeline.

Built in Python with both GUI (Tkinter) and terminal interfaces.

---

## How to Run

From the `PROJECT` directory:

```bash
pip install numpy torch
cd product
python main.py  # prompts to choose GUI or terminal
```

> **GPU**: The NN-MCTS agent automatically detects and uses CUDA if available. CPU-only inference is supported but slower.

A pre-trained checkpoint is loaded automatically from `product/nn/checkpoints/` (RL-trained preferred over supervised-only).

---

## Core Components

### Go Engine (`product/go_engine.py`)
The rules engine underpinning all gameplay and AI search.

- **Board representation**: 2D list with a `Player` enum (BLACK, WHITE, EMPTY)
- **Move validation**: on-board checks, empty position, suicide prevention, ko enforcement
- **Group detection**: flood-fill for connected stone groups
- **Liberty counting**: cached per group ID to reduce repeated recalculation
- **Capture mechanics** and **territory scoring** with komi (6.5 for White)
- **Undo/redo**: full state snapshots via a history stack
- **Performance**: `get_adjacent_points_fast()` generator, efficient board cloning that preserves cache state, `make_move_fast()` / `pass_turn_fast()` for high-throughput MCTS use

---

### MCTS Agent (`product/mcts_agent.py`)
Original hand-crafted AI using Monte Carlo Tree Search (UCB1-based).

- **Phases**: Selection (UCB1) → Expansion → Simulation (random playouts + heuristics) → Backpropagation
- **Advanced features**:
  - Tree reuse between turns
  - Adaptive heuristic weights (center preference, liberty awareness, connection bonus, capture pressure, edge penalties)
  - Pattern table learning from simulation outcomes
  - Move filtering to reduce branching factor
  - Early termination when position is clearly decided
- **Evaluation heuristics**: stone count with positional weighting, territory estimation via adjacent influence, capture count

---

### Neural Network — `AlphaGoZeroNetwork` (`product/nn/model.py`)
An **AlphaGo Zero-style CNN** implemented in PyTorch.

| Component | Details |
|---|---|
| **Input** | 16-channel 9×9 tensor (board features encoded by `GoBoardEncoder`) |
| **Shared trunk** | Initial 3×3 Conv + BatchNorm + ReLU, then **5 Residual blocks** (128 channels) |
| **Policy head** | 1×1 Conv → Flatten → Linear → **82 raw logits** (81 board squares + 1 pass) |
| **Value head** | 1×1 Conv → Flatten → Linear(128) → Linear(1) → **Tanh** → scalar in [−1, 1] |

The value output is from Black's perspective; the agent converts it to the current player's win probability at inference time.

---

### NN-MCTS Agent (`product/nn_mcts_agent.py`)
Replaces random rollouts with neural network evaluation.

- **Selection**: PUCT formula — `Q(s,a) + c_puct × P(s,a) × √N_parent / (1 + N_child)` — using policy priors from the network
- **Evaluation**: value head replaces Monte Carlo playouts, giving a position assessment in O(1) network forward passes
- **Score-aware pass logic**: blocks premature passing when behind (margin < 5 pts); forces pass when clearly ahead (margin ≥ 10 pts)
- **Inference cache**: board-state hash → (policy probs, value); evicted at 50,000 entries to cap RAM usage
- **Tree reuse**: subtree from the previous move is retained and re-rooted
- **Multiprocess safety**: lazy torch imports so worker processes stay lightweight; enum pickling issues resolved

---

### Self-Play RL Pipeline (`product/nn/self_play.py`)
AlphaZero-style reinforcement learning loop.

- The NN-MCTS agent plays games against itself, generating `(state, policy_target, value_target)` training examples
- The network is updated from self-play data, then the improved agent replaces the previous one
- Provides automated agent improvement without human-authored heuristics

---

### Supervised Pre-training (`product/nn/train.py` + `product/nn/parse_sgf.py`)
- **SGF parser** (`parse_sgf.py`): converts professional game records into board states and move labels
- **Pre-training**: CNN trained on ~1.8 million professional game samples using the **Adam optimizer** before RL begins, giving the agent a strong prior
- **Dataset / Encoder** (`dataset.py`, `encoder.py`): encode board state into 16 feature planes (stone positions, liberties, turn, ko, etc.)

---

### GUI (`product/game_gui.py`)
Tkinter graphical interface.

- **Agents selectable**: Human, MCTS, **NN-MCTS** (new)
- Board rendering with wood-texture effect, stones with shadows
- Last-move highlight, legal-move hover preview
- Dark/light theme, capture counters, player indicator
- Configurable AI thinking time (0.5–60 s)
- Undo/redo, pass, new game, quit controls
- AI-vs-AI mode for automated evaluation sessions

---

### Terminal UI (`product/game_ui.py`)
Text-based interface.

- ASCII board display with coordinate input (e.g. `D4`, `pass`)
- Same game modes as the GUI

---

### Tournament / Profiling (`product/tournament.py`, `product/profiler_utils.py`)
- `tournament.py`: runs automated head-to-head matches between any two agent types; used to verify NN-MCTS beats the baseline MCTS
- `profiler_utils.py`: timing and throughput utilities for benchmarking MCTS simulation speed

---

## Testing

| File | Purpose |
|---|---|
| `product/test_go_engine.py` | Engine logic: captures, liberties, legality, ko/suicide, scoring, cloning |
| `product/test_mcts_agent.py` | AI sanity checks: legal move selection, board not mutated during search |
| `product/test_ai_speed.py` | Performance benchmarking of MCTS throughput |
| `product/nn/test_encoder.py` | Board encoder correctness (feature planes) |
| `product/nn/test_dataset.py` | SGF-parsed dataset integrity checks |
| `product/nn/cnn_throughput_test.py` | NN inference throughput (CPU vs GPU) |
| `product/nn/cuda_smoke_test.py` | Verifies CUDA availability and basic tensor ops |

---

## File Structure

```text
PROJECT/
├── README.md
├── diary.md
├── .gitlab-ci.yml
└── product/
    ├── main.py                  # Entry point (GUI / terminal selection)
    ├── go_engine.py             # Core game rules and board logic
    ├── mcts_agent.py            # Hand-crafted MCTS AI (UCB1 + heuristics)
    ├── nn_mcts_agent.py         # Neural Network-guided MCTS AI (PUCT)
    ├── game_gui.py              # Tkinter graphical interface
    ├── game_ui.py               # Terminal interface
    ├── play_gui.py              # Quick GUI launcher
    ├── parallel_rollout.py      # Parallel rollout utilities
    ├── tournament.py            # Automated head-to-head evaluation
    ├── profiler_utils.py        # Timing and throughput utilities
    ├── test_go_engine.py        # Engine unit tests
    ├── test_mcts_agent.py       # AI unit tests
    ├── test_ai_speed.py         # Performance benchmarks
    └── nn/
        ├── model.py             # AlphaGoZeroNetwork (CNN: trunk + policy + value heads)
        ├── encoder.py           # GoBoardEncoder (16-channel feature planes)
        ├── dataset.py           # PyTorch dataset for supervised training
        ├── parse_sgf.py         # SGF game record parser
        ├── train.py             # Supervised pre-training (Adam optimizer)
        ├── train_on_gui_games.py# Fine-tuning from GUI-recorded human/AI games
        ├── self_play.py         # Self-Play RL pipeline
        ├── generate_data.py     # Self-play data generation utilities
        ├── test_encoder.py      # Encoder unit tests
        ├── test_dataset.py      # Dataset unit tests
        ├── cnn_throughput_test.py # CNN inference throughput benchmark
        └── cuda_smoke_test.py   # CUDA sanity check
```

---

## Technical Notes

### Why Liberty Caching?
Go requires frequent liberty counting. Naive counting is O(n) per group; caching group IDs and their current liberty sets reduces repeated recalculation during gameplay and across thousands of MCTS simulations.

### Why Tree Reuse?
When the opponent plays a move already explored by MCTS, the corresponding subtree can be re-rooted instead of discarding it, saving many simulations and improving play strength with the same time budget.

### Why PUCT over UCB1?
UCB1 treats all moves uniformly. PUCT incorporates the policy network's prior `P(s,a)`, directing search towards moves the neural network already considers promising, achieving much higher quality search with fewer simulations.

### Why Score-Aware Pass Gating?
The value head can become overconfident about a winning position, causing the agent to pass prematurely and cede territory. The gate blocks passing when the engine's own board evaluation shows a margin below 5 points, acting as a safety net independent of the network's internals.

### Why Supervised Pre-training Before RL?
Starting RL from random weights results in very poor self-play games, making the learning signal weak. Pre-training on ~1.8 M professional games with Adam gives the network a sensible policy prior, so self-play games are immediately more informative and RL converges faster.