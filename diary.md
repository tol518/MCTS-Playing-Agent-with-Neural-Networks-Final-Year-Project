Go-playing AI using Monte Carlo Tree Search (MCTS) for 9x9 boards. Features full game engine, strategic AI, and both GUI and terminal interfaces.

## 📅 Development Timeline

### Phase 1: Core Game Engine

- ✅ Implemented 9x9 Go board with complete rule validatin
- ✅ Added capture mechanics, ko rule, suicide prevention
- ✅ Built territory scoring with komi
- ✅ Created group detection and liberty counting
- ✅ Comprehensive unit tests (10 tests, all passing)

### Phase 2: MCTS AI Implementation

- ✅ Built full MCTS algorithm with four phases
- ✅ Implemented UCB1 selection formula
- ✅ Added configurable thinking time (0.5-60s)
- ✅ Created simulation with fast heuristics
- ✅ Performance: 500+ simulations in 5 seconds

### Phase 3: User Interfaces

- ✅ **Terminal Interface**: Text-based gameplay with menu system
- ✅ **Graphical Interface**: Full 2D GUI with tkinter
- ✅ Added multiple game modes (Human vs AI, Human vs Human, AI vs AI)
- ✅ Visual feedback, move preview, coordinate labels

### Phase 4: Optimization & Fixes

- 🐛 **Critical Bug**: AI always passed instead of playing
- ✅ **Root Cause**: Pass move added to untried moves from start
- ✅ **Fix**: Only use pass when no legal moves available
- 🐛 **Performance Issue**: Only 4 simulations in 6 seconds
- ✅ **Fix**: Removed expensive board cloning in simulations
- ✅ **Result**: 500+ simulations in 2-5 seconds

## 🏗️ Architecture Decisions

### Core Components

- `go_engine.py` (500+ lines): Complete Go game logic
- `mcts_agent.py` (350+ lines): MCTS AI implementation
- `game_ui.py` (300+ lines): Terminal interface
- `game_gui.py` (580+ lines): Graphical interface

### Key Design Choices

- **Modular**: Separate engine, AI, and UI layers
- **Type Hints**: Full type annotations throughout
- **Docstrings**: Comprehensive documentation
- **Error Handling**: Robust input validation
- **Performance**: Fast simulations without board cloning

## 🧪 Testing & Quality

- **Unit Tests**: 10 comprehensive tests covering core functionality
- **AI Verification**: Speed tests ensure proper move selection
- **Integration**: Both interfaces tested and working
- **Code Quality**: Clean, readable, well-documented

## 🎮 Final Features

- **Game Engine**: Complete 9x9 Go with all rules
- **AI Strength**: Intermediate level (5s thinking)
- **Interfaces**: GUI (recommended) + Terminal
- **Game Modes**: Human vs AI (both colors), Human vs Human
- **Performance**: Production-ready, responsive

## 📊 Project Metrics

- **Total Files**: 14 (7 Python, 4 docs, 3 config)
- **Code Lines**: ~1,500 lines of Python
- **Test Coverage**: Core engine fully tested
- **Status**: ✅ Production ready

```bash
pip install numpy
python main.py  # Choose GUI or terminal
```
