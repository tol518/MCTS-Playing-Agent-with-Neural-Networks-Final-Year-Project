# Development Diary (high-level, from git commits)

**Scope:** This diary is derived from `git log` on the `main` branch. It describes what was *committed* and adds high-level context about the likely intent and impact of each milestone.

**Note:** Uncommitted/untracked work (e.g. currently `interim_report.md`) will not appear here until committed.

---

### 2025-09-24 — Project kickoff and repo baseline
- **Context**: The project is being formalised as a deliverable (FYP), so the first step is making it traceable, reproducible, and easy to understand.
- **What happened (commit: `50e388e`)**: Repository initialized with a minimal starting point (`README.md`).
- **Outcome**: A clean baseline that future work can build on, with an initial “source of truth” for what the project is.
- **Next focus**: Deliver a first playable end-to-end version.

### 2025-10-25 — First end-to-end “vertical slice”: playable Go + baseline AI
- **Context**: Early success criteria is not perfect play; it’s having the full system working: rules engine + UI + an AI that can take turns.
- **What happened (commit: `b16b2a3`)**: Core implementation landed: Go rules/engine, MCTS agent, and both GUI + terminal ways to play, plus early tests/benchmarks scaffolding.
- **Outcome**: The project becomes demonstrable: you can run a game, interact with the board, and play against an AI.
- **Next focus**: Stabilize correctness (rules edge-cases), then iterate on performance and usability.

### 2025-10-25 — Automation foundation: CI added
- **Context**: As complexity grows (rules engine + AI search), regressions become easy; automation is needed to keep quality under control.
- **What happened (commit: `e0e5e5d`)**: CI configuration added.
- **Outcome**: A pathway to consistent, repeatable checks when code changes.
- **Next focus**: Expand tests so CI meaningfully guards correctness and prevents performance regressions.

### 2025-10-25 — Branch history consolidation and docs polish
- **Context**: After the initial drop, the priority is making the repository “presentable”: coherent branch history and better onboarding.
- **What happened**:
  - `c20ca7d`: merged `master` → `main` (administrative integration).
  - `bd95f37`: adjusted `diary.md`.
  - `95d3059`: refined `README.md`.
- **Outcome**: Cleaner project presentation for supervision/review and easier navigation for a new reader.
- **Next focus**: Turn the prototype into a more robust system (tests, speed, better UX).

### 2025-12-02 — Hardening milestone: correctness, speed, and UX improvements
- **Context**: Once the vertical slice exists, the real engineering work is making it reliable and fast enough for repeated AI simulations, while improving player experience.
- **What happened (commit: `415a404`)**:
  - Testing expanded (including AI-focused tests).
  - Engine + MCTS improved for logic correctness and performance.
  - UI improvements shipped, including **undo/redo** controls (critical for usability and debugging gameplay/AI decisions).
- **Outcome**: A more trustworthy and usable system: better protected against regressions, faster iterations, and more user-friendly gameplay.
- **Next focus**: Measure/validate AI strength and performance systematically; consider stronger playout heuristics, better evaluation, and richer tooling (e.g. SGF save/load).

### 2025-12-02 — Integration + documentation capture
- **Context**: After a major milestone, the work is consolidated and written up while details are fresh.
- **What happened**:
  - `626938b` and `89ae257`: merges from a testing-focused branch into `main` (integration events).
  - `8908224`: diary filled out.
- **Outcome**: The milestone is merged, stabilized, and documented, making it easier to explain in reports and demos.
- **Next focus**: Future work can build from a stable `main` with documented design choices.

---

### 2025-12-19 — Documentation polish: diary and README tidied up
- **Context**: After a feature push the written-up documentation had stale or rough sections; a dedicated cleanup pass was needed before moving on to the next major phase of work.
- **What happened**:
  - `9ad4646`: diary and README both adjusted/cleaned up (merged via MR!4).
  - `874bc41`: a further minor README tweak (merged via MR!5).
- **Outcome**: The project's written face is cleaner and more accurate for anyone reading the repo cold (supervisor, examiner, future self).
- **Next focus**: Begin the neural-network / reinforcement-learning phase of the project.

---

### 2026-03-04 — Major milestone: Neural-Network MCTS + Self-Play RL pipeline
- **Context**: The vanilla MCTS agent was competitive but had no learned intuition. The next step towards AlphaGo-style play is combining a neural network with MCTS and then training that network through self-play rather than purely human-authored heuristics.
- **What happened (commit: `c5da982`, merged via MR!6)**:
  - **Engine throughput**: Baseline MCTS was re-optimised to hit thousands of simulations per second — a prerequisite for affordable self-play.
  - **AlphaGo-style CNN**: Designed and implemented a PyTorch convolutional network with a shared trunk feeding both a **Policy head** (move probabilities) and a **Value head** (position evaluation).
  - **Supervised pre-training**: The CNN was pre-trained on ~1.8 million professional game samples using the **Adam optimizer**, giving it a strong prior before RL begins.
  - **NN-MCTS integration**: The network was plugged into MCTS via **PUCT selection** (policy-guided upper-confidence bound) and value-based leaf evaluation, replacing random rollouts.
  - **Self-Play RL pipeline**: An automated loop was built so the agent generates its own training data by playing against itself, then updates the network — a core AlphaZero-style feedback cycle.
  - **Overconfidence guard**: A safeguard was added to prevent the NN from passing too early when it incorrectly thinks it is winning by a large margin.
- **Outcome**: The project moves from a search-only MCTS agent to a learning agent. The NN-guided MCTS demonstrably plays stronger moves and the RL loop allows continued improvement without human intervention.
- **Next focus**: Integrate the NN-MCTS agent into the GUI so it can be played against and evaluated interactively; continue RL training cycles.

---

### 2026-03-18 — GUI integration of NN-MCTS, score-aware passing, and system robustness
- **Context**: The NN-MCTS existed as a back-end capability but was not yet accessible through the graphical interface that players and evaluators actually use. Additionally, practical issues (multiprocessing crashes, RAM growth) needed fixing before the system could be used reliably for extended sessions.
- **What happened (commit: `cda52fe`, merged via MR!7)**:
  - **NN-MCTS in the GUI**: The neural-network-guided MCTS agent was surfaced in the game GUI, making it selectable as an opponent for human or AI-vs-AI play.
  - **Score-aware pass logic**: Passing behaviour was made smarter — the agent now factors in the estimated score before deciding to pass, reducing premature resignations and wrong-game-end decisions.
  - **Multiprocess enum pickling fix**: A crash caused by Python's `multiprocessing` module failing to pickle custom enums across process boundaries was diagnosed and resolved.
  - **RAM optimisation**: Cache size limiting and explicit garbage-collection triggers were added to prevent memory growth during long self-play or evaluation sessions.
  - **Milestone result**: "NN is now good enough to beat the old MCTS" — the learned agent surpasses the hand-crafted baseline.
- **Outcome**: The NN-MCTS is now a first-class, stable, playable agent in the full system. The project has a demonstrable AI that out-plays the original MCTS, which is a key FYP deliverable.
- **Next focus**: Continued RL training, evaluation metrics, and final project write-up.
