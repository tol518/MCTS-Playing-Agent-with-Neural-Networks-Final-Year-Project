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
