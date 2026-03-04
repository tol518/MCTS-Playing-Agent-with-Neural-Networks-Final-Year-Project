"""
Week 6: Self-Play Reinforcement Learning Loop.

Generates training data by playing the NN-guided MCTS against itself,
then trains the network on (state, policy, value) tuples derived from
its own games.  This is the key step that transforms the imitation-trained
network into one that genuinely understands Go strategy.
"""

import os
import sys
import time
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── path setup ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_DIR = PROJECT_ROOT / "product"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PRODUCT_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCT_DIR))

try:
    from product.nn.model import AlphaGoZeroNetwork
    from product.nn.encoder import GoBoardEncoder
    from product.go_engine import GoBoard, Player
except ModuleNotFoundError:
    from nn.model import AlphaGoZeroNetwork
    from nn.encoder import GoBoardEncoder
    from go_engine import GoBoard, Player


# ════════════════════════════════════════════════════════════════════════
#  Self-Play Game Generator
# ════════════════════════════════════════════════════════════════════════

class SelfPlayGame:
    """
    Plays one complete game of NN-guided MCTS vs itself.
    Records (state_tensor, mcts_policy, current_player) at every move.
    After the game, fills in the value targets from the actual outcome.
    """

    def __init__(
        self,
        model: AlphaGoZeroNetwork,
        encoder: GoBoardEncoder,
        device: torch.device,
        board_size: int = 9,
        simulations: int = 200,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        temp_drop_move: int = 16,
    ):
        self.model = model
        self.encoder = encoder
        self.device = device
        self.board_size = board_size
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.temp_drop_move = temp_drop_move
        self.action_size = board_size * board_size + 1  # 82

    def play(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Play one complete game.  Returns:
            states  – [N, 16, 9, 9]
            policies – [N, 82]
            values   – [N, 1]
        """
        board = GoBoard(self.board_size)

        states: List[torch.Tensor] = []
        policies: List[torch.Tensor] = []
        players: List[Player] = []

        move_num = 0
        consecutive_passes = 0
        max_moves = 90   # 9x9 games rarely exceed 81 meaningful moves
        game_timeout = 180.0  # max seconds per game
        game_start = time.time()

        while not board.is_game_over() and move_num < max_moves and consecutive_passes < 2:
            # Safety timeout
            if time.time() - game_start > game_timeout:
                break
            # ── Encode current state ────────────────────────────────
            state_tensor = self.encoder.encode(board)
            states.append(state_tensor)
            players.append(board.current_player)

            # ── Run mini-MCTS to get visit counts ───────────────────
            move, policy_vec = self._mcts_search(board, move_num)
            policies.append(policy_vec)

            # ── Apply the move ──────────────────────────────────────
            if move is None:
                board.pass_turn()
                consecutive_passes += 1
            else:
                row, col = move
                if board.is_legal_move(row, col):
                    board.make_move(row, col)
                    consecutive_passes = 0
                else:
                    board.pass_turn()
                    consecutive_passes += 1

            move_num += 1

        # ── Determine winner ────────────────────────────────────────
        winner = self._score_game(board)

        # ── Build value targets from game outcome ───────────────────
        value_list = []
        for p in players:
            if p == winner:
                value_list.append(1.0)
            elif winner is None:
                value_list.append(0.0)   # draw (rare in Go)
            else:
                value_list.append(-1.0)

        states_t = torch.stack(states)           # [N, 16, 9, 9]
        policies_t = torch.stack(policies)       # [N, 82]
        values_t = torch.tensor(value_list, dtype=torch.float32).unsqueeze(1)  # [N, 1]

        return states_t, policies_t, values_t

    # ── Mini MCTS (lightweight, no tree reuse) ──────────────────────

    def _mcts_search(
        self, board: GoBoard, move_num: int,
    ) -> Tuple[Optional[Tuple[int, int]], torch.Tensor]:
        """
        Run a small MCTS search from the current position.
        Returns the selected move and the MCTS visit-count policy vector.
        """
        root = _MCTSNode(board.clone())
        self._expand(root)

        for _ in range(self.simulations):
            node = root
            # SELECT
            while node.is_expanded and node.children:
                node = self._puct_select(node)
            # EXPAND + EVALUATE
            if not node.is_expanded and not node.board.is_game_over():
                value = self._expand(node)
            else:
                value = self._terminal_value(node)
            # BACKPROP
            self._backprop(node, value)

        # ── Build policy from visit counts ──────────────────────────
        visit_counts = torch.zeros(self.action_size, dtype=torch.float32)
        for child in root.children:
            if child.move is None:
                visit_counts[self.action_size - 1] = child.visit_count
            else:
                r, c = child.move
                visit_counts[r * self.board_size + c] = child.visit_count

        # Temperature-based selection
        if move_num < self.temp_drop_move:
            # Proportional to visit counts (exploration)
            temp = self.temperature
            if visit_counts.sum() > 0:
                probs = visit_counts ** (1.0 / temp)
                probs /= probs.sum()
            else:
                probs = torch.ones(self.action_size) / self.action_size
        else:
            # Greedy (pick the most visited)
            probs = torch.zeros(self.action_size)
            best_idx = visit_counts.argmax().item()
            probs[best_idx] = 1.0

        # Sample move from policy
        action = torch.multinomial(probs, 1).item()

        if action == self.action_size - 1:
            chosen_move = None  # pass
        else:
            chosen_move = (action // self.board_size, action % self.board_size)

        # The mcts policy target is the normalized visit counts (not the temperature-sampled one)
        policy_target = visit_counts / max(1.0, visit_counts.sum().item())

        return chosen_move, policy_target

    def _expand(self, node: '_MCTSNode') -> float:
        """Expand node with policy priors, return value estimate."""
        board = node.board
        state = self.encoder.encode(board).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value_pred = self.model(state)

        policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        raw_value = value_pred.item()

        # Value from current player's perspective [0, 1]
        if board.current_player == Player.BLACK:
            value = (raw_value + 1.0) / 2.0
        else:
            value = (-raw_value + 1.0) / 2.0

        # Create children for legal moves — limit branching to top-K by prior
        legal_moves = board.get_legal_moves_fast()
        if not legal_moves:
            legal_moves = []

        # Score all legal moves by their policy prior
        scored = []
        for move in legal_moves:
            r, c = move
            prior = policy_probs[r * 9 + c]
            scored.append((move, prior))

        # Keep only top-30 moves (by prior) to limit tree width
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:30]

        total_prior = 0.0
        move_priors = []
        for move, prior in scored:
            move_priors.append((move, prior))
            total_prior += prior

        # Add pass (only late game)
        num_moves = len(board.move_history)
        if num_moves >= 50 or not legal_moves:
            pass_prior = policy_probs[81]
            move_priors.append((None, pass_prior))
            total_prior += pass_prior

        # Normalize and create children
        for move, prior in move_priors:
            new_board = board.clone()
            if move is None:
                new_board.pass_turn_fast()
            else:
                r, c = move
                new_board.make_move_fast(r, c)
            p = prior / total_prior if total_prior > 0 else 1.0 / len(move_priors)
            child = _MCTSNode(new_board, move=move, parent=node, prior=p)
            node.children.append(child)

        node.is_expanded = True
        return value

    def _puct_select(self, node: '_MCTSNode') -> '_MCTSNode':
        sqrt_parent = math.sqrt(node.visit_count) if node.visit_count > 0 else 1.0
        best_score = -float('inf')
        best_child = None
        for child in node.children:
            q = child.total_value / child.visit_count if child.visit_count > 0 else 0.0
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _backprop(self, node: '_MCTSNode', value: float):
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = 1.0 - value
            node = node.parent

    def _terminal_value(self, node: '_MCTSNode') -> float:
        if node.board.is_game_over():
            winner = self._score_game(node.board)
            if winner == node.board.current_player:
                return 1.0
            elif winner is None:
                return 0.5
            else:
                return 0.0
        # Already expanded — re-evaluate
        state = self.encoder.encode(node.board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, v = self.model(state)
        raw = v.item()
        if node.board.current_player == Player.BLACK:
            return (raw + 1.0) / 2.0
        else:
            return (-raw + 1.0) / 2.0

    def _score_game(self, board: GoBoard) -> Optional[Player]:
        """Simple area scoring. Returns the winner or None for draw."""
        try:
            score = board.calculate_score()
            return score.get('winner', None)
        except Exception:
            # Fallback: count stones + captures + komi
            size = board.size
            black = 0.0
            white = 0.0
            for r in range(size):
                for c in range(size):
                    s = board.board[r][c]
                    if s == Player.BLACK:
                        black += 1.0
                    elif s == Player.WHITE:
                        white += 1.0
            black += board.captured_stones[Player.BLACK]
            white += board.captured_stones[Player.WHITE] + 6.5
            if black > white:
                return Player.BLACK
            elif white > black:
                return Player.WHITE
            return None


class _MCTSNode:
    """Lightweight MCTS node for self-play."""
    __slots__ = ['board', 'move', 'parent', 'children',
                 'prior', 'visit_count', 'total_value', 'is_expanded']

    def __init__(self, board, move=None, parent=None, prior=0.0):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = []
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.is_expanded = False


# ════════════════════════════════════════════════════════════════════════
#  Training Loop
# ════════════════════════════════════════════════════════════════════════

def train_on_buffer(
    model: AlphaGoZeroNetwork,
    optimizer: torch.optim.Optimizer,
    buffer_states: torch.Tensor,
    buffer_policies: torch.Tensor,
    buffer_values: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    epochs: int = 5,
):
    """Train the model on the replay buffer for a few epochs."""
    model.train()
    dataset = TensorDataset(buffer_states, buffer_policies, buffer_values)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        n_batches = 0

        for states, policies, values in loader:
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

            optimizer.zero_grad()

            policy_logits, value_pred = model(states)

            p_loss = policy_criterion(policy_logits, policies)
            v_loss = value_criterion(value_pred, values)
            loss = p_loss + v_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_p = total_p_loss / max(1, n_batches)
        avg_v = total_v_loss / max(1, n_batches)
        print(f"  Train epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} "
              f"(policy={avg_p:.4f}, value={avg_v:.4f})")

    model.eval()


# ════════════════════════════════════════════════════════════════════════
#  Main Self-Play Loop
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Week 6: Self-Play RL Loop")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of self-play → train iterations")
    parser.add_argument("--games-per-iter", type=int, default=20,
                        help="Games of self-play per iteration")
    parser.add_argument("--simulations", type=int, default=200,
                        help="MCTS simulations per move during self-play")
    parser.add_argument("--train-epochs", type=int, default=5,
                        help="Training epochs per iteration on replay buffer")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for Adam")
    parser.add_argument("--buffer-size", type=int, default=50000,
                        help="Max samples to keep in replay buffer")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Exploration temperature for first N moves")
    parser.add_argument("--temp-drop", type=int, default=16,
                        help="Move number to switch from temp to greedy")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load or create model ────────────────────────────────────────
    model = AlphaGoZeroNetwork(
        in_channels=16, num_res_blocks=5, channels=128, board_size=9
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    selfplay_data_dir = Path(__file__).resolve().parent / "data" / "selfplay"
    selfplay_data_dir.mkdir(parents=True, exist_ok=True)

    start_iter = 0

    # Try to resume
    rl_ckpt = checkpoint_dir / "rl_last.pt"
    joint_ckpt = checkpoint_dir / "joint_best.pt"

    if args.resume and rl_ckpt.exists():
        ckpt = torch.load(rl_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_iter = ckpt.get("iteration", 0)
        print(f"Resumed from RL checkpoint at iteration {start_iter}")
    elif joint_ckpt.exists():
        ckpt = torch.load(joint_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Initialized from supervised checkpoint: {joint_ckpt}")
    else:
        print("Starting from scratch (random weights)")

    encoder = GoBoardEncoder()

    # ── Replay buffer (circular) ────────────────────────────────────
    buffer_states = []
    buffer_policies = []
    buffer_values = []

    best_loss = float('inf')

    # ── Main loop ───────────────────────────────────────────────────
    for iteration in range(start_iter, start_iter + args.iterations):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}")
        print(f"{'='*60}")

        # ── Phase 1: Self-Play ──────────────────────────────────────
        print(f"\nPhase 1: Generating {args.games_per_iter} self-play games "
              f"({args.simulations} sims/move)...")
        model.eval()

        game_generator = SelfPlayGame(
            model=model,
            encoder=encoder,
            device=device,
            simulations=args.simulations,
            c_puct=1.5,
            temperature=args.temperature,
            temp_drop_move=args.temp_drop,
        )

        iter_states = []
        iter_policies = []
        iter_values = []
        total_moves = 0
        black_wins = 0
        white_wins = 0

        for game_num in range(args.games_per_iter):
            game_start = time.time()
            s, p, v = game_generator.play()
            game_time = time.time() - game_start

            iter_states.append(s)
            iter_policies.append(p)
            iter_values.append(v)
            total_moves += s.size(0)

            # Count wins
            if v[0].item() > 0:
                black_wins += 1
            else:
                white_wins += 1

            print(f"  Game {game_num + 1}/{args.games_per_iter}: "
                  f"{s.size(0)} moves, {game_time:.1f}s "
                  f"(winner: {'Black' if v[0].item() > 0 else 'White'})")

        print(f"\nSelf-play done: {total_moves} positions, "
              f"B wins: {black_wins}, W wins: {white_wins}")

        # ── Add to replay buffer ────────────────────────────────────
        new_states = torch.cat(iter_states)
        new_policies = torch.cat(iter_policies)
        new_values = torch.cat(iter_values)

        buffer_states.append(new_states)
        buffer_policies.append(new_policies)
        buffer_values.append(new_values)

        # Concatenate and trim to buffer size
        all_states = torch.cat(buffer_states)
        all_policies = torch.cat(buffer_policies)
        all_values = torch.cat(buffer_values)

        if all_states.size(0) > args.buffer_size:
            # Keep the most recent samples
            all_states = all_states[-args.buffer_size:]
            all_policies = all_policies[-args.buffer_size:]
            all_values = all_values[-args.buffer_size:]

        buffer_states = [all_states]
        buffer_policies = [all_policies]
        buffer_values = [all_values]

        print(f"Replay buffer: {all_states.size(0)} samples")

        # ── Save self-play data ─────────────────────────────────────
        sp_path = selfplay_data_dir / f"selfplay_iter{iteration+1:03d}.pt"
        torch.save({
            'states': new_states, 'policies': new_policies, 'values': new_values
        }, sp_path)

        # ── Phase 2: Train ──────────────────────────────────────────
        print(f"\nPhase 2: Training for {args.train_epochs} epochs "
              f"on {all_states.size(0)} samples...")

        train_on_buffer(
            model, optimizer,
            all_states, all_policies, all_values,
            device,
            batch_size=args.batch_size,
            epochs=args.train_epochs,
        )

        # ── Save checkpoint ─────────────────────────────────────────
        ckpt_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration + 1,
        }
        torch.save(ckpt_data, checkpoint_dir / "rl_last.pt")

        iter_time = time.time() - iter_start
        print(f"\nIteration {iteration + 1} completed in {iter_time:.1f}s")

    print(f"\n{'='*60}")
    print("Self-Play RL training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
