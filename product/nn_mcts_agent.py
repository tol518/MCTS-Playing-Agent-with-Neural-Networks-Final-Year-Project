"""
Week 5: Neural Network-guided MCTS Agent using PUCT.

Replaces random rollouts with value network evaluation
and uses policy network priors for PUCT move selection.
"""

import math
import time
import os
import sys
from typing import Optional, Tuple, Dict

# ---------------------------------------------------------------------------
# Lazy imports for torch / NN so multiprocessing workers stay lightweight.
# ---------------------------------------------------------------------------
_torch = None
_model_class = None
_encoder_class = None

def _ensure_imports():
    global _torch, _model_class, _encoder_class
    if _torch is None:
        import torch as _t
        _torch = _t
        try:
            from product.nn.model import AlphaGoZeroNetwork
            from product.nn.encoder import GoBoardEncoder
        except ModuleNotFoundError:
            try:
                from nn.model import AlphaGoZeroNetwork
                from nn.encoder import GoBoardEncoder
            except ModuleNotFoundError:
                import sys
                from pathlib import Path
                product_dir = Path(__file__).resolve().parent
                project_root = product_dir.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                if str(product_dir) not in sys.path:
                    sys.path.insert(0, str(product_dir))
                from nn.model import AlphaGoZeroNetwork
                from nn.encoder import GoBoardEncoder
        _model_class = AlphaGoZeroNetwork
        _encoder_class = GoBoardEncoder


try:
    from product.go_engine import GoBoard, Player, column_to_label
except ModuleNotFoundError:
    from go_engine import GoBoard, Player, column_to_label


class NNMCTSNode:
    """
    A node in the NN-guided MCTS tree.
    Stores policy prior P(s,a) alongside visit counts and value accumulations.
    """
    __slots__ = [
        'board', 'move', 'parent', 'children',
        'prior', 'visit_count', 'total_value', 'player',
        'is_expanded',
    ]

    def __init__(self, board: GoBoard, move: Optional[Tuple[int, int]] = None,
                 parent: Optional['NNMCTSNode'] = None, prior: float = 0.0):
        self.board = board
        self.move = move
        self.parent = parent
        self.children: list = []  # list of NNMCTSNode
        self.prior = prior        # P(s,a) from the policy network
        self.visit_count = 0
        self.total_value = 0.0
        self.player = board.current_player
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        """Mean action value Q(s,a)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def best_child_puct(self, c_puct: float = 1.5) -> 'NNMCTSNode':
        """Select the child with the highest PUCT score."""
        parent_visits = self.visit_count
        sqrt_parent = math.sqrt(parent_visits) if parent_visits > 0 else 1.0

        best_score = -float('inf')
        best_node = None

        for child in self.children:
            # Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)
            q = child.q_value
            u = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u

            if score > best_score:
                best_score = score
                best_node = child

        return best_node


class NNMCTSAgent:
    """
    Neural Network-guided Monte Carlo Tree Search agent.

    Uses the trained AlphaGoZeroNetwork to:
    - Evaluate leaf nodes via the Value head (replacing random rollouts)
    - Bias move selection via the Policy head (PUCT formula)
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        simulation_time: float = 5.0,
        max_simulations: int = None,
        c_puct: float = 1.5,
        device: str = None,
    ):
        _ensure_imports()

        self.simulation_time = simulation_time
        self.max_simulations = max_simulations
        self.c_puct = c_puct

        # Default checkpoint path relative to this file
        # Prefer RL-trained checkpoint over supervised-only
        if checkpoint_path is None:
            _this_dir = os.path.dirname(os.path.abspath(__file__))
            rl_path = os.path.join(_this_dir, "nn", "checkpoints", "rl_last.pt")
            supervised_path = os.path.join(_this_dir, "nn", "checkpoints", "joint_best.pt")
            checkpoint_path = rl_path if os.path.exists(rl_path) else supervised_path

        # Device selection
        if device is None:
            self._device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        else:
            self._device = _torch.device(device)

        # Load model
        self._model = _model_class(
            in_channels=16, num_res_blocks=5, channels=128, board_size=9
        ).to(self._device)
        self._model.eval()

        # Load checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = _torch.load(checkpoint_path, map_location=self._device, weights_only=False)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[NNMCTSAgent] Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"[NNMCTSAgent] WARNING: No checkpoint found at {checkpoint_path}, using random weights!")

        # Encoder
        self._encoder = _encoder_class()

        # Inference cache: board_hash -> (policy_probs, value)
        self._cache: Dict[str, Tuple] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_max_size = 50000  # evict when exceeded to prevent RAM exhaustion

        # Tree reuse
        self._cached_tree = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_move(self, board: GoBoard) -> Optional[Tuple[int, int]]:
        """Select the best move using NN-guided MCTS."""
        # Try to reuse tree from previous search
        root = self._try_reuse_tree(board)
        if root is None:
            root = NNMCTSNode(board.clone())

        # Expand root if needed
        if not root.is_expanded:
            self._expand_node(root)

        start_time = time.time()
        simulations = 0
        reused_visits = root.visit_count

        while True:
            # Termination
            if self.max_simulations and simulations >= self.max_simulations:
                break
            if not self.max_simulations and (time.time() - start_time) >= self.simulation_time:
                break

            # 1. SELECT: traverse tree using PUCT
            node = self._select(root)

            # 2. EXPAND + EVALUATE: expand with policy priors, evaluate with value head
            if not node.is_expanded and not node.board.is_game_over():
                value = self._expand_node(node)
            else:
                # Terminal or already expanded — just get the value
                value = self._evaluate_terminal(node)

            # 3. BACKPROPAGATE
            self._backpropagate(node, value)
            simulations += 1

        # Select the most visited child
        if not root.children:
            return None

        best_child = max(root.children, key=lambda c: c.visit_count)

        # Score-aware pass gate (bidirectional):
        #   - If behind/even → block pass, force a real move
        #   - If massively ahead → force pass to end the game
        my_color = board.current_player
        score_margin = self._quick_score(board, my_color)

        if best_child.move is None:
            # NN wants to pass — only allow if genuinely ahead by 5+
            non_pass = [c for c in root.children if c.move is not None and c.visit_count > 0]
            if non_pass and score_margin < 5.0:
                best_child = max(non_pass, key=lambda c: c.visit_count)
        else:
            # NN wants to play a move — but if we're ahead by 10+,
            # the game is clearly won, so force a pass to end it
            if score_margin >= 10.0:
                # Find the pass child (or just return None directly)
                pass_child = next((c for c in root.children if c.move is None), None)
                if pass_child is not None:
                    best_child = pass_child

        # Cache subtree for reuse
        self._cache_subtree(best_child)

        # Print stats
        reused = f" (+{reused_visits} reused)" if reused_visits > 0 else ""
        elapsed = time.time() - start_time
        print(f"NN-MCTS completed {simulations} simulations{reused} in {elapsed:.2f}s")
        if best_child.move is None:
            move_label = "pass"
        else:
            row, col = best_child.move
            move_label = f"({row + 1}, {column_to_label(col)})"
        win_rate = best_child.q_value
        print(
            f"Selected move: {move_label}, Visits: {best_child.visit_count}, "
            f"Value: {win_rate:.4f}, Cache: {self._cache_hits}h/{self._cache_misses}m"
        )

        return best_child.move

    # ------------------------------------------------------------------
    # Score-aware pass gating
    # ------------------------------------------------------------------

    def _quick_score(self, board, my_color) -> float:
        """
        Return how many points `my_color` is ahead by (negative = behind).
        Uses the GoBoard's built-in Chinese scoring (stones + territory + komi).
        """
        result = board.calculate_score(komi=6.5)
        if my_color == Player.BLACK:
            return result['black'] - result['white']  # positive = Black ahead
        else:
            return result['white'] - result['black']  # positive = White ahead

    # ------------------------------------------------------------------
    # MCTS Phases
    # ------------------------------------------------------------------

    def _select(self, node: NNMCTSNode) -> NNMCTSNode:
        """Walk down the tree using PUCT until we reach an unexpanded node."""
        while node.is_expanded and node.children:
            node = node.best_child_puct(self.c_puct)
        return node

    def _expand_node(self, node: NNMCTSNode) -> float:
        """
        Expand a leaf node:
        1. Run NN inference to get policy priors and value.
        2. Create child nodes for each legal move with their priors.
        3. Return the value estimate.
        """
        board = node.board
        policy_probs, value = self._nn_evaluate(board)

        # Get legal moves
        legal_moves = board.get_legal_moves_fast()
        if not legal_moves:
            legal_moves = []

        # Build move priors from legal board moves
        total_prior = 0.0
        move_priors = []

        for move in legal_moves:
            row, col = move
            idx = row * 9 + col
            prior = policy_probs[idx]
            move_priors.append((move, prior))
            total_prior += prior

        # Always include pass but with a heavy penalty to discourage the
        # Value Head's overconfidence from inflating pass visit counts.
        # The score-aware gate in select_move() is the real safeguard.
        pass_prior = policy_probs[81] if len(policy_probs) > 81 else 0.01
        pass_prior *= 0.01  # 100x penalty to prevent cache-spinning
        move_priors.append((None, pass_prior))
        total_prior += pass_prior

        # Normalize priors so they sum to 1 over legal moves
        if total_prior > 0:
            for move, prior in move_priors:
                new_board = board.clone()
                if move is None:
                    new_board.pass_turn_fast()
                else:
                    row, col = move
                    new_board.make_move_fast(row, col)
                child = NNMCTSNode(new_board, move, node, prior=prior / total_prior)
                node.children.append(child)
        else:
            # Fallback: uniform priors
            uniform = 1.0 / max(1, len(move_priors))
            for move, _ in move_priors:
                new_board = board.clone()
                if move is None:
                    new_board.pass_turn_fast()
                else:
                    row, col = move
                    new_board.make_move_fast(row, col)
                child = NNMCTSNode(new_board, move, node, prior=uniform)
                node.children.append(child)

        node.is_expanded = True
        return value

    def _evaluate_terminal(self, node: NNMCTSNode) -> float:
        """Evaluate a terminal or already-expanded node."""
        if node.board.is_game_over():
            # Use actual game result via simple scoring
            board = node.board
            size = board.size
            black_score = 0.0
            white_score = 0.0
            for r in range(size):
                for c in range(size):
                    stone = board.board[r][c]
                    if stone == Player.BLACK:
                        black_score += 1.0
                    elif stone == Player.WHITE:
                        white_score += 1.0
            black_score += board.captured_stones[Player.BLACK]
            white_score += board.captured_stones[Player.WHITE] + 6.5  # komi
            if node.player == Player.BLACK:
                return 1.0 if black_score > white_score else 0.0
            else:
                return 1.0 if white_score > black_score else 0.0
        else:
            # Already expanded — use cached NN value
            _, value = self._nn_evaluate(node.board)
            return value

    def _backpropagate(self, node: NNMCTSNode, value: float):
        """Propagate value up the tree, flipping perspective at each level."""
        while node is not None:
            node.visit_count += 1
            # The value is from the perspective of the node's player
            # We need to flip it when going up to the parent (opponent's perspective)
            node.total_value += value
            value = 1.0 - value  # flip perspective for parent
            node = node.parent

    # ------------------------------------------------------------------
    # Neural Network Inference
    # ------------------------------------------------------------------

    def _nn_evaluate(self, board: GoBoard) -> Tuple:
        """
        Run the neural network on a board state.
        Returns (policy_probs, value) where value is in [0, 1] from
        the current player's perspective.
        
        Results are cached by board state hash.
        """
        # Create a hashable key from the board state
        board_key = self._board_hash(board)

        if board_key in self._cache:
            self._cache_hits += 1
            return self._cache[board_key]

        self._cache_misses += 1

        # Encode the board into a tensor
        state_tensor = self._encoder.encode(board)  # [16, 9, 9]
        state_tensor = state_tensor.unsqueeze(0).to(self._device)  # [1, 16, 9, 9]

        with _torch.no_grad():
            policy_logits, value_pred = self._model(state_tensor)

        # Convert policy logits to probabilities
        policy_probs = _torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        # Convert value from [-1, 1] to [0, 1]
        # The network outputs value from Black's perspective
        raw_value = value_pred.item()  # in [-1, 1]

        # Convert to current player's win probability [0, 1]
        if board.current_player == Player.BLACK:
            value = (raw_value + 1.0) / 2.0  # map [-1,1] -> [0,1]
        else:
            value = (-raw_value + 1.0) / 2.0  # flip for white

        result = (policy_probs, value)

        # Evict cache if too large to prevent RAM exhaustion
        if len(self._cache) >= self._cache_max_size:
            self._cache.clear()

        self._cache[board_key] = result
        return result

    def _board_hash(self, board: GoBoard) -> str:
        """Create a hashable representation of the board state."""
        return str(board.get_board_state()) + str(board.current_player)

    # ------------------------------------------------------------------
    # Tree Reuse
    # ------------------------------------------------------------------

    def _try_reuse_tree(self, board: GoBoard) -> Optional[NNMCTSNode]:
        """Try to find a subtree from the cached tree that matches the current board."""
        if self._cached_tree is None:
            return None

        for child in self._cached_tree.children:
            if child.board.get_board_state() == board.get_board_state():
                child.parent = None
                return child

        self._cached_tree = None
        return None

    def _cache_subtree(self, node: NNMCTSNode):
        """Cache the subtree rooted at node for potential reuse."""
        node.parent = None
        self._cached_tree = node

    def shutdown(self):
        """Clean up resources."""
        self._cache.clear()
