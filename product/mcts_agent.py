import math
import random
import time
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Tuple, Set
try:
    from product.go_engine import GoBoard, Player, column_to_label
    from product.parallel_rollout import run_batch_rollouts, serialize_board_state
except ModuleNotFoundError:
    from go_engine import GoBoard, Player, column_to_label
    from parallel_rollout import run_batch_rollouts, serialize_board_state


class MCTSNode:
    """
    A node in the MCTS tree.
    Each node represents a game state and stores statistics for move selection.
    """
    
    def __init__(self, board: GoBoard, move: Optional[Tuple[int, int]] = None, 
                 parent: Optional['MCTSNode'] = None, use_fast_legal: bool = False):
        """
        Initialize an MCTS node.
        
        Args:
            board: The game board state at this node
            move: The move that led to this state (row, col) or None for root
            parent: The parent node in the tree
        """
        self.board = board
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.use_fast_legal = use_fast_legal
        if use_fast_legal:
            self.untried_moves = board.get_legal_moves_fast()
        else:
            self.untried_moves = board.get_legal_moves()
        # If no legal moves exist, the only option is to pass
        if not self.untried_moves:
            self.untried_moves = [None]
        else:
            # Shuffle and sort by distance to center (center-biased expansion)
            # Sort descending so center moves are at end (pop() takes from end)
            random.shuffle(self.untried_moves)
            center = board.size // 2
            self.untried_moves.sort(
                key=lambda m: abs(m[0] - center) + abs(m[1] - center) if m else -1,
                reverse=True
            )
        self.player = board.current_player
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible moves have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        return self.board.is_game_over()
    
    def best_child(self, exploration_weight: float = 1.414) -> 'MCTSNode':
        def ucb1_score(child: 'MCTSNode') -> float:
            if child.visits == 0:
                return float('inf')
            
            exploitation = child.wins / child.visits
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            return exploitation + exploration
        
        return max(self.children, key=ucb1_score)
    
    def expand(self) -> 'MCTSNode':
        move = self.untried_moves.pop()
        new_board = self.board.clone()
        
        if move is None:
            # Pass move
            if self.use_fast_legal:
                new_board.pass_turn_fast()
            else:
                new_board.pass_turn()
        else:
            # Regular move
            row, col = move
            if self.use_fast_legal:
                new_board.make_move_fast(row, col)
            else:
                new_board.make_move(row, col)
        
        child_node = MCTSNode(new_board, move, self, use_fast_legal=self.use_fast_legal)
        self.children.append(child_node)
        return child_node
    
    def update(self, result: float):
        self.visits += 1
        self.wins += result


class MCTSAgent:
    
    def __init__(self, simulation_time: float = 5.0, max_simulations: int = None,
                 exploration_weight: float = 1.414):
       
        self.simulation_time = simulation_time
        self.max_simulations = max_simulations
        self.exploration_weight = exploration_weight
        
        # === adaptive heuristic weights (learn which heuristics work best) ===
        # seeded with stronger priors: center and liberties matter more initially
        self.heuristic_weights = {
            'center': 1.5,      # favor center play from the start
            'liberties': 1.3,   # value breathing room
            'connect': 1.2,     # connecting is good
            'pressure': 1.4,    # captures matter
            'follow': 1.0,      # neutral
            'pattern': 1.0,     # learns over time
            'edge_penalty': 1.0,  # penalize edge moves
        }
        self.heuristic_stats = {k: {'wins': 0.0, 'count': 0} for k in self.heuristic_weights}
        
        # === lightweight pattern table (learns from experience) ===
        # key: (friendly_neighbors, enemy_neighbors, libs_after, dist_to_center_bucket)
        # value: {'wins': float, 'count': int}
        # seed with some basic priors
        self.pattern_table = {
            # center moves with good liberties tend to win
            (0, 0, 4, 0): {'wins': 6.0, 'count': 10},  # open center, 4 libs
            (0, 0, 4, 1): {'wins': 5.5, 'count': 10},  # near center
            (1, 0, 4, 0): {'wins': 6.5, 'count': 10},  # connecting in center
            (1, 0, 4, 1): {'wins': 6.0, 'count': 10},
            # edge moves with few libs tend to lose
            (0, 0, 2, 3): {'wins': 3.0, 'count': 10},  # corner/edge, 2 libs
            (0, 0, 3, 3): {'wins': 3.5, 'count': 10},  # edge, 3 libs
        }
        
        # === TREE REUSE ===
        self._cached_tree = None  # root node from previous turn
        self._last_board_hash = None
        # === REUSABLE SIM BOARD FOR ROLLOUTS ===
        self._sim_board = None
        # === PARALLEL ROLLOUTS ===
        self._num_workers = max(1, os.cpu_count() - 1)  # leave one core for main thread
        self._use_parallel = self._num_workers > 1
        self._rollouts_per_batch = 8  # rollouts per worker task
        self._executor = None  # lazily created, persistent pool
    
    def select_move(self, board: GoBoard) -> Optional[Tuple[int, int]]:
        # === TREE REUSE: try to find subtree from previous search ===
        root = self._try_reuse_tree(board)
        if root is None:
            root = MCTSNode(board.clone(), use_fast_legal=True)
        
        # Run simulations
        start_time = time.time()
        simulations = 0
        reused_visits = root.visits  # track how many we got for free
        
        # Use parallel rollouts if available
        if self._use_parallel and self._num_workers > 1:
            simulations = self._run_parallel_mcts(root, start_time)
        else:
            simulations = self._run_sequential_mcts(root, start_time)
        
        # Select the move with the most visits (most robust choice)
        if not root.children:
            # No legal moves, must pass
            return None
        
        best_child = self._select_best_legal_child(root, board)
        if best_child is None:
            return None
        
        # === TREE REUSE: cache the chosen subtree for next turn ===
        self._cache_subtree(best_child)
        
        reused = f" (+{reused_visits} reused)" if reused_visits > 0 else ""
        print(f"MCTS completed {simulations} simulations{reused} in {time.time() - start_time:.2f}s")
        if best_child.move is None:
            move_label = "pass"
        else:
            row, col = best_child.move
            move_label = f"({row + 1}, {column_to_label(col)})"
        print(
            f"Selected move: {move_label}, Visits: {best_child.visits}, "
            f"Win rate: {best_child.wins / best_child.visits if best_child.visits > 0 else 0:.2%}"
        )
        
        return best_child.move

    def _select_best_legal_child(self, root: MCTSNode, board: GoBoard) -> Optional[MCTSNode]:
        """Pick the most visited child that is legal on the real board."""
        children_sorted = sorted(root.children, key=lambda c: c.visits, reverse=True)
        for child in children_sorted:
            if child.move is None:
                return child
            row, col = child.move
            if board.is_legal_move(row, col):
                return child
        return None
    
    def _run_sequential_mcts(self, root: MCTSNode, start_time: float) -> int:
        """Run MCTS with sequential rollouts (single-threaded)."""
        simulations = 0
        
        while True:
            # Check termination conditions
            if self.max_simulations and simulations >= self.max_simulations:
                break
            if not self.max_simulations and (time.time() - start_time) >= self.simulation_time:
                break
            
            # === conditioning on tree statistics ===
            avoided_moves = set()
            for child in root.children:
                if child.visits >= 5 and child.move is not None:
                    win_rate = child.wins / child.visits
                    if win_rate < 0.3:
                        avoided_moves.add(child.move)
            
            # MCTS four phases
            node = self._select(root)
            if not node.is_terminal():
                node = self._expand(node)
            result = self._simulate(node, avoided_moves)
            self._backpropagate(node, result)
            simulations += 1
        
        return simulations
    
    def _get_executor(self):
        """Get or create the persistent process pool."""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self._num_workers)
        return self._executor

    def shutdown(self):
        """Shut down the process pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def _run_parallel_mcts(self, root: MCTSNode, start_time: float) -> int:
        """Run MCTS with parallel rollouts using multiprocessing."""
        simulations = 0
        # Use num_workers (not *2) to reduce simultaneous memory pressure on Windows
        tasks_per_batch = self._num_workers
        rollouts_per_task = self._rollouts_per_batch
        
        try:
            executor = self._get_executor()
        except Exception:
            return self._run_sequential_mcts(root, start_time)
        
        while True:
            # Check termination
            if self.max_simulations and simulations >= self.max_simulations:
                break
            if not self.max_simulations and (time.time() - start_time) >= self.simulation_time:
                break
            
            # Collect nodes to simulate in parallel
            nodes_to_sim = []
            for _ in range(tasks_per_batch):
                node = self._select(root)
                if not node.is_terminal():
                    node = self._expand(node)
                nodes_to_sim.append(node)
            
            # Submit parallel rollouts (multiple rollouts per task)
            futures = []
            for node in nodes_to_sim:
                board_state = serialize_board_state(node.board)
                try:
                    future = executor.submit(
                        run_batch_rollouts,
                        board_state,
                        node.player,
                        rollouts_per_task
                    )
                    futures.append((future, node))
                except Exception:
                    # Pool broken, fall back to sequential
                    self._executor = None
                    result = self._simulate(node, set())
                    self._backpropagate(node, result)
                    simulations += 1
            
            # Collect results and backpropagate
            for future, node in futures:
                try:
                    results = future.result(timeout=5.0)
                    for result in results:
                        self._backpropagate(node, result)
                        simulations += 1
                except Exception:
                    result = self._simulate(node, set())
                    self._backpropagate(node, result)
                    simulations += 1
        
        return simulations
    
    def _try_reuse_tree(self, board: GoBoard) -> Optional[MCTSNode]:
        """Try to find a subtree from cached tree that matches current board"""
        if self._cached_tree is None:
            return None
        
        # check if opponent made a move we explored
        for child in self._cached_tree.children:
            if child.board.get_board_state() == board.get_board_state():
                # found matching subtree!
                child.parent = None  # detach from old tree
                return child
        
        # no match, discard cache
        self._cached_tree = None
        return None
    
    def _cache_subtree(self, node: MCTSNode):
        """Cache the subtree rooted at node for potential reuse"""
        node.parent = None  # detach from parent
        self._cached_tree = node
    
    def _update_heuristic_weights(self, heuristics_used: List[str], result: float):
        """adapt heuristic weights based on which ones led to winning simulations"""
        for h in heuristics_used:
            if h in self.heuristic_stats:
                self.heuristic_stats[h]['wins'] += result
                self.heuristic_stats[h]['count'] += 1
        
        # periodically recalculate weights based on accumulated stats
        for h, stats in self.heuristic_stats.items():
            if stats['count'] >= 20:  # enough data to adjust
                success_rate = stats['wins'] / stats['count']
                # scale weight: good heuristics get boosted, bad ones shrink
                self.heuristic_weights[h] = 0.5 + success_rate  # range [0.5, 1.5]
                # decay stats slowly so we keep adapting
                stats['wins'] *= 0.9
                stats['count'] = int(stats['count'] * 0.9)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal() and node.is_fully_expanded():
            if not node.children:
                # No children to select (e.g., forced pass state)
                break
            node = node.best_child(self.exploration_weight)
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        return node.expand()
    
    def _simulate(self, node: MCTSNode, avoided_moves: set = None) -> float:
        """Run a fast rollout and return result."""
        # reuse a single board to avoid per-rollout clones
        if self._sim_board is None or self._sim_board.size != node.board.size:
            self._sim_board = GoBoard(node.board.size)
        simulation_board = self._sim_board
        simulation_board.copy_state_from(node.board)
        starting_player = node.player
        avoided_moves = avoided_moves or set()
        
        # cache last move for filtering
        last_move = None
        if simulation_board.move_history:
            lm = simulation_board.move_history[-1]
            if lm[0] >= 0:
                last_move = (lm[0], lm[1])
        frontier = self._build_frontier(simulation_board, last_move)
        
        max_moves = 25
        moves_made = 0
        consecutive_passes = 0
        size = simulation_board.size
        
        while consecutive_passes < 2 and moves_made < max_moves:
            legal_moves = self._filter_moves(simulation_board, last_move, frontier)
            
            if avoided_moves and legal_moves:
                filtered = [m for m in legal_moves if m not in avoided_moves]
                if filtered:
                    legal_moves = filtered
            
            if not legal_moves:
                simulation_board.pass_turn_fast()
                consecutive_passes += 1
                last_move = None
            else:
                move = self._pick_rollout_move(simulation_board, legal_moves)
                if move:
                    row, col = move
                    simulation_board.make_move_fast(row, col)
                    consecutive_passes = 0
                    last_move = move
                    self._update_frontier(frontier, move, size)
                else:
                    simulation_board.pass_turn_fast()
                    consecutive_passes += 1
                    last_move = None
            
            moves_made += 1
            
            # Early termination on large capture difference
            if moves_made % 10 == 0:
                cap_diff = simulation_board.captured_stones[starting_player] - simulation_board.captured_stones[
                    Player.WHITE if starting_player == Player.BLACK else Player.BLACK
                ]
                if cap_diff > 8 or cap_diff < -8:
                    break
        
        return self._fast_evaluate(simulation_board, starting_player)

    def _add_radius(self, frontier: Set[Tuple[int, int]], row: int, col: int, size: int, radius: int):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = row + dr, col + dc
                if 0 <= r < size and 0 <= c < size:
                    frontier.add((r, c))

    def _add_influence_points(self, frontier: Set[Tuple[int, int]], row: int, col: int, size: int):
        for adj_r, adj_c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= adj_r < size and 0 <= adj_c < size:
                frontier.add((adj_r, adj_c))
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < size and 0 <= c < size:
                frontier.add((r, c))

    def _update_frontier(self, frontier: Set[Tuple[int, int]], move: Optional[Tuple[int, int]], size: int):
        if move is None:
            return
        row, col = move
        frontier.discard((row, col))
        self._add_influence_points(frontier, row, col, size)
        self._add_radius(frontier, row, col, size, 2)

    def _build_frontier(self, board: GoBoard, last_move: Optional[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        size = board.size
        frontier: Set[Tuple[int, int]] = set()
        board_data = board.board
        for row in range(size):
            row_data = board_data[row]
            for col in range(size):
                if row_data[col] != Player.EMPTY:
                    self._add_influence_points(frontier, row, col, size)
        if last_move:
            lr, lc = last_move
            self._add_radius(frontier, lr, lc, size, 2)
        if len(board.move_history) < 20:
            center = size // 2
            frontier.add((center, center))
            if size >= 9:
                for sp in [(2, 2), (2, 6), (6, 2), (6, 6), (2, center), (6, center), (center, 2), (center, 6)]:
                    if sp[0] < size and sp[1] < size:
                        frontier.add(sp)
        return frontier
    
    def _filter_moves(
        self,
        board: GoBoard,
        last_move: Optional[Tuple[int, int]],
        frontier: Optional[Set[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int]]:
        """Ultra-fast move filtering - inlined for speed"""
        size = board.size
        board_data = board.board
        group_id = board._group_id
        group_libs = board._group_liberties
        current_player = board.current_player
        opponent = Player.WHITE if current_player == Player.BLACK else Player.BLACK
        
        candidates = frontier if frontier is not None else set()
        if not candidates and last_move:
            candidates = set()
            lr, lc = last_move
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = lr + dr, lc + dc
                    if 0 <= r < size and 0 <= c < size:
                        candidates.add((r, c))
        
        # Inlined legality check for speed
        legal = []
        for r, c in candidates:
            if board_data[r][c] != Player.EMPTY:
                continue
            
            # Inlined is_legal_move_fast
            is_legal = False
            for adj_r, adj_c in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if adj_r < 0 or adj_r >= size or adj_c < 0 or adj_c >= size:
                    continue
                stone = board_data[adj_r][adj_c]
                if stone == Player.EMPTY:
                    is_legal = True
                    break
                elif stone == opponent:
                    gid = group_id[adj_r][adj_c]
                    if gid > 0 and gid in group_libs and len(group_libs[gid]) == 1:
                        is_legal = True
                        break
                elif stone == current_player:
                    gid = group_id[adj_r][adj_c]
                    if gid > 0 and gid in group_libs and len(group_libs[gid]) > 1:
                        is_legal = True
                        break
            
            if is_legal:
                legal.append((r, c))
        
        # Minimal fallback - just pick random empties without full validation
        if len(legal) < 3:
            for r in range(size):
                row_data = board_data[r]
                for c in range(size):
                    if row_data[c] == Player.EMPTY and (r, c) not in candidates:
                        # Quick check: has adjacent empty = legal
                        for adj_r, adj_c in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                            if 0 <= adj_r < size and 0 <= adj_c < size:
                                if board_data[adj_r][adj_c] == Player.EMPTY:
                                    legal.append((r, c))
                                    break
                        if len(legal) >= 5:
                            return legal
        
        return legal
    
    def _pick_rollout_move(self, board: GoBoard, legal_moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Ultra-fast move selection for rollouts."""
        n = len(legal_moves)
        if n == 0:
            return None
        if n <= 3:
            return legal_moves[int(random.random() * n)]
        if n > 20:
            return legal_moves[int(random.random() * n)]
        
        current_player = board.current_player
        opponent = Player.WHITE if current_player == Player.BLACK else Player.BLACK
        size = board.size
        center = size // 2
        size_m1 = size - 1
        board_data = board.board
        group_id = board._group_id
        group_libs = board._group_liberties
        
        last_row, last_col = -1, -1
        if board.move_history:
            lm = board.move_history[-1]
            last_row, last_col = lm[0], lm[1]
        
        scores = []
        for row, col in legal_moves:
            score = 1.0
            
            # Edge penalty
            if row == 0 or row == size_m1 or col == 0 or col == size_m1:
                score = 0.25
            elif row == 1 or row == size_m1 - 1 or col == 1 or col == size_m1 - 1:
                score = 0.65
            
            # Center bonus
            score += max(0, size - abs(row - center) - abs(col - center)) * 0.14
            
            # Neighbor analysis (inlined)
            enemy_in_atari = False
            friendly_in_atari = False
            empty_adj = 0
            for adj_r, adj_c in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
                if adj_r < 0 or adj_r >= size or adj_c < 0 or adj_c >= size:
                    continue
                stone = board_data[adj_r][adj_c]
                if stone == current_player:
                    score += 0.25
                    gid = group_id[adj_r][adj_c]
                    if gid > 0 and gid in group_libs and len(group_libs[gid]) == 1:
                        friendly_in_atari = True
                elif stone == opponent:
                    gid = group_id[adj_r][adj_c]
                    if gid > 0 and gid in group_libs and len(group_libs[gid]) == 1:
                        enemy_in_atari = True
                    else:
                        score += 0.3
                else:
                    empty_adj += 1
            
            if enemy_in_atari:
                score += 2.5
            if friendly_in_atari:
                score += 1.2
            if empty_adj <= 1 and not enemy_in_atari and not friendly_in_atari:
                score *= 0.35
            
            # Follow last move
            if last_row >= 0:
                dist = abs(row - last_row) + abs(col - last_col)
                if dist <= 2:
                    score += 0.5
                elif dist <= 4:
                    score += 0.2
            
            scores.append(score if score > 0.01 else 0.01)
        
        # Weighted random pick
        total = 0.0
        for s in scores:
            total += s
        pick = random.random() * total
        cumulative = 0.0
        for i in range(n):
            cumulative += scores[i]
            if pick <= cumulative:
                return legal_moves[i]
        return legal_moves[-1]
    
    def _fast_evaluate(self, board: GoBoard, player: Player) -> float:
        """Single-pass evaluation: stones + influence + territory in one loop"""
        size = board.size
        center = size // 2
        board_data = board.board  # direct access
        
        black_score = 0.0
        white_score = 0.0
        
        # single pass over all points
        for row in range(size):
            row_data = board_data[row]
            for col in range(size):
                stone = row_data[col]
                dist = abs(row - center) + abs(col - center)
                influence_mult = max(0, (size - dist)) * 0.08
                
                if stone == Player.BLACK:
                    black_score += 1.0 + influence_mult
                elif stone == Player.WHITE:
                    white_score += 1.0 + influence_mult
                else:
                    # empty: quick territory estimate
                    black_adj = 0
                    white_adj = 0
                    # inline adjacent check
                    if row > 0:
                        s = board_data[row-1][col]
                        if s == Player.BLACK: black_adj += 1
                        elif s == Player.WHITE: white_adj += 1
                    if row < size - 1:
                        s = board_data[row+1][col]
                        if s == Player.BLACK: black_adj += 1
                        elif s == Player.WHITE: white_adj += 1
                    if col > 0:
                        s = row_data[col-1]
                        if s == Player.BLACK: black_adj += 1
                        elif s == Player.WHITE: white_adj += 1
                    if col < size - 1:
                        s = row_data[col+1]
                        if s == Player.BLACK: black_adj += 1
                        elif s == Player.WHITE: white_adj += 1
                    
                    if black_adj > 0 and white_adj == 0:
                        black_score += 0.25 * black_adj
                    elif white_adj > 0 and black_adj == 0:
                        white_score += 0.25 * white_adj
        
        # add captures
        black_score += board.captured_stones[Player.BLACK]
        white_score += board.captured_stones[Player.WHITE] + 6.5  # komi
        
        # normalize
        diff = black_score - white_score
        if player == Player.WHITE:
            diff = -diff
        
        return 0.5 + max(-0.5, min(0.5, diff / 30.0))
    
    def _evaluate_position(self, board: GoBoard, player: Player) -> float:
        return self._fast_evaluate(board, player)
    
    def _backpropagate(self, node: MCTSNode, result: float):
        while node is not None:
            if node.player == node.board.current_player:
                node.update(result)
            else:
                node.update(1.0 - result)
            
            node = node.parent


class RandomAgent:
    
    def select_move(self, board: GoBoard) -> Optional[Tuple[int, int]]:
        
        legal_moves = board.get_legal_moves()
        
        if not legal_moves:
            return None
        
        # 90% chance to make a move, 10% chance to pass
        if random.random() < 0.9:
            return random.choice(legal_moves)
        else:
            return None
