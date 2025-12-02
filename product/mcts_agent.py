import math
import random
import time
from typing import Optional, List, Tuple
from go_engine import GoBoard, Player


class MCTSNode:
    """
    A node in the MCTS tree.
    Each node represents a game state and stores statistics for move selection.
    """
    
    def __init__(self, board: GoBoard, move: Optional[Tuple[int, int]] = None, 
                 parent: Optional['MCTSNode'] = None):
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
        self.untried_moves = board.get_legal_moves()
        # If no legal moves exist, the only option is to pass
        if not self.untried_moves:
            self.untried_moves = [None]
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
            new_board.pass_turn()
        else:
            # Regular move
            row, col = move
            new_board.make_move(row, col)
        
        child_node = MCTSNode(new_board, move, self)
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
    
    def select_move(self, board: GoBoard) -> Optional[Tuple[int, int]]:
        # === TREE REUSE: try to find subtree from previous search ===
        root = self._try_reuse_tree(board)
        if root is None:
            root = MCTSNode(board.clone())
        
        # Run simulations
        start_time = time.time()
        simulations = 0
        reused_visits = root.visits  # track how many we got for free
        
        while True:
            # Check termination conditions
            if self.max_simulations and simulations >= self.max_simulations:
                break
            if not self.max_simulations and (time.time() - start_time) >= self.simulation_time:
                break
            
            # === conditioning on tree statistics ===
            # find moves with high visits but low win rate to avoid in simulations
            avoided_moves = set()
            for child in root.children:
                if child.visits >= 5 and child.move is not None:
                    win_rate = child.wins / child.visits
                    if win_rate < 0.3:  # poorly performing move
                        avoided_moves.add(child.move)
            
            # MCTS four phases
            node = self._select(root)
            if not node.is_terminal():
                node = self._expand(node)
            result, heuristics_used = self._simulate(node, avoided_moves)
            self._backpropagate(node, result)
            
            # === adapt heuristic weights based on simulation outcome ===
            self._update_heuristic_weights(heuristics_used, result)
            
            simulations += 1
        
        # Select the move with the most visits (most robust choice)
        if not root.children:
            # No legal moves, must pass
            return None
        
        best_child = max(root.children, key=lambda c: c.visits)
        
        # === TREE REUSE: cache the chosen subtree for next turn ===
        self._cache_subtree(best_child)
        
        reused = f" (+{reused_visits} reused)" if reused_visits > 0 else ""
        print(f"MCTS completed {simulations} simulations{reused} in {time.time() - start_time:.2f}s")
        print(f"Selected move: {best_child.move}, Visits: {best_child.visits}, "
              f"Win rate: {best_child.wins / best_child.visits if best_child.visits > 0 else 0:.2%}")
        
        return best_child.move
    
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
    
    def _simulate(self, node: MCTSNode, avoided_moves: set = None) -> Tuple[float, List[str]]:
        """run a simulation and return (result, list of heuristics that influenced moves)"""
        simulation_board = node.board.clone()
        starting_player = node.player
        avoided_moves = avoided_moves or set()
        
        # track which heuristics contributed to moves this simulation
        heuristics_used = []
        # track patterns used so we can update them after seeing the result
        patterns_used = []
        
        # cache last move for filtering
        last_move = None
        if simulation_board.move_history:
            lm = simulation_board.move_history[-1]
            if lm[0] >= 0:
                last_move = (lm[0], lm[1])
        
        # Play moves until game is over or max moves
        max_moves = 25  # reduced for speed
        moves_made = 0
        consecutive_passes = 0
        
        while consecutive_passes < 2 and moves_made < max_moves:
            # === MOVE FILTERING: only consider relevant moves ===
            legal_moves = self._filter_moves(simulation_board, last_move)
            
            # filter out avoided moves (from tree statistics)
            if avoided_moves and legal_moves:
                filtered = [m for m in legal_moves if m not in avoided_moves]
                if filtered:
                    legal_moves = filtered
            
            if not legal_moves:
                simulation_board.pass_turn()
                consecutive_passes += 1
                last_move = None
            else:
                # Choose a move using adaptive heuristics
                move, h_used, pattern_key = self._select_simulation_move_fast(simulation_board, legal_moves)
                heuristics_used.extend(h_used)
                if pattern_key:
                    patterns_used.append(pattern_key)
                if move:
                    row, col = move
                    simulation_board.make_move(row, col)
                    consecutive_passes = 0
                    last_move = move
                else:
                    simulation_board.pass_turn()
                    consecutive_passes += 1
                    last_move = None
            
            moves_made += 1
            
            # === EARLY TERMINATION: stop if position is clearly decided ===
            if moves_made % 8 == 0:  # check every 8 moves
                quick_eval = self._fast_evaluate(simulation_board, starting_player)
                if quick_eval > 0.85 or quick_eval < 0.15:
                    break  # clear winner, stop early
        
        # Fast evaluation
        result = self._fast_evaluate(simulation_board, starting_player)
        
        # update pattern table
        for pattern_key in patterns_used:
            if pattern_key not in self.pattern_table:
                self.pattern_table[pattern_key] = {'wins': 0.0, 'count': 0}
            self.pattern_table[pattern_key]['wins'] += result
            self.pattern_table[pattern_key]['count'] += 1
        
        return result, heuristics_used
    
    def _filter_moves(self, board: GoBoard, last_move: Optional[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Filter legal moves to relevant ones for faster simulation"""
        size = board.size
        center = size // 2
        
        # collect candidate moves
        candidates = set()
        
        # 1) moves near last move (local response) - most important
        if last_move:
            lr, lc = last_move
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    r, c = lr + dr, lc + dc
                    if 0 <= r < size and 0 <= c < size:
                        candidates.add((r, c))
        
        # 2) moves adjacent to existing stones (expand/reduce groups)
        for row in range(size):
            for col in range(size):
                if board.board[row][col] != Player.EMPTY:
                    for adj_r, adj_c in board.get_adjacent_points_fast(row, col):
                        candidates.add((adj_r, adj_c))
                    # also one step further for influence
                    for dr, dc in [(-2,0), (2,0), (0,-2), (0,2), (-1,-1), (-1,1), (1,-1), (1,1)]:
                        r, c = row + dr, col + dc
                        if 0 <= r < size and 0 <= c < size:
                            candidates.add((r, c))
        
        # 3) star points / center for early game (handle different board sizes)
        if len(board.move_history) < 20:
            # add center and star points that fit on board
            candidates.add((center, center))
            if size >= 9:
                for sp in [(2,2), (2,6), (6,2), (6,6), (2, center), (6, center), (center, 2), (center, 6)]:
                    if sp[0] < size and sp[1] < size:
                        candidates.add(sp)
        
        # filter to legal moves only
        legal = []
        for r, c in candidates:
            if 0 <= r < size and 0 <= c < size:
                if board.board[r][c] == Player.EMPTY and board.is_legal_move(r, c):
                    legal.append((r, c))
        
        # fallback: if too few candidates, use all legal moves
        if len(legal) < 5:
            return board.get_legal_moves()
        
        return legal
    
    def _select_simulation_move_fast(self, board: GoBoard, legal_moves: List[Tuple[int, int]]) -> Tuple[Optional[Tuple[int, int]], List[str], Optional[tuple]]:
        """Fast move selection for simulations - optimized version"""
        if not legal_moves:
            return None, [], None
        
        # for very small move lists, just pick randomly with basic heuristics
        if len(legal_moves) <= 3:
            move = random.choice(legal_moves)
            return move, [], None
        
        current_player = board.current_player
        opponent = board.get_opponent(current_player)
        size = board.size
        center = size // 2
        size_minus_1 = size - 1
        
        scores = []
        best_pattern_key = None
        dominant_h = []
        
        # get last move once
        last_row, last_col = -1, -1
        if board.move_history:
            lm = board.move_history[-1]
            last_row, last_col = lm[0], lm[1]
        
        for row, col in legal_moves:
            score = 1.0
            
            # edge penalty (inlined)
            if row == 0 or row == size_minus_1 or col == 0 or col == size_minus_1:
                score *= 0.4
            
            # center bonus (simplified)
            dist_to_center = abs(row - center) + abs(col - center)
            score += max(0, (size - dist_to_center)) * 0.12
            
            # neighbor analysis (single pass)
            friendly = 0
            enemy = 0
            enemy_in_atari = False
            
            for adj_r, adj_c in board.get_adjacent_points_fast(row, col):
                stone = board.board[adj_r][adj_c]
                if stone == current_player:
                    friendly += 1
                elif stone == opponent:
                    enemy += 1
                    # check if in atari using cache
                    gid = board._group_id[adj_r][adj_c]
                    if gid > 0 and gid in board._group_liberties:
                        if len(board._group_liberties[gid]) == 1:
                            enemy_in_atari = True
            
            # connection bonus
            score += friendly * 0.25
            
            # capture bonus
            if enemy_in_atari:
                score += 2.5
            elif enemy > 0:
                score += 0.3
            
            # follow last move
            if last_row >= 0:
                dist = abs(row - last_row) + abs(col - last_col)
                if dist <= 2:
                    score += 0.5
                elif dist <= 4:
                    score += 0.2
            
            scores.append(max(score, 0.01))
        
        # weighted selection
        total = sum(scores)
        pick = random.random() * total
        cumulative = 0.0
        for i, s in enumerate(scores):
            cumulative += s
            if pick <= cumulative:
                return legal_moves[i], dominant_h, best_pattern_key
        
        return legal_moves[-1], dominant_h, best_pattern_key
    
    def _select_simulation_move(self, board: GoBoard, legal_moves: List[Tuple[int, int]]) -> Tuple[Optional[Tuple[int, int]], List[str], Optional[tuple]]:
        """Full move selection with all heuristics - used for expansion"""
        return self._select_simulation_move_fast(board, legal_moves)
    
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
