"""
Parallel rollout worker for MCTS simulations.
This module provides standalone functions that can be pickled and run in separate processes.
"""
import random
from typing import List, Tuple, Optional, Set
try:
    from product.go_engine import GoBoard, Player
except ModuleNotFoundError:
    from go_engine import GoBoard, Player


def _build_frontier_fast(board_data: List[List[Player]], size: int, last_move: Optional[Tuple[int, int]], move_count: int) -> Set[Tuple[int, int]]:
    """Build frontier set for move filtering."""
    frontier: Set[Tuple[int, int]] = set()
    
    # Add influence points around existing stones
    for row in range(size):
        row_data = board_data[row]
        for col in range(size):
            if row_data[col] != Player.EMPTY:
                # Add adjacent points
                for adj_r, adj_c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                    if 0 <= adj_r < size and 0 <= adj_c < size:
                        frontier.add((adj_r, adj_c))
                # Add diagonal and knight moves
                for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    r, c = row + dr, col + dc
                    if 0 <= r < size and 0 <= c < size:
                        frontier.add((r, c))
    
    # Add radius around last move
    if last_move:
        lr, lc = last_move
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = lr + dr, lc + dc
                if 0 <= r < size and 0 <= c < size:
                    frontier.add((r, c))
    
    # Add star points in early game
    if move_count < 20:
        center = size // 2
        frontier.add((center, center))
        if size >= 9:
            for sp in [(2, 2), (2, 6), (6, 2), (6, 6), (2, center), (6, center), (center, 2), (center, 6)]:
                if sp[0] < size and sp[1] < size:
                    frontier.add(sp)
    
    return frontier


def _filter_moves_fast(board: GoBoard, frontier: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Ultra-fast move filtering using frontier."""
    size = board.size
    board_data = board.board
    legal = []
    
    for r, c in frontier:
        if board_data[r][c] == Player.EMPTY and board.is_legal_move_fast(r, c):
            legal.append((r, c))
    
    # If too few, sample random empty points
    if len(legal) < 5:
        empty_points = []
        for r in range(size):
            row_data = board_data[r]
            for c in range(size):
                if row_data[c] == Player.EMPTY and (r, c) not in frontier:
                    empty_points.append((r, c))
        
        if empty_points:
            sample_size = min(8, len(empty_points))
            sampled = random.sample(empty_points, sample_size)
            for r, c in sampled:
                if board.is_legal_move_fast(r, c):
                    legal.append((r, c))
                    if len(legal) >= 6:
                        break
    
    return legal


def _update_frontier_fast(frontier: Set[Tuple[int, int]], move: Tuple[int, int], size: int):
    """Update frontier after a move."""
    row, col = move
    frontier.discard((row, col))
    # Add neighbors
    for adj_r, adj_c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
        if 0 <= adj_r < size and 0 <= adj_c < size:
            frontier.add((adj_r, adj_c))
    # Add radius 2
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r, c = row + dr, col + dc
            if 0 <= r < size and 0 <= c < size:
                frontier.add((r, c))


def _select_move_heuristic(
    board: GoBoard,
    legal_moves: List[Tuple[int, int]],
    last_move: Optional[Tuple[int, int]]
) -> Tuple[int, int]:
    """Lightweight heuristic selection for rollouts."""
    if len(legal_moves) == 1:
        return legal_moves[0]

    size = board.size
    center = size // 2
    size_minus_1 = size - 1
    board_data = board.board
    current_player = board.current_player
    opponent = Player.WHITE if current_player == Player.BLACK else Player.BLACK
    group_id = board._group_id
    group_libs = board._group_liberties

    scores = []
    last_row, last_col = (-1, -1) if last_move is None else last_move

    for row, col in legal_moves:
        score = 1.0

        # Strong edge/corner penalty (outer 2 rings)
        if row == 0 or row == size_minus_1 or col == 0 or col == size_minus_1:
            score *= 0.15  # very strong penalty for first line
        elif row == 1 or row == size_minus_1 - 1 or col == 1 or col == size_minus_1 - 1:
            score *= 0.5   # moderate penalty for second line

        # center bonus (stronger)
        dist_to_center = abs(row - center) + abs(col - center)
        score += max(0, (size - dist_to_center)) * 0.2

        friendly = 0
        enemy = 0
        enemy_in_atari = False

        for adj_r, adj_c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= adj_r < size and 0 <= adj_c < size:
                stone = board_data[adj_r][adj_c]
                if stone == current_player:
                    friendly += 1
                elif stone == opponent:
                    enemy += 1
                    gid = group_id[adj_r][adj_c]
                    if gid > 0 and gid in group_libs and len(group_libs[gid]) == 1:
                        enemy_in_atari = True

        score += friendly * 0.25
        if enemy_in_atari:
            score += 2.5
        elif enemy > 0:
            score += 0.3

        if last_row >= 0:
            dist = abs(row - last_row) + abs(col - last_col)
            if dist <= 2:
                score += 0.5
            elif dist <= 4:
                score += 0.2

        scores.append(max(score, 0.01))

    total = sum(scores)
    pick = random.random() * total
    cumulative = 0.0
    for i, s in enumerate(scores):
        cumulative += s
        if pick <= cumulative:
            return legal_moves[i]

    return legal_moves[-1]


def _fast_evaluate_standalone(board: GoBoard, player: Player) -> float:
    """Standalone fast evaluation."""
    size = board.size
    center = size // 2
    board_data = board.board
    
    black_score = 0.0
    white_score = 0.0
    
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
                black_adj = 0
                white_adj = 0
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
    
    try:
        black_score += board.captured_stones.get(Player.BLACK, 0)
        white_score += board.captured_stones.get(Player.WHITE, 0) + 6.5
    except (KeyError, TypeError):
        # Fallback: enum keys may not survive cross-process pickling
        white_score += 6.5
    
    diff = black_score - white_score
    if player == Player.WHITE:
        diff = -diff
    
    return 0.5 + max(-0.5, min(0.5, diff / 30.0))


def run_single_rollout(board_state: dict, starting_player: Player) -> float:
    """
    Run a single rollout from given board state.
    This function is designed to be called from a worker process.
    
    Args:
        board_state: Serialized board state dict
        starting_player: Player who started the simulation
    
    Returns:
        Evaluation result as float [0, 1]
    """
    # Reconstruct board from state
    board = GoBoard(board_state['size'])
    board.board = [row[:] for row in board_state['board']]
    board.current_player = board_state['current_player']
    board.move_history = board_state['move_history'][:]
    # Rebuild captured_stones with proper Player enum keys
    raw_caps = board_state['captured_stones']
    board.captured_stones = {}
    for k, v in raw_caps.items():
        if isinstance(k, Player):
            board.captured_stones[k] = v
        elif k == 1 or k == Player.BLACK.value:
            board.captured_stones[Player.BLACK] = v
        elif k == 2 or k == Player.WHITE.value:
            board.captured_stones[Player.WHITE] = v
    # Ensure both keys exist
    board.captured_stones.setdefault(Player.BLACK, 0)
    board.captured_stones.setdefault(Player.WHITE, 0)
    board.ko_point = board_state['ko_point']
    board.pass_count = board_state['pass_count']
    board._group_id = [row[:] for row in board_state['group_id']]
    board._group_liberties = {k: v.copy() for k, v in board_state['group_liberties'].items()}
    board._group_stones = {k: v.copy() for k, v in board_state['group_stones'].items()}
    board._next_group_id = board_state['next_group_id']
    
    # Build initial frontier
    last_move = None
    if board.move_history:
        lm = board.move_history[-1]
        if lm[0] >= 0:
            last_move = (lm[0], lm[1])
    
    frontier = _build_frontier_fast(board.board, board.size, last_move, len(board.move_history))
    
    # Run rollout
    max_moves = 25
    moves_made = 0
    consecutive_passes = 0
    size = board.size
    
    while consecutive_passes < 2 and moves_made < max_moves:
        legal_moves = _filter_moves_fast(board, frontier)
        
        if not legal_moves:
            board.pass_turn_fast()
            consecutive_passes += 1
            last_move = None
        else:
            move = _select_move_heuristic(board, legal_moves, last_move)
            row, col = move
            board.make_move_fast(row, col)
            consecutive_passes = 0
            last_move = move
            _update_frontier_fast(frontier, move, size)
        
        moves_made += 1
        
        # Early termination based on captures
        if moves_made % 10 == 0:
            opponent = Player.WHITE if starting_player == Player.BLACK else Player.BLACK
            cap_diff = board.captured_stones[starting_player] - board.captured_stones[opponent]
            if cap_diff > 8 or cap_diff < -8:
                break
    
    return _fast_evaluate_standalone(board, starting_player)


def run_batch_rollouts(board_state: dict, starting_player: Player, count: int) -> List[float]:
    """
    Run multiple rollouts from the same board state.
    
    Args:
        board_state: Serialized board state dict
        starting_player: Player who started the simulation
        count: Number of rollouts to run
    
    Returns:
        List of evaluation results
    """
    results = []
    for _ in range(count):
        results.append(run_single_rollout(board_state, starting_player))
    return results


def serialize_board_state(board: GoBoard) -> dict:
    """Serialize a GoBoard to a dict that can be pickled."""
    return {
        'size': board.size,
        'board': [row[:] for row in board.board],
        'current_player': board.current_player,
        'move_history': board.move_history[:],
        'captured_stones': board.captured_stones.copy(),
        'ko_point': board.ko_point,
        'pass_count': board.pass_count,
        'group_id': [row[:] for row in board._group_id],
        'group_liberties': {k: v.copy() for k, v in board._group_liberties.items()},
        'group_stones': {k: v.copy() for k, v in board._group_stones.items()},
        'next_group_id': board._next_group_id,
    }
