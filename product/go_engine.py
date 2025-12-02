import copy
from typing import List, Tuple, Set, Optional
from enum import Enum


class Player(Enum):
    """Represents the two players in Go."""
    BLACK = 1
    WHITE = 2
    EMPTY = 0


def column_to_label(index: int) -> str:
    """Convert column index (0-based) to board letter, skipping 'I'."""
    if index < 0:
        raise ValueError("Column index must be non-negative")
    base = ord('A')
    label_ord = base + index
    if label_ord >= ord('I'):
        label_ord += 1  # skip the letter I
    return chr(label_ord)


def label_to_column(label: str) -> int:
    """Convert board letter to column index (0-based), skipping 'I'."""
    if not label or not label.isalpha():
        raise ValueError("Invalid column label")
    letter = label.upper()
    if letter == 'I':
        raise ValueError("Column I is skipped in Go coordinates")
    offset = ord(letter) - ord('A')
    if ord(letter) > ord('I'):
        offset -= 1
    if offset < 0:
        raise ValueError("Invalid column label")
    return offset


class GoBoard:
    """
    Represents a 9x9 Go board with complete game logic.
    Includes liberty caching for performance optimization.
    """
    
    def __init__(self, size: int = 9):
        self.size = size
        self.board = [[Player.EMPTY for _ in range(size)] for _ in range(size)]
        self.current_player = Player.BLACK
        self.move_history = []
        self.captured_stones = {Player.BLACK: 0, Player.WHITE: 0}
        self.ko_point = None  # Tracks ko position
        self.pass_count = 0
        
        # undo/redo stacks - store full board snapshots
        self._undo_stack = []
        self._redo_stack = []
        
        # === LIBERTY CACHING ===
        # group_id[row][col] = unique ID for the group at that position (0 if empty)
        self._group_id = [[0] * size for _ in range(size)]
        # group_liberties[group_id] = set of liberty positions for that group
        self._group_liberties = {}
        # group_stones[group_id] = set of stone positions in that group
        self._group_stones = {}
        self._next_group_id = 1
        
        # precompute adjacent offsets for speed
        self._adj_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
    def get_opponent(self, player: Player) -> Player:
        if player == Player.BLACK:
            return Player.WHITE
        elif player == Player.WHITE:
            return Player.BLACK
        return Player.EMPTY
    
    def is_on_board(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size
    
    def get_stone(self, row: int, col: int) -> Player:
        if not self.is_on_board(row, col):
            return None
        return self.board[row][col]
    
    def set_stone(self, row: int, col: int, player: Player):
        if self.is_on_board(row, col):
            self.board[row][col] = player
    
    def get_adjacent_points(self, row: int, col: int) -> List[Tuple[int, int]]:
        # inlined bounds check for speed
        adjacent = []
        if row > 0:
            adjacent.append((row - 1, col))
        if row < self.size - 1:
            adjacent.append((row + 1, col))
        if col > 0:
            adjacent.append((row, col - 1))
        if col < self.size - 1:
            adjacent.append((row, col + 1))
        return adjacent
    
    def get_adjacent_points_fast(self, row: int, col: int):
        """Generator version - avoids list allocation"""
        if row > 0:
            yield (row - 1, col)
        if row < self.size - 1:
            yield (row + 1, col)
        if col > 0:
            yield (row, col - 1)
        if col < self.size - 1:
            yield (row, col + 1)
    
    def get_group(self, row: int, col: int) -> Set[Tuple[int, int]]:
        # collect the connected stones that share the same color as (row, col)
        stone_color = self.get_stone(row, col)
        if stone_color == Player.EMPTY:
            return set()
        
        group = set()
        to_check = [(row, col)]
        
        while to_check:
            current = to_check.pop()
            if current in group:
                continue
            
            r, c = current
            if self.get_stone(r, c) == stone_color:
                group.add(current)
                for adjacent in self.get_adjacent_points(r, c):
                    if adjacent not in group:
                        to_check.append(adjacent)
        
        return group
    
    def count_liberties(self, row: int, col: int) -> int:
        """Count liberties using cache if available, else compute"""
        gid = self._group_id[row][col]
        if gid > 0 and gid in self._group_liberties:
            return len(self._group_liberties[gid])
        
        # fallback to computation (for temp stones not in cache)
        group = self.get_group(row, col)
        liberties = set()
        for r, c in group:
            for adj_r, adj_c in self.get_adjacent_points_fast(r, c):
                if self.board[adj_r][adj_c] == Player.EMPTY:
                    liberties.add((adj_r, adj_c))
        return len(liberties)
    
    def count_liberties_fast(self, row: int, col: int) -> int:
        """Ultra-fast liberty count using only cache - returns 0 if not cached"""
        gid = self._group_id[row][col]
        if gid > 0 and gid in self._group_liberties:
            return len(self._group_liberties[gid])
        return 0
    
    def get_liberties(self, row: int, col: int) -> Set[Tuple[int, int]]:
        """Get liberties using cache if available"""
        gid = self._group_id[row][col]
        if gid > 0 and gid in self._group_liberties:
            return self._group_liberties[gid].copy()
        
        # fallback
        group = self.get_group(row, col)
        liberties = set()
        for r, c in group:
            for adj_r, adj_c in self.get_adjacent_points_fast(r, c):
                if self.board[adj_r][adj_c] == Player.EMPTY:
                    liberties.add((adj_r, adj_c))
        return liberties
    
    def remove_group(self, row: int, col: int) -> int:
        """Remove a group and update liberty cache"""
        gid = self._group_id[row][col]
        if gid > 0 and gid in self._group_stones:
            group = self._group_stones[gid]
        else:
            group = self.get_group(row, col)
        
        removed_positions = list(group)
        
        # remove from board and clear cache
        for r, c in removed_positions:
            self.board[r][c] = Player.EMPTY
            self._group_id[r][c] = 0
        
        # clean up group data
        if gid > 0:
            self._group_liberties.pop(gid, None)
            self._group_stones.pop(gid, None)
        
        # update liberties of adjacent groups (they gain liberties)
        for r, c in removed_positions:
            for adj_r, adj_c in self.get_adjacent_points_fast(r, c):
                adj_gid = self._group_id[adj_r][adj_c]
                if adj_gid > 0 and adj_gid in self._group_liberties:
                    self._group_liberties[adj_gid].add((r, c))
        
        return len(removed_positions)
    
    def capture_adjacent_groups(self, row: int, col: int, player: Player) -> List[Tuple[int, int]]:
        # After a move, look at every neighboring opponent group and remove it if it has no liberties
        opponent = self.get_opponent(player)
        captured = []
        
        for adj_r, adj_c in self.get_adjacent_points(row, col):
            if self.get_stone(adj_r, adj_c) == opponent:
                if self.count_liberties(adj_r, adj_c) == 0:
                    group = self.get_group(adj_r, adj_c)
                    captured.extend(list(group))
                    num_captured = self.remove_group(adj_r, adj_c)
                    self.captured_stones[player] += num_captured
        
        return captured
    
    def is_suicide_move(self, row: int, col: int, player: Player) -> bool:
        # Temporarily drop the stone to see whether it would instantly die without capturing
        self.set_stone(row, col, player)
        
        opponent = self.get_opponent(player)
        captures_opponent = False
        for adj_r, adj_c in self.get_adjacent_points(row, col):
            if self.get_stone(adj_r, adj_c) == opponent:
                if self.count_liberties(adj_r, adj_c) == 0:
                    captures_opponent = True
                    break
        
        # Check if the placed stone's group has liberties
        has_liberties = self.count_liberties(row, col) > 0
        
        # Remove the temporary stone
        self.set_stone(row, col, Player.EMPTY)
        
        return not has_liberties and not captures_opponent
    
    def violates_ko(self, row: int, col: int) -> bool:
        return self.ko_point is not None and (row, col) == self.ko_point
    
    def is_legal_move(self, row: int, col: int, player: Player = None) -> bool:
        if player is None:
            player = self.current_player
        
        if not self.is_on_board(row, col):
            return False
        
        # Check if position is empty
        if self.get_stone(row, col) != Player.EMPTY:
            return False
        
        # Check ko rule
        if self.violates_ko(row, col):
            return False
        
        # Check suicide rule
        if self.is_suicide_move(row, col, player):
            return False
        
        return True
    
    def make_move(self, row: int, col: int, player: Player = None) -> bool:
        if player is None:
            player = self.current_player
        
        if not self.is_legal_move(row, col, player):
            return False
        
        # save snapshot for undo before we do anything
        self._save_state_for_undo()
        self._redo_stack.clear()  # clear redo when new move is made
        
        # Store board state for ko detection
        previous_board = copy.deepcopy(self.board)
        
        # Place the stone and update cache
        self.board[row][col] = player
        self._place_stone_update_cache(row, col, player)
        
        # Capture opponent groups
        captured = self.capture_adjacent_groups(row, col, player)
        
        # Check for ko
        # Ko occurs when a single stone is captured and the board returns to previous state
        self.ko_point = None
        if len(captured) == 1:
            # Check if recapturing would return to previous board state
            cap_r, cap_c = captured[0]
            if previous_board == [[Player.EMPTY if c == player and r == row and col == col 
                                   else previous_board[r][c] 
                                   for c in range(self.size)] 
                                  for r in range(self.size)]:
                self.ko_point = captured[0]
        
        # Record move in history
        self.move_history.append((row, col, player, len(captured)))
        
        # Switch players
        self.current_player = self.get_opponent(self.current_player)
        self.pass_count = 0
        
        return True
    
    def pass_turn(self):
        # save state before passing
        self._save_state_for_undo()
        self._redo_stack.clear()
        
        self.move_history.append((-1, -1, self.current_player, 0))
        self.current_player = self.get_opponent(self.current_player)
        self.pass_count += 1
    
    def is_game_over(self) -> bool:
        return self.pass_count >= 2
    
    def get_legal_moves(self, player: Player = None) -> List[Tuple[int, int]]:
        if player is None:
            player = self.current_player
        
        legal_moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.is_legal_move(row, col, player):
                    legal_moves.append((row, col))
        
        return legal_moves
    
    def calculate_score(self, komi: float = 6.5) -> dict:
        visited = [[False for _ in range(self.size)] for _ in range(self.size)]
        territory = {Player.BLACK: 0, Player.WHITE: 0}
        
        def get_territory_owner(row: int, col: int) -> Optional[Player]:
            if visited[row][col] or self.get_stone(row, col) != Player.EMPTY:
                return None
            
            territory_points = []
            adjacent_colors = set()
            to_check = [(row, col)]
            
            while to_check:
                r, c = to_check.pop()
                if visited[r][c]:
                    continue
                
                visited[r][c] = True
                stone = self.get_stone(r, c)
                
                if stone == Player.EMPTY:
                    territory_points.append((r, c))
                    for adj_r, adj_c in self.get_adjacent_points(r, c):
                        if not visited[adj_r][adj_c]:
                            to_check.append((adj_r, adj_c))
                else:
                    adjacent_colors.add(stone)
            
            if len(adjacent_colors) == 1:
                return adjacent_colors.pop()
            return None
        
        stones = {Player.BLACK: 0, Player.WHITE: 0}
        for row in range(self.size):
            for col in range(self.size):
                stone = self.get_stone(row, col)
                if stone in [Player.BLACK, Player.WHITE]:
                    stones[stone] += 1
        
        # Count territory
        for row in range(self.size):
            for col in range(self.size):
                if self.get_stone(row, col) == Player.EMPTY and not visited[row][col]:
                    owner = get_territory_owner(row, col)
                    if owner:
                        # Count territory points
                        temp_territory = []
                        to_check = [(row, col)]
                        temp_visited = set()
                        
                        while to_check:
                            r, c = to_check.pop()
                            if (r, c) in temp_visited or self.get_stone(r, c) != Player.EMPTY:
                                continue
                            temp_visited.add((r, c))
                            temp_territory.append((r, c))
                            
                            for adj_r, adj_c in self.get_adjacent_points(r, c):
                                if (adj_r, adj_c) not in temp_visited:
                                    to_check.append((adj_r, adj_c))
                        
                        territory[owner] += len(temp_territory)
        
        # Calculate final scores
        black_score = stones[Player.BLACK] + territory[Player.BLACK]
        white_score = stones[Player.WHITE] + territory[Player.WHITE] + komi
        
        winner = Player.BLACK if black_score > white_score else Player.WHITE
        
        return {
            'black': black_score,
            'white': white_score,
            'winner': winner,
            'black_stones': stones[Player.BLACK],
            'white_stones': stones[Player.WHITE],
            'black_territory': territory[Player.BLACK],
            'white_territory': territory[Player.WHITE],
            'black_captures': self.captured_stones[Player.BLACK],
            'white_captures': self.captured_stones[Player.WHITE]
        }
    
    def get_board_state(self) -> List[List[int]]:
        """Get the current board state as a 2D list of integers."""
        return [[cell.value for cell in row] for row in self.board]
    
    def clone(self):
        """Create a deep copy of the current board with cache."""
        # MCTS relies on clone() to explore futures without touching the real board
        new_board = GoBoard(self.size)
        new_board.board = [row[:] for row in self.board]  # faster than deepcopy
        new_board.current_player = self.current_player
        new_board.move_history = self.move_history[:]  # shallow copy is fine
        new_board.captured_stones = self.captured_stones.copy()
        new_board.ko_point = self.ko_point
        new_board.pass_count = self.pass_count
        
        # copy liberty cache
        new_board._group_id = [row[:] for row in self._group_id]
        new_board._group_liberties = {k: v.copy() for k, v in self._group_liberties.items()}
        new_board._group_stones = {k: v.copy() for k, v in self._group_stones.items()}
        new_board._next_group_id = self._next_group_id
        # dont copy undo/redo stacks for clones (mcts clones dont need em)
        return new_board
    
    def _save_state_for_undo(self):
        """save current state to undo stack"""
        state = {
            'board': [row[:] for row in self.board],
            'current_player': self.current_player,
            'move_history': self.move_history[:],
            'captured_stones': self.captured_stones.copy(),
            'ko_point': self.ko_point,
            'pass_count': self.pass_count,
            'group_id': [row[:] for row in self._group_id],
            'group_liberties': {k: v.copy() for k, v in self._group_liberties.items()},
            'group_stones': {k: v.copy() for k, v in self._group_stones.items()},
            'next_group_id': self._next_group_id,
        }
        self._undo_stack.append(state)
    
    def _restore_state(self, state):
        """restore board from a saved state dict"""
        self.board = [row[:] for row in state['board']]
        self.current_player = state['current_player']
        self.move_history = state['move_history'][:]
        self.captured_stones = state['captured_stones'].copy()
        self.ko_point = state['ko_point']
        self.pass_count = state['pass_count']
        self._group_id = [row[:] for row in state['group_id']]
        self._group_liberties = {k: v.copy() for k, v in state['group_liberties'].items()}
        self._group_stones = {k: v.copy() for k, v in state['group_stones'].items()}
        self._next_group_id = state['next_group_id']
    
    def can_undo(self) -> bool:
        """check if undo is possible"""
        return len(self._undo_stack) > 0
    
    def can_redo(self) -> bool:
        """check if redo is possible"""
        return len(self._redo_stack) > 0
    
    def undo(self) -> bool:
        """go back one move. returns True if successful"""
        if not self.can_undo():
            return False
        
        # save current state to redo stack first
        current_state = {
            'board': [row[:] for row in self.board],
            'current_player': self.current_player,
            'move_history': self.move_history[:],
            'captured_stones': self.captured_stones.copy(),
            'ko_point': self.ko_point,
            'pass_count': self.pass_count,
            'group_id': [row[:] for row in self._group_id],
            'group_liberties': {k: v.copy() for k, v in self._group_liberties.items()},
            'group_stones': {k: v.copy() for k, v in self._group_stones.items()},
            'next_group_id': self._next_group_id,
        }
        self._redo_stack.append(current_state)
        
        # pop and restore previous state
        prev_state = self._undo_stack.pop()
        self._restore_state(prev_state)
        return True
    
    def redo(self) -> bool:
        """go forward one move. returns True if successful"""
        if not self.can_redo():
            return False
        
        # save current to undo stack
        current_state = {
            'board': [row[:] for row in self.board],
            'current_player': self.current_player,
            'move_history': self.move_history[:],
            'captured_stones': self.captured_stones.copy(),
            'ko_point': self.ko_point,
            'pass_count': self.pass_count,
            'group_id': [row[:] for row in self._group_id],
            'group_liberties': {k: v.copy() for k, v in self._group_liberties.items()},
            'group_stones': {k: v.copy() for k, v in self._group_stones.items()},
            'next_group_id': self._next_group_id,
        }
        self._undo_stack.append(current_state)
        
        # pop and restore next state
        next_state = self._redo_stack.pop()
        self._restore_state(next_state)
        return True
    
    def _place_stone_update_cache(self, row: int, col: int, player: Player):
        """Update liberty cache after placing a stone"""
        # find adjacent friendly groups to merge with
        friendly_gids = set()
        enemy_gids = set()
        
        for adj_r, adj_c in self.get_adjacent_points_fast(row, col):
            adj_gid = self._group_id[adj_r][adj_c]
            if adj_gid > 0:
                adj_color = self.board[adj_r][adj_c]
                if adj_color == player:
                    friendly_gids.add(adj_gid)
                elif adj_color != Player.EMPTY:
                    enemy_gids.add(adj_gid)
        
        # remove (row, col) from enemy groups' liberties
        for gid in enemy_gids:
            if gid in self._group_liberties:
                self._group_liberties[gid].discard((row, col))
        
        if friendly_gids:
            # merge into the first friendly group
            main_gid = min(friendly_gids)  # pick consistent one
            self._group_id[row][col] = main_gid
            self._group_stones[main_gid].add((row, col))
            
            # merge other friendly groups into main
            for gid in friendly_gids:
                if gid != main_gid and gid in self._group_stones:
                    for r, c in self._group_stones[gid]:
                        self._group_id[r][c] = main_gid
                        self._group_stones[main_gid].add((r, c))
                    if gid in self._group_liberties:
                        self._group_liberties[main_gid] |= self._group_liberties[gid]
                    self._group_stones.pop(gid, None)
                    self._group_liberties.pop(gid, None)
            
            # remove placed position from liberties, add new liberties
            self._group_liberties[main_gid].discard((row, col))
            for adj_r, adj_c in self.get_adjacent_points_fast(row, col):
                if self.board[adj_r][adj_c] == Player.EMPTY:
                    self._group_liberties[main_gid].add((adj_r, adj_c))
        else:
            # create new group
            gid = self._next_group_id
            self._next_group_id += 1
            self._group_id[row][col] = gid
            self._group_stones[gid] = {(row, col)}
            self._group_liberties[gid] = set()
            for adj_r, adj_c in self.get_adjacent_points_fast(row, col):
                if self.board[adj_r][adj_c] == Player.EMPTY:
                    self._group_liberties[gid].add((adj_r, adj_c))
    
    def _rebuild_cache(self):
        """Rebuild the entire liberty cache from scratch"""
        self._group_id = [[0] * self.size for _ in range(self.size)]
        self._group_liberties = {}
        self._group_stones = {}
        self._next_group_id = 1
        
        visited = [[False] * self.size for _ in range(self.size)]
        
        for row in range(self.size):
            for col in range(self.size):
                if not visited[row][col] and self.board[row][col] != Player.EMPTY:
                    # BFS to find group
                    group = set()
                    liberties = set()
                    color = self.board[row][col]
                    stack = [(row, col)]
                    
                    while stack:
                        r, c = stack.pop()
                        if visited[r][c]:
                            continue
                        if self.board[r][c] != color:
                            if self.board[r][c] == Player.EMPTY:
                                liberties.add((r, c))
                            continue
                        visited[r][c] = True
                        group.add((r, c))
                        for adj_r, adj_c in self.get_adjacent_points_fast(r, c):
                            if not visited[adj_r][adj_c]:
                                stack.append((adj_r, adj_c))
                    
                    gid = self._next_group_id
                    self._next_group_id += 1
                    for r, c in group:
                        self._group_id[r][c] = gid
                    self._group_stones[gid] = group
                    self._group_liberties[gid] = liberties
    
    def __str__(self) -> str:
        """String representation of the board."""
        result = "  " + " ".join([column_to_label(i) for i in range(self.size)]) + "\n"
        for i, row in enumerate(self.board):
            row_str = f"{i + 1} "
            for cell in row:
                if cell == Player.BLACK:
                    row_str += "● "
                elif cell == Player.WHITE:
                    row_str += "○ "
                else:
                    row_str += "· "
            result += row_str + "\n"
        return result
