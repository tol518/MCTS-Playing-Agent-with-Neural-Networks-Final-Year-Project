import copy
from typing import List, Tuple, Set, Optional
from enum import Enum


class Player(Enum):
    """Represents the two players in Go."""
    BLACK = 1
    WHITE = 2
    EMPTY = 0


class GoBoard:
    """
    Represents a 9x9 Go board with complete game logic.
    """
    
    def __init__(self, size: int = 9):
        self.size = size
        self.board = [[Player.EMPTY for _ in range(size)] for _ in range(size)]
        self.current_player = Player.BLACK
        self.move_history = []
        self.captured_stones = {Player.BLACK: 0, Player.WHITE: 0}
        self.ko_point = None  # Tracks ko position
        self.pass_count = 0
        
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
        adjacent = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if self.is_on_board(new_row, new_col):
                adjacent.append((new_row, new_col))
        return adjacent
    
    def get_group(self, row: int, col: int) -> Set[Tuple[int, int]]:
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
        group = self.get_group(row, col)
        liberties = set()
        
        for r, c in group:
            for adj_r, adj_c in self.get_adjacent_points(r, c):
                if self.get_stone(adj_r, adj_c) == Player.EMPTY:
                    liberties.add((adj_r, adj_c))
        
        return len(liberties)
    
    def get_liberties(self, row: int, col: int) -> Set[Tuple[int, int]]:
        group = self.get_group(row, col)
        liberties = set()
        
        for r, c in group:
            for adj_r, adj_c in self.get_adjacent_points(r, c):
                if self.get_stone(adj_r, adj_c) == Player.EMPTY:
                    liberties.add((adj_r, adj_c))
        
        return liberties
    
    def remove_group(self, row: int, col: int) -> int:
        group = self.get_group(row, col)
        for r, c in group:
            self.set_stone(r, c, Player.EMPTY)
        return len(group)
    
    def capture_adjacent_groups(self, row: int, col: int, player: Player) -> List[Tuple[int, int]]:
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
        
        # Store board state for ko detection
        previous_board = copy.deepcopy(self.board)
        
        # Place the stone
        self.set_stone(row, col, player)
        
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
        """Create a deep copy of the current board."""
        new_board = GoBoard(self.size)
        new_board.board = copy.deepcopy(self.board)
        new_board.current_player = self.current_player
        new_board.move_history = copy.deepcopy(self.move_history)
        new_board.captured_stones = copy.deepcopy(self.captured_stones)
        new_board.ko_point = self.ko_point
        new_board.pass_count = self.pass_count
        return new_board
    
    def __str__(self) -> str:
        """String representation of the board."""
        result = "  " + " ".join([chr(65 + i) for i in range(self.size)]) + "\n"
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
