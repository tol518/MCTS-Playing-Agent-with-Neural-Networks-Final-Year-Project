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
        # Don't add pass as an untried move - only use it if no legal moves
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
    
    def select_move(self, board: GoBoard) -> Optional[Tuple[int, int]]:

        root = MCTSNode(board.clone())
        
        # Run simulations
        start_time = time.time()
        simulations = 0
        
        while True:
            # Check termination conditions
            if self.max_simulations and simulations >= self.max_simulations:
                break
            if not self.max_simulations and (time.time() - start_time) >= self.simulation_time:
                break
            
            # MCTS four phases
            node = self._select(root)
            if not node.is_terminal():
                node = self._expand(node)
            result = self._simulate(node)
            self._backpropagate(node, result)
            
            simulations += 1
        
        # Select the move with the most visits (most robust choice)
        if not root.children:
            # No legal moves, must pass
            return None
        
        best_child = max(root.children, key=lambda c: c.visits)
        
        print(f"MCTS completed {simulations} simulations in {time.time() - start_time:.2f}s")
        print(f"Selected move: {best_child.move}, Visits: {best_child.visits}, "
              f"Win rate: {best_child.wins / best_child.visits if best_child.visits > 0 else 0:.2%}")
        
        return best_child.move
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        return node.expand()
    
    def _simulate(self, node: MCTSNode) -> float:
        simulation_board = node.board.clone()
        starting_player = node.player
        
        # Play random moves until game is over or max moves
        max_moves = 30  # Keep simulations short for speed
        moves_made = 0
        consecutive_passes = 0
        
        while consecutive_passes < 2 and moves_made < max_moves:
            legal_moves = simulation_board.get_legal_moves()
            
            if not legal_moves:
                # Must pass if no legal moves
                simulation_board.pass_turn()
                consecutive_passes += 1
            else:
                # Choose a random legal move with some heuristics
                move = self._select_simulation_move(simulation_board, legal_moves)
                if move:
                    row, col = move
                    simulation_board.make_move(row, col)
                    consecutive_passes = 0
                else:
                    simulation_board.pass_turn()
                    consecutive_passes += 1
            
            moves_made += 1
        
        # Fast evaluation without expensive scoring
        return self._fast_evaluate(simulation_board, starting_player)
    
    def _select_simulation_move(self, board: GoBoard, legal_moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if not legal_moves:
            return None
        
        # Use simple fast heuristics without expensive board cloning
        # 80% random moves, 20% prefer center/corners for faster simulations
        if random.random() < 0.8:
            return random.choice(legal_moves)
        
        # Prefer center area moves (simple heuristic, no cloning needed)
        center_moves = []
        for row, col in legal_moves:
            # Prefer moves closer to center
            if 2 <= row <= 6 and 2 <= col <= 6:
                center_moves.append((row, col))
        
        if center_moves:
            return random.choice(center_moves)
        else:
            return random.choice(legal_moves)
    
    def _fast_evaluate(self, board: GoBoard, player: Player) -> float:
        black_stones = 0
        white_stones = 0
        
        for row in range(board.size):
            for col in range(board.size):
                stone = board.get_stone(row, col)
                if stone == Player.BLACK:
                    black_stones += 1
                elif stone == Player.WHITE:
                    white_stones += 1
        
        # Add captures
        black_score = black_stones + board.captured_stones[Player.BLACK]
        white_score = white_stones + board.captured_stones[Player.WHITE] + 6.5  # komi
        
        # Determine winner
        if player == Player.BLACK:
            if black_score > white_score:
                advantage = (black_score - white_score) / 20.0
                return 0.5 + min(0.5, advantage)
            else:
                disadvantage = (white_score - black_score) / 20.0
                return 0.5 - min(0.5, disadvantage)
        else:  # WHITE
            if white_score > black_score:
                advantage = (white_score - black_score) / 20.0
                return 0.5 + min(0.5, advantage)
            else:
                disadvantage = (black_score - white_score) / 20.0
                return 0.5 - min(0.5, disadvantage)
    
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
