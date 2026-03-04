import torch
from product.go_engine import GoBoard, Player

class GoBoardEncoder:
    """
    Encodes a GoBoard into a 16-channel PyTorch tensor representing the board state.
    """
    
    def encode(self, board: GoBoard) -> torch.Tensor:
        """
        Takes a GoBoard and returns a PyTorch tensor of shape [16, board.size, board.size].
        
        Channels:
        0: Current player stones (1 if yes, 0 otherwise)
        1: Opponent stones
        2: Empty spots
        3: Legal moves
        4: Current player stones with 1 liberty
        5: Current player stones with 2 liberties
        6: Current player stones with >=3 liberties
        7: Opponent stones with 1 liberty
        8: Opponent stones with 2 liberties
        9: Opponent stones with >=3 liberties
        10: Ko point
        11-14: The 4 most recent moves (11: most recent, 14: 4th most recent)
        15: Player color (all 1s if Black, all 0s if White)
        """
        size = board.size
        tensor = torch.zeros((16, size, size), dtype=torch.float32)
        
        current_player = board.current_player
        opponent = board.get_opponent(current_player)
        
        # Channel 15: Player color (1s if BLACK)
        if current_player == Player.BLACK:
            tensor[15, :, :] = 1.0
            
        # Channel 10: Ko point
        if board.ko_point is not None:
            r, c = board.ko_point
            tensor[10, r, c] = 1.0
            
        # Channels 11-14: The 4 most recent moves
        history = board.move_history
        for i in range(min(4, len(history))):
            move = history[-(i + 1)]
            r, c, _, _ = move
            if r != -1 and c != -1:  # Not a pass move
                tensor[11 + i, r, c] = 1.0
                
        # Loop through the board to fill in stones and liberties
        for r in range(size):
            for c in range(size):
                stone = board.get_stone(r, c)
                
                # Channel 2: Empty spots
                if stone == Player.EMPTY:
                    tensor[2, r, c] = 1.0
                    
                    # Channel 3: Legal moves
                    if board.is_legal_move(r, c, current_player):
                        tensor[3, r, c] = 1.0
                else:
                    libs = board.count_liberties(r, c)
                    
                    if stone == current_player:
                        # Channel 0: Current player stones
                        tensor[0, r, c] = 1.0
                        
                        # Current player liberties (Channels 4-6)
                        if libs == 1:
                            tensor[4, r, c] = 1.0
                        elif libs == 2:
                            tensor[5, r, c] = 1.0
                        elif libs >= 3:
                            tensor[6, r, c] = 1.0
                            
                    elif stone == opponent:
                        # Channel 1: Opponent stones
                        tensor[1, r, c] = 1.0
                        
                        # Opponent liberties (Channels 7-9)
                        if libs == 1:
                            tensor[7, r, c] = 1.0
                        elif libs == 2:
                            tensor[8, r, c] = 1.0
                        elif libs >= 3:
                            tensor[9, r, c] = 1.0
                            
        return tensor
