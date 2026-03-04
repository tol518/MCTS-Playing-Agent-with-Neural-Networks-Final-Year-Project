import unittest
import torch
from product.go_engine import GoBoard, Player
from product.nn.encoder import GoBoardEncoder

class TestGoBoardEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = GoBoardEncoder()
        self.board = GoBoard(size=9)

    def test_encode_shape(self):
        tensor = self.encoder.encode(self.board)
        self.assertEqual(tensor.shape, (16, 9, 9))

    def test_empty_board(self):
        tensor = self.encoder.encode(self.board)
        
        # Channel 15: Black to move, so all 1s
        self.assertTrue(torch.all(tensor[15] == 1.0))
        
        # Channel 2: All spots are empty
        self.assertTrue(torch.all(tensor[2] == 1.0))
        
        # Channel 0, 1: No stones
        self.assertTrue(torch.all(tensor[0] == 0.0))
        self.assertTrue(torch.all(tensor[1] == 0.0))
        
        # Channel 3: All spots are legal moves
        self.assertTrue(torch.all(tensor[3] == 1.0))

    def test_stone_placement_and_liberties(self):
        self.board.make_move(4, 4)  # Black plays center
        tensor = self.encoder.encode(self.board)
        
        # It's White's turn now
        self.assertEqual(self.board.current_player, Player.WHITE)
        
        # Channel 15: White to move, so all 0s
        self.assertTrue(torch.all(tensor[15] == 0.0))
        
        # The center stone is Black, which is the opponent (Channel 1)
        self.assertEqual(tensor[1, 4, 4].item(), 1.0)
        self.assertEqual(tensor[0, 4, 4].item(), 0.0)
        
        # It has 4 liberties, so Channel 9 (Opponent >=3 libs) should be 1
        self.assertEqual(tensor[9, 4, 4].item(), 1.0)
        self.assertEqual(tensor[7, 4, 4].item(), 0.0)
        self.assertEqual(tensor[8, 4, 4].item(), 0.0)
        
        # Channel 11: Most recent move
        self.assertEqual(tensor[11, 4, 4].item(), 1.0)
        
        # Channel 12-14: Should be empty
        self.assertTrue(torch.all(tensor[12:15] == 0.0))
        
    def test_one_and_two_liberties(self):
        # We will set up a board where a stone has 1 liberty and 2 liberties
        # Black plays (0, 0) -> 2 liberties
        self.board.make_move(0, 0)
        
        # White plays (0, 1) -> Black now has 1 liberty
        self.board.make_move(0, 1)
        
        # It's Black's turn
        tensor = self.encoder.encode(self.board)
        
        self.assertEqual(self.board.current_player, Player.BLACK)
        
        # Black (Current player) stone at (0, 0) has 1 liberty (at (1, 0))
        self.assertEqual(tensor[0, 0, 0].item(), 1.0)
        self.assertEqual(tensor[4, 0, 0].item(), 1.0)  # Current player 1 liberty
        self.assertEqual(tensor[5, 0, 0].item(), 0.0)
        self.assertEqual(tensor[6, 0, 0].item(), 0.0)
        
        # White (Opponent) stone at (0, 1) has 2 liberties (at (0, 2), (1, 1))
        self.assertEqual(tensor[1, 0, 1].item(), 1.0)
        self.assertEqual(tensor[8, 0, 1].item(), 1.0)  # Opponent 2 liberties
        self.assertEqual(tensor[7, 0, 1].item(), 0.0)
        self.assertEqual(tensor[9, 0, 1].item(), 0.0)
        
        # Moves history
        # Most recent: White (0, 1)
        self.assertEqual(tensor[11, 0, 1].item(), 1.0)
        # Second most recent: Black (0, 0)
        self.assertEqual(tensor[12, 0, 0].item(), 1.0)
        
    def test_ko_point(self):
        # Create a basic ko shape
        self.board.make_move(0, 1) # B
        self.board.make_move(0, 2) # W
        self.board.make_move(1, 0) # B
        self.board.make_move(1, 1) # W (will be captured)
        self.board.make_move(2, 1) # B
        self.board.make_move(1, 3) # W
        self.board.pass_turn()     # B passes
        self.board.make_move(2, 2) # W
        
        # B plays (1, 2) capturing W at (1, 1)
        succ = self.board.make_move(1, 2)
        self.assertTrue(succ)
        
        # W attempts to play (1, 1) capturing B at (1, 2) -> should be blocked by Ko
        succ = self.board.make_move(1, 1)
        self.assertFalse(succ)
        
        tensor = self.encoder.encode(self.board)
        
        # Now it's White's turn. The point (1, 1) is restricted by Ko.
        self.assertIsNotNone(self.board.ko_point)
        r, c = self.board.ko_point
        self.assertEqual((r, c), (1, 1))
        
        # Channel 10: Ko point
        self.assertEqual(tensor[10, r, c].item(), 1.0)
        
        # The ko point should not be a legal move
        self.assertEqual(tensor[3, r, c].item(), 0.0)

if __name__ == '__main__':
    unittest.main()
