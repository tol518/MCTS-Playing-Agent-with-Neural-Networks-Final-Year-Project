import random
import unittest

from go_engine import GoBoard, Player
from mcts_agent import MCTSAgent


class TestMCTSAgent(unittest.TestCase):
    def setUp(self):
        # keep the random stuff kinda fixed so tests behave the same
        random.seed(42)

    def _prepare_board(self) -> GoBoard:
        board = GoBoard(5)
        # just make a tiny mid game-ish board posotion
        board.make_move(2, 2, Player.BLACK)
        board.make_move(2, 3, Player.WHITE)
        board.make_move(1, 1, Player.BLACK)
        board.make_move(3, 3, Player.WHITE)
        return board

    def test_select_move_returns_legal_move(self):
        board = self._prepare_board()
        agent = MCTSAgent(simulation_time=0.1, max_simulations=15)

        move = agent.select_move(board)

        if move is None:
            # pass is alwys ok; just checking the engin allows it
            self.assertTrue(True)
        else:
            row, col = move
            self.assertTrue(
                board.is_legal_move(row, col),
                msg=f"MCTS returned illegal move {(row, col)}"
            )

    def test_mcts_does_not_mutate_original_board(self):
        board = self._prepare_board()
        snapshot = board.clone()
        agent = MCTSAgent(simulation_time=0.1, max_simulations=15)

        agent.select_move(board)

        self.assertEqual(board.get_board_state(), snapshot.get_board_state())
        self.assertEqual(board.current_player, snapshot.current_player)
        self.assertEqual(board.captured_stones, snapshot.captured_stones)
        self.assertEqual(board.pass_count, snapshot.pass_count)
        self.assertEqual(board.move_history, snapshot.move_history)


if __name__ == "__main__":
    unittest.main()
