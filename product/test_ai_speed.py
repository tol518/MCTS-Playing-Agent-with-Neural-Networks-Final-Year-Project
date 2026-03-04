from go_engine import GoBoard, Player, column_to_label
from mcts_agent import MCTSAgent
import time
import os


if __name__ == '__main__':
    print("Testing AI performance...\n")
    print(f"CPU count: {os.cpu_count()}")

    board = GoBoard(9)
    print(f"Legal moves available: {len(board.get_legal_moves())}")

    # Test 1: Sequential (force single-threaded)
    print("\n--- Sequential MCTS (no multiprocessing) ---")
    ai_seq = MCTSAgent(simulation_time=5.0)
    ai_seq._use_parallel = False  # force sequential
    start = time.time()
    move = ai_seq.select_move(board)
    elapsed = time.time() - start
    if move:
        print(f"  Move: ({move[0]+1}, {column_to_label(move[1])})")

    # Test 2: Parallel
    print("\n--- Parallel MCTS ---")
    board2 = GoBoard(9)
    ai_par = MCTSAgent(simulation_time=5.0)
    ai_par._use_parallel = True
    print(f"  Workers: {ai_par._num_workers}")
    start = time.time()
    move = ai_par.select_move(board2)
    elapsed = time.time() - start
    if move:
        print(f"  Move: ({move[0]+1}, {column_to_label(move[1])})")

    print("\nDone!")
