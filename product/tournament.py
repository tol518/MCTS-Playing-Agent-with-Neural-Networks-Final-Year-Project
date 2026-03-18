"""
Week 8: Tournament Evaluation

Pits the NN-guided MCTS (using the self-play RL checkpoint) against
the baseline Old-MCTS to definitively measure the win rate and improvement.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from product.go_engine import GoBoard, Player, column_to_label
from product.mcts_agent import MCTSAgent
try:
    from product.nn_mcts_agent import NNMCTSAgent
except ImportError:
    from nn_mcts_agent import NNMCTSAgent


def play_game(black_agent, white_agent, board_size=9, render=True):
    """Play one complete game between two agents."""
    board = GoBoard(board_size)
    consecutive_passes = 0
    move_count = 0
    max_moves = 120

    while not board.is_game_over() and move_count < max_moves and consecutive_passes < 2:
        current_agent = black_agent if board.current_player == Player.BLACK else white_agent
        player_name = "Black" if board.current_player == Player.BLACK else "White"
        
        start_time = time.time()
        move = current_agent.select_move(board)
        elapsed = time.time() - start_time
        
        if move is None:
            if render:
                print(f"Move {move_count+1}: {player_name} passes ({elapsed:.1f}s)")
            board.pass_turn()
            consecutive_passes += 1
        else:
            row, col = move
            if not board.is_legal_move(row, col):
                print(f"WARNING: {player_name} attempted illegal move, forcing pass.")
                board.pass_turn()
                consecutive_passes += 1
            else:
                if render:
                    print(f"Move {move_count+1}: {player_name} plays ({row+1}, {column_to_label(col)}) ({elapsed:.1f}s)")
                board.make_move(row, col)
                consecutive_passes = 0
                
            
        move_count += 1

    # Score game
    try:
        score_data = board.calculate_score()
        winner = score_data.get('winner')
    except Exception:
        # Fallback scoring
        b, w = 0, 0
        for r in range(board_size):
            for c in range(board_size):
                if board.board[r][c] == Player.BLACK: b += 1
                elif board.board[r][c] == Player.WHITE: w += 1
        b += board.captured_stones[Player.BLACK]
        w += board.captured_stones[Player.WHITE] + 6.5
        winner = Player.BLACK if b > w else Player.WHITE

    if render:
        print(f"\nGame over! Winner: {'Black' if winner == Player.BLACK else 'White'}")
        
    return winner


def run_tournament(num_games=10, think_time=5.0):
    print("=" * 60)
    print("🏆 NEURAL NETWORK VS BASELINE TOURNAMENT 🏆")
    print("=" * 60)
    
    # Load the Self-Play checkpoint
    checkpoint_path = Path(__file__).resolve().parent / "nn" / "checkpoints" / "rl_last.pt"
    if not checkpoint_path.exists():
        print(f"Error: Could not find RL checkpoint at {checkpoint_path}")
        return

    print(f"Loading NN Model from: {checkpoint_path.name}")
    nn_agent = NNMCTSAgent(
        checkpoint_path=str(checkpoint_path),
        simulation_time=think_time,
        c_puct=1.5
    )
    
    old_agent = MCTSAgent(simulation_time=think_time)
    
    nn_wins = 0
    old_wins = 0
    
    # Play alternating colors so it's fair
    for game in range(num_games):
        print(f"\n--- Match {game + 1}/{num_games} ---")
        
        if game % 2 == 0:
            print("NN-MCTS is Black ⚫ vs Old-MCTS is White ⚪")
            winner = play_game(black_agent=nn_agent, white_agent=old_agent, render=True)
            if winner == Player.BLACK:
                nn_wins += 1
            else:
                old_wins += 1
        else:
            print("Old-MCTS is Black ⚫ vs NN-MCTS is White ⚪")
            winner = play_game(black_agent=old_agent, white_agent=nn_agent, render=True)
            if winner == Player.WHITE:
                nn_wins += 1
            else:
                old_wins += 1
                
        print(f"Current Score -> NN-MCTS: {nn_wins} | Old-MCTS: {old_wins}")
        
    print("\n" + "=" * 60)
    print("🏁 TOURNAMENT RESULTS 🏁")
    print(f"NN-MCTS (Self-Play): {nn_wins} wins")
    print(f"Old-MCTS (Baseline): {old_wins} wins")
    win_rate = (nn_wins / num_games) * 100
    print(f"NN Win Rate: {win_rate:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_tournament(num_games=4, think_time=5.0)
