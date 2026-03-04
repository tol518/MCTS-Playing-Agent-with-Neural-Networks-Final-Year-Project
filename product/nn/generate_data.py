import os
import sys
import time
import argparse
import multiprocessing as mp
from pathlib import Path

# NOTE: torch is NOT imported here on purpose!
# Importing torch at module level causes every multiprocessing worker to load
# the massive CUDA DLLs (~2GB), exhausting the Windows paging file (WinError 1455).
# Instead, torch is imported lazily inside generate_self_play_data() AFTER the
# MCTS agents (and their process pools) have already been created.

# Support both package-style imports (product.*) and flat imports (go_engine.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_DIR = PROJECT_ROOT / "product"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PRODUCT_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCT_DIR))

# Import only non-torch modules at top level (go_engine, mcts_agent are pure Python).
try:
    from product.go_engine import GoBoard, Player
    from product.mcts_agent import MCTSAgent
except ModuleNotFoundError:
    from go_engine import GoBoard, Player
    from mcts_agent import MCTSAgent

def generate_self_play_data(
    num_games=2,
    output_dir="product/nn/data/",
    max_simulations=50,
    max_moves=150,
    output_filename=None,
):
    """
    Generates self-play data by pitting MCTS against itself.
    Records board states, chosen moves (as one-hot policies), and final game values.
    """
    # Lazy-import torch and torch-dependent modules HERE, after the process pool
    # inside MCTSAgent has been created with lightweight workers.
    import torch
    try:
        from product.nn.encoder import GoBoardEncoder
        from product.nn.dataset import save_game_data
    except ModuleNotFoundError:
        try:
            from nn.encoder import GoBoardEncoder
            from nn.dataset import save_game_data
        except ModuleNotFoundError:
            from encoder import GoBoardEncoder
            from dataset import save_game_data

    os.makedirs(output_dir, exist_ok=True)
    encoder = GoBoardEncoder()
    
    agent_black = MCTSAgent(max_simulations=max_simulations)
    agent_white = MCTSAgent(max_simulations=max_simulations)
    
    all_states = []
    all_policies = []
    all_values = []
    
    start_time = time.time()
    
    for game_idx in range(num_games):
        print(f"Starting game {game_idx + 1}/{num_games}")
        board = GoBoard(size=9)
        
        game_states = []
        game_policies = []
        game_players = []
        
        move_count = 0
        # Hard cap on moves to prevent infinite loops in random/low-sim games
        while not board.is_game_over() and move_count < max_moves:
            current_player = board.current_player
            
            # 1. Get current board state tensor
            state_tensor = encoder.encode(board)
            
            # 2. Get best move from MCTS
            agent = agent_black if current_player == Player.BLACK else agent_white
            move = agent.select_move(board)
            
            # Create one-hot policy vector (82-length)
            policy = torch.zeros(82, dtype=torch.float32)
            if move is None:
                policy[81] = 1.0  # Pass move
            else:
                row, col = move
                policy[row * board.size + col] = 1.0
                
            # Store data
            game_states.append(state_tensor)
            game_policies.append(policy)
            game_players.append(current_player)
            
            # Apply move to board
            if move is None:
                board.pass_turn()
            else:
                board.make_move(move[0], move[1])
                
            move_count += 1
                
        # Calculate final game value
        score = board.calculate_score()
        winner = score['winner']
        print(f"Game {game_idx + 1} finished in {move_count} moves. Winner: {winner.name}")
        
        # Assign values: +1 if current player won, -1 if lost
        for p in game_players:
            if p == winner:
                all_values.append([1.0])
            else:
                all_values.append([-1.0])
                
        all_states.extend(game_states)
        all_policies.extend(game_policies)
        
    # Shutdown executor if any to prevent hanging processes
    if hasattr(agent_black, 'shutdown'): agent_black.shutdown()
    if hasattr(agent_white, 'shutdown'): agent_white.shutdown()
        
    # Convert lists to tensors
    states_tensor = torch.stack(all_states)
    policies_tensor = torch.stack(all_policies)
    values_tensor = torch.tensor(all_values, dtype=torch.float32)
    
    # Save the dataset
    if output_filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"selfplay_{num_games}g_{max_simulations}sims_{timestamp}.pt"
    filepath = os.path.join(output_dir, output_filename)
    save_game_data(filepath, states_tensor, policies_tensor, values_tensor)
    
    elapsed = time.time() - start_time
    print(f"Saved {len(all_states)} samples to {filepath}")
    print(f"Data shapes: states={states_tensor.shape}, policies={policies_tensor.shape}, values={values_tensor.shape}")
    print(f"Generation took {elapsed:.2f} seconds.")
    return filepath

if __name__ == "__main__":
    # On Windows, force 'spawn' start method so child processes start clean
    # without inheriting the parent's loaded CUDA DLLs.
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    parser = argparse.ArgumentParser(
        description="Generate MCTS self-play data for Week 3 policy warm-start."
    )
    parser.add_argument("--num-games", type=int, default=2)
    parser.add_argument("--max-simulations", type=int, default=50)
    parser.add_argument("--max-moves", type=int, default=150)
    parser.add_argument("--output-dir", type=str, default="product/nn/data")
    parser.add_argument("--output-filename", type=str, default=None)
    args = parser.parse_args()

    generate_self_play_data(
        num_games=args.num_games,
        output_dir=args.output_dir,
        max_simulations=args.max_simulations,
        max_moves=args.max_moves,
        output_filename=args.output_filename,
    )
