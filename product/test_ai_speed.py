from go_engine import GoBoard, Player
from mcts_agent import MCTSAgent
import time

print("Testing AI performance...\n")

# Create empty board
board = GoBoard(9)
print("Empty 9x9 board created")
print(f"Legal moves available: {len(board.get_legal_moves())}")

# Create AI with short thinking time
ai = MCTSAgent(simulation_time=2.0)
print(f"AI created with 2 second thinking time\n")

# Test AI move
print("AI is thinking...")
start = time.time()
move = ai.select_move(board)
elapsed = time.time() - start

print(f"\nResult:")
print(f"  Time taken: {elapsed:.2f}s")
print(f"  Move selected: {move}")

if move is None:
    print("  ❌ FAIL: AI passed instead of making a move!")
else:
    row, col = move
    move_str = f"{chr(65 + col)}{row + 1}"
    print(f"  ✅ SUCCESS: AI played {move_str}")

print("\n" + "="*50)
print("If AI selected an actual move (not None), the fix worked!")
print("="*50)
