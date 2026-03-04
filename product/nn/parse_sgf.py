"""
parse_sgf.py — Convert downloaded .sgf files (from CGOS / online) into
PyTorch .pt training datasets compatible with the NN training pipeline.

Usage:
    $env:PYTHONPATH = "."; python product/nn/parse_sgf.py --sgf-dir 9x9_2018_11 --output-dir product/nn/data
"""

import os
import sys
import re
import gc
import argparse
import time
from pathlib import Path

# Lazy-import torch (same pattern as generate_data.py to avoid WinError 1455)
# torch is imported inside convert_all() below.

# Setup paths for both package and flat imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_DIR = PROJECT_ROOT / "product"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PRODUCT_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCT_DIR))

try:
    from product.go_engine import GoBoard, Player
except ModuleNotFoundError:
    from go_engine import GoBoard, Player


# ---------------------------------------------------------------------------
# SGF Parsing helpers
# ---------------------------------------------------------------------------

def _parse_sgf_metadata(sgf_text: str) -> dict:
    """Extract key metadata from an SGF header."""
    meta = {}
    # Board size
    m = re.search(r'SZ\[(\d+)\]', sgf_text)
    meta['size'] = int(m.group(1)) if m else 19

    # Result
    m = re.search(r'RE\[([^\]]+)\]', sgf_text)
    meta['result'] = m.group(1) if m else None

    # Komi
    m = re.search(r'KM\[([^\]]+)\]', sgf_text)
    meta['komi'] = float(m.group(1)) if m else 7.0

    # Player names
    m = re.search(r'PB\[([^\]]+)\]', sgf_text)
    meta['black'] = m.group(1) if m else 'Unknown'
    m = re.search(r'PW\[([^\]]+)\]', sgf_text)
    meta['white'] = m.group(1) if m else 'Unknown'

    return meta


def _parse_sgf_moves(sgf_text: str) -> list:
    """
    Extract the ordered list of moves from an SGF string.
    Returns list of (color, row, col) tuples.
    color is 'B' or 'W'.
    row, col are 0-indexed integers.  None for pass moves.
    """
    moves = []
    # Match ;B[xy] or ;W[xy] patterns (including passes like ;B[] or ;W[])
    pattern = re.compile(r';([BW])\[([a-i]{0,2})\]')
    for match in pattern.finditer(sgf_text):
        color = match.group(1)
        coords = match.group(2)
        if len(coords) == 2:
            # SGF coords: 'a' = column 0, row 0  (col first, then row)
            col = ord(coords[0]) - ord('a')
            row = ord(coords[1]) - ord('a')
            moves.append((color, row, col))
        else:
            # Empty brackets = pass
            moves.append((color, None, None))
    return moves


def _result_to_winner(result_str: str) -> str:
    """
    Parse SGF result string into 'B', 'W', or 'DRAW'.
    Examples: 'B+6.0', 'W+Resign', 'B+R', 'W+3.5', '0' (draw/jigo)
    """
    if result_str is None:
        return None
    result_str = result_str.strip()
    if result_str.startswith('B+'):
        return 'B'
    elif result_str.startswith('W+'):
        return 'W'
    elif result_str == '0' or 'Jigo' in result_str:
        return 'DRAW'
    return None


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def convert_single_sgf(sgf_path: str, encoder, torch_module):
    """
    Convert a single .sgf file into (states, policies, values) tensors.

    Returns:
        (states_list, policies_list, values_list) — lists of tensors for
        each position in the game, or None if the file is unusable.
    """
    torch = torch_module

    with open(sgf_path, 'r', encoding='utf-8', errors='ignore') as f:
        sgf_text = f.read()

    meta = _parse_sgf_metadata(sgf_text)

    # Only process 9x9 games
    if meta['size'] != 9:
        return None

    winner_str = _result_to_winner(meta['result'])
    if winner_str is None or winner_str == 'DRAW':
        # Skip games with unknown or draw results
        return None

    winner = Player.BLACK if winner_str == 'B' else Player.WHITE
    moves = _parse_sgf_moves(sgf_text)

    if len(moves) < 5:
        # Skip trivially short games
        return None

    board = GoBoard(size=9)
    action_size = 9 * 9 + 1  # 82

    states = []
    policies = []
    values = []

    for color, row, col in moves:
        expected_player = board.current_player
        sgf_player = Player.BLACK if color == 'B' else Player.WHITE

        # Sanity check: SGF color should match engine's current player
        if sgf_player != expected_player:
            # Mismatch — likely a corrupt or non-standard SGF; skip remaining
            break

        # 1. Encode current board state
        state_tensor = encoder.encode(board)

        # 2. Create one-hot policy target
        policy = torch.zeros(action_size, dtype=torch.float32)
        if row is None:
            # Pass move
            policy[action_size - 1] = 1.0
            board.pass_turn()
        else:
            idx = row * 9 + col
            policy[idx] = 1.0
            if not board.make_move(row, col):
                # Illegal move in our engine — stop processing this game
                break

        # 3. Assign value based on game outcome
        value = 1.0 if expected_player == winner else -1.0

        states.append(state_tensor)
        policies.append(policy)
        values.append([value])

    if len(states) < 5:
        return None

    return states, policies, values


def convert_all(sgf_dir: str, output_dir: str, output_filename: str = None,
                max_games: int = None, chunk_size: int = 250,
                skip_files: int = 0):
    """
    Walk sgf_dir, parse all .sgf files, and save datasets in chunks.
    Rather than accumulating all data in RAM (which crashes), we flush
    every `chunk_size` successful games to a separate .pt file.
    This means you end up with multiple chunk files that together form
    the full dataset.
    """
    # Lazy imports (torch + encoder + dataset saver)
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

    # Collect all .sgf files
    sgf_files = []
    for root, _dirs, files in os.walk(sgf_dir):
        for fname in files:
            if fname.lower().endswith('.sgf'):
                sgf_files.append(os.path.join(root, fname))

    if not sgf_files:
        print(f"No .sgf files found in {sgf_dir}")
        return None

    print(f"Found {len(sgf_files)} .sgf files in {sgf_dir}")
    if skip_files > 0:
        sgf_files = sgf_files[skip_files:]
        print(f"Skipping first {skip_files} files (resuming)")
    if max_games:
        sgf_files = sgf_files[:max_games]
        print(f"Processing first {max_games} games")

    # In-progress chunk buffers
    chunk_states = []
    chunk_policies = []
    chunk_values = []

    start_time = time.time()
    success = 0
    skipped = 0
    chunk_idx = 0
    saved_files = []

    def _flush_chunk():
        """Save the current chunk buffer to disk and clear it."""
        nonlocal chunk_idx
        if not chunk_states:
            return
        states_t = torch.stack(chunk_states)
        policies_t = torch.stack(chunk_policies)
        values_t = torch.tensor(chunk_values, dtype=torch.float32)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fname = f"cgos_chunk{chunk_idx:03d}_{len(chunk_states)}samples_{timestamp}.pt"
        fpath = os.path.join(output_dir, fname)
        save_game_data(fpath, states_t, policies_t, values_t)
        saved_files.append(fpath)
        print(f"\n  [Chunk {chunk_idx}] Saved {len(chunk_states)} samples -> {fname}")
        chunk_states.clear()
        chunk_policies.clear()
        chunk_values.clear()
        # Force Python + PyTorch to release memory back to the OS
        del states_t, policies_t, values_t
        gc.collect()
        chunk_idx += 1

    for i, sgf_path in enumerate(sgf_files):
        try:
            result = convert_single_sgf(sgf_path, encoder, torch)
        except Exception:
            skipped += 1
            result = None

        if result is None:
            skipped += 1
        else:
            states, policies, values = result
            chunk_states.extend(states)
            chunk_policies.extend(policies)
            chunk_values.extend(values)
            success += 1

            # Flush chunk when we hit chunk_size successful games
            if success % chunk_size == 0:
                _flush_chunk()

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(sgf_files)} files "
                  f"({success} ok, {skipped} skipped, "
                  f"{len(chunk_states)} in current chunk)")

    # Flush any remaining data
    _flush_chunk()

    elapsed = time.time() - start_time
    print(f"\n=== SGF Conversion Complete ===")
    print(f"Games processed:  {success} (skipped {skipped})")
    print(f"Chunks saved:     {len(saved_files)}")
    for f in saved_files:
        print(f"  {f}")
    print(f"Time:             {elapsed:.2f}s")
    return saved_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .sgf game files into PyTorch training data."
    )
    parser.add_argument("--sgf-dir", type=str, required=True,
                        help="Root directory containing .sgf files")
    parser.add_argument("--output-dir", type=str, default="product/nn/data",
                        help="Directory to save the .pt output file")
    parser.add_argument("--output-filename", type=str, default=None,
                        help="Custom filename for the output .pt file")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Max number of SGF files to process")
    parser.add_argument("--skip-files", type=int, default=0,
                        help="Number of files to skip (resume from this point)")
    args = parser.parse_args()

    convert_all(
        sgf_dir=args.sgf_dir,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        max_games=args.max_games,
        skip_files=args.skip_files,
    )
