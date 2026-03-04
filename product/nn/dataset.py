import os
import gc
import glob
import bisect
import torch
from torch.utils.data import Dataset

def save_game_data(filepath, states, policies, values):
    """
    Saves game data to a .pt file.
    
    Args:
        filepath (str): Path to save the .pt file.
        states (torch.Tensor): Tensor of shape [N, 16, 9, 9].
        policies (torch.Tensor): Tensor of shape [N, 82].
        values (torch.Tensor): Tensor of shape [N, 1].
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    torch.save({
        'states': states,
        'policies': policies,
        'values': values
    }, filepath)

from torch.utils.data import IterableDataset
import random

class GoDataset(IterableDataset):
    def __init__(self, data_dir, shuffle=True):
        """
        Scans data_dir for .pt files and yields samples chunk by chunk.
        This prevents disk thrashing and OOMs caused by random access across
        multiple large files.
        """
        self.data_dir = data_dir
        self.files = sorted(glob.glob(os.path.join(data_dir, '**', '*.pt'), recursive=True))
        self.shuffle = shuffle
        
        # We still need to calculate total length for the progress bar / epoch sizing
        self.total_length = 0
        for f in self.files:
            data = torch.load(f, map_location='cpu', weights_only=True)
            self.total_length += data['states'].size(0)
            del data
            
        print(f"Dataset ready: {len(self.files)} files, {self.total_length} total samples")

    def __len__(self):
        """Returns the total number of samples so DataLoader knows epoch size."""
        return self.total_length

    def __iter__(self):
        """Yields individual (state, policy, value) samples sequentially."""
        files_to_read = list(self.files)
        if self.shuffle:
            # Shuffle the order of chunk files we process
            random.shuffle(files_to_read)

        for f in files_to_read:
            # Load one full chunk into memory
            data = torch.load(f, map_location='cpu', weights_only=True)
            states = data['states']
            policies = data['policies']
            values = data['values']
            chunk_size = states.size(0)
            
            indices = list(range(chunk_size))
            if self.shuffle:
                # Shuffle the samples within this specific chunk
                random.shuffle(indices)
                
            for idx in indices:
                yield states[idx], policies[idx], values[idx]
                
            # Free memory before loading the next chunk
            del data, states, policies, values
            gc.collect()
