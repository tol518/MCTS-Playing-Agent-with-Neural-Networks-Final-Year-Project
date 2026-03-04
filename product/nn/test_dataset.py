import os
import shutil
import torch
from torch.utils.data import DataLoader
from dataset import GoDataset, save_game_data

def main():
    test_dir = "dummy_test_data"
    
    print(f"Creating test directory: {test_dir}")
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # 1. Generate dummy data
        print("Generating dummy data...")
        N1 = 20  # Samples in file 1
        states1 = torch.randn(N1, 16, 9, 9)
        policies1 = torch.rand(N1, 82)
        values1 = torch.randn(N1, 1)
        
        N2 = 35  # Samples in file 2
        states2 = torch.randn(N2, 16, 9, 9)
        policies2 = torch.rand(N2, 82)
        values2 = torch.randn(N2, 1)
        
        # 2. Save it using the helper function
        print("Saving game data using save_game_data()...")
        save_game_data(os.path.join(test_dir, "game_001.pt"), states1, policies1, values1)
        save_game_data(os.path.join(test_dir, "game_002.pt"), states2, policies2, values2)
        
        # 3. Initialize GoDataset
        print("Initializing GoDataset...")
        dataset = GoDataset(test_dir)
        print(f"Dataset total length: {len(dataset)} (Expected: {N1 + N2})")
        assert len(dataset) == N1 + N2, "Dataset length mismatch!"
        
        # 4. Initialize DataLoader
        print("Initializing DataLoader...")
        batch_size = 16
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 5. Iterate over a batch and verify shapes
        print("Iterating over a batch...")
        states_batch, policies_batch, values_batch = next(iter(dataloader))
        
        print("\nBatch Shapes:")
        print(f"  states: {states_batch.shape}  (Expected: [{batch_size}, 16, 9, 9])")
        print(f"  policies: {policies_batch.shape}  (Expected: [{batch_size}, 82])")
        print(f"  values: {values_batch.shape}  (Expected: [{batch_size}, 1])")
        
        # Assertions to ensure correctness
        assert states_batch.shape == (batch_size, 16, 9, 9), "States shape mismatch!"
        assert policies_batch.shape == (batch_size, 82), "Policies shape mismatch!"
        assert values_batch.shape == (batch_size, 1), "Values shape mismatch!"
        
        print("\nAll tests passed successfully!")
        
    finally:
        # Cleanup dummy data
        print(f"\nCleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    main()
