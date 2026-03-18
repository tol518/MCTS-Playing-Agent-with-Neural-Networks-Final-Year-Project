"""
Train the RL checkpoint on GUI game data (AI vs AI and Human vs AI games).
These are games recorded from the GUI that haven't been used in RL self-play training.
"""

import torch
import os
import sys
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from product.nn.model import AlphaGoZeroNetwork

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the latest RL checkpoint
    rl_path = PROJECT_ROOT / "product" / "nn" / "checkpoints" / "rl_last.pt"
    checkpoint = torch.load(str(rl_path), map_location=device, weights_only=False)
    iteration = checkpoint.get("iteration", "?")
    print(f"Loaded RL checkpoint (iteration {iteration})")

    model = AlphaGoZeroNetwork(
        in_channels=16, num_res_blocks=5, channels=128, board_size=9
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    if "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception:
            print("Optimizer state mismatch, using fresh optimizer.")

    policy_criterion = torch.nn.CrossEntropyLoss()
    value_criterion = torch.nn.MSELoss()

    # Load ALL human_games .pt files
    data_dir = PROJECT_ROOT / "product" / "nn" / "data" / "human_games"
    files = sorted(glob.glob(str(data_dir / "*.pt")))
    print(f"Found {len(files)} game data files:")

    all_states = []
    all_policies = []
    all_values = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=True)
        n_samples = d["states"].shape[0]
        all_states.append(d["states"])
        all_policies.append(d["policies"])
        all_values.append(d["values"])
        print(f"  {os.path.basename(f)}: {n_samples} samples")

    states = torch.cat(all_states, dim=0)
    policies = torch.cat(all_policies, dim=0)
    values = torch.cat(all_values, dim=0)
    print(f"\nTotal training samples: {states.shape[0]}")

    # Train for 10 epochs
    dataset = torch.utils.data.TensorDataset(states, policies, values)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(1, 11):
        total_loss = 0.0
        total_ploss = 0.0
        total_vloss = 0.0
        n = 0
        for s, p, v in loader:
            s = s.to(device)
            p_targets = torch.argmax(p, dim=1).long().to(device)
            v = v.to(device)

            optimizer.zero_grad()
            policy_logits, value_pred = model(s)
            p_loss = policy_criterion(policy_logits, p_targets)
            v_loss = value_criterion(value_pred, v)
            loss = p_loss + v_loss
            loss.backward()
            optimizer.step()

            bs = s.size(0)
            total_loss += loss.item() * bs
            total_ploss += p_loss.item() * bs
            total_vloss += v_loss.item() * bs
            n += bs

        avg = total_loss / n
        pavg = total_ploss / n
        vavg = total_vloss / n
        print(f"Epoch {epoch}/10: loss={avg:.4f} (policy={pavg:.4f}, value={vavg:.4f})")

    # Save updated checkpoint
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    new_iter = checkpoint.get("iteration", 26) + 1
    checkpoint["iteration"] = new_iter
    torch.save(checkpoint, str(rl_path))
    print(f"\nSaved updated checkpoint to {rl_path} (iteration {new_iter})")
    print("Done!")


if __name__ == "__main__":
    main()
