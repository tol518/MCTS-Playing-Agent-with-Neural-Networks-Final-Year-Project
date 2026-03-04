import argparse
import random
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Support both package-style imports (product.*) and local imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_DIR = PROJECT_ROOT / "product"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PRODUCT_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCT_DIR))

try:
    from product.nn.dataset import GoDataset
    from product.nn.model import AlphaGoZeroNetwork
except ModuleNotFoundError:
    from dataset import GoDataset
    from model import AlphaGoZeroNetwork


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(data_dir: str, val_split: float, seed: int):
    """
    For an IterableDataset, we split the actual files first,
    then return two dataset instances (train and val).
    """
    import glob
    import os
    files = sorted(glob.glob(os.path.join(data_dir, '**', '*.pt'), recursive=True))
    if not files:
        raise ValueError("No .pt files found in data_dir.")
        
    random.seed(seed)
    # Shuffle files before splitting so val set is random chunks
    shuffled_files = list(files)
    random.shuffle(shuffled_files)
    
    val_count = max(1, int(len(shuffled_files) * val_split)) if len(shuffled_files) > 1 else 0
    val_files = shuffled_files[:val_count]
    train_files = shuffled_files[val_count:]
    
    if not train_files:
        val_files = []
        train_files = shuffled_files
        
    # We create two instances of the dataset and inject the specific file lists
    train_set = GoDataset(data_dir, shuffle=True)
    train_set.files = train_files
    
    # Recalculate lengths
    train_set.total_length = 0
    import torch
    for f in train_set.files:
        d = torch.load(f, map_location='cpu', weights_only=True)
        train_set.total_length += d['states'].size(0)
        del d
        
    val_set = None
    if val_files:
        val_set = GoDataset(data_dir, shuffle=False)
        val_set.files = val_files
        val_set.total_length = 0
        for f in val_set.files:
            d = torch.load(f, map_location='cpu', weights_only=True)
            val_set.total_length += d['states'].size(0)
            del d
            
    return train_set, val_set


def policy_targets_from_one_hot(policies: torch.Tensor) -> torch.Tensor:
    # policies is shape [B, 82] as one-hot (or soft labels in future).
    return torch.argmax(policies, dim=1).long()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    policy_criterion: nn.Module,
    value_criterion: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler = None,
    train_mode: str = "joint",
) -> Tuple[float, float, float, float]:
    """Returns (avg_loss, policy_acc, value_mse, avg_policy_loss)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_correct = 0
    total_samples = 0

    for states, policies, values in loader:
        states = states.to(device, non_blocking=True)
        targets = policy_targets_from_one_hot(policies).to(device, non_blocking=True)
        value_targets = values.to(device, non_blocking=True)  # shape [B, 1]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            policy_logits, value_pred = model(states)
            p_loss = policy_criterion(policy_logits, targets)
            v_loss = value_criterion(value_pred, value_targets)

            if train_mode == "joint":
                loss = p_loss + v_loss
            elif train_mode == "value":
                loss = v_loss
            else:  # policy-only
                loss = p_loss

        if is_train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = states.size(0)
        total_loss += loss.item() * batch_size
        total_policy_loss += p_loss.item() * batch_size
        total_value_loss += v_loss.item() * batch_size
        total_correct += (policy_logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_policy_acc = total_correct / total_samples
    avg_value_mse = total_value_loss / total_samples
    avg_policy_loss = total_policy_loss / total_samples
    return avg_loss, avg_policy_acc, avg_value_mse, avg_policy_loss


def main():
    parser = argparse.ArgumentParser(
        description="Week 4: joint policy + value supervised training for Go CNN."
    )
    parser.add_argument("--data-dir", type=str, default="product/nn/data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--res-blocks", type=int, default=5)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--resume", action="store_true", help="Resume from joint_last.pt (or policy_last.pt) checkpoint.")
    parser.add_argument("--train-mode", type=str, default="joint", choices=["policy", "value", "joint"],
                        help="Which heads to train: policy, value, or joint (default: joint).")
    parser.add_argument("--checkpoint-dir", type=str, default="product/nn/checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    train_set, val_set = split_dataset(args.data_dir, args.val_split, args.seed)
    print(f"Train: {len(train_set)} samples")
    if val_set:
        print(f"Val: {len(val_set)} samples")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    model = AlphaGoZeroNetwork(
        in_channels=16,
        num_res_blocks=args.res_blocks,
        channels=args.channels,
        board_size=9,
    ).to(device)

    # Select which parameters to train based on --train-mode
    if args.train_mode == "policy":
        trainable_params = [
            p for n, p in model.named_parameters()
            if not n.startswith("value_head.")
        ]
        print("Training mode: POLICY only (value head frozen)")
    elif args.train_mode == "value":
        trainable_params = [
            p for n, p in model.named_parameters()
            if not n.startswith("policy_head.")
        ]
        print("Training mode: VALUE only (policy head frozen)")
    else:  # joint
        trainable_params = list(model.parameters())
        print("Training mode: JOINT (policy + value)")

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    if args.resume:
        # Try joint checkpoint first, fall back to policy checkpoint
        resume_path = checkpoint_dir / "joint_last.pt"
        if not resume_path.exists():
            resume_path = checkpoint_dir / "policy_last.pt"
        if resume_path.exists():
            print(f"Resuming from checkpoint {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            # Only load optimizer state if the param groups match
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (ValueError, KeyError):
                print("Optimizer state mismatch (head change), using fresh optimizer.")
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"Starting at epoch {start_epoch}")
        else:
            print("No checkpoint found to resume from, starting from scratch.")

    best_val_loss = float("inf")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss, train_acc, train_mse, train_ploss = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            policy_criterion=policy_criterion,
            value_criterion=value_criterion,
            optimizer=optimizer,
            use_amp=use_amp,
            scaler=scaler,
            train_mode=args.train_mode,
        )

        if val_loader is not None:
            with torch.no_grad():
                val_loss, val_acc, val_mse, val_ploss = run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    policy_criterion=policy_criterion,
                    value_criterion=value_criterion,
                    optimizer=None,
                    use_amp=use_amp,
                    scaler=None,
                    train_mode=args.train_mode,
                )
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"loss={train_loss:.4f} p_acc={train_acc:.4f} v_mse={train_mse:.4f} | "
                f"val_loss={val_loss:.4f} val_p_acc={val_acc:.4f} val_v_mse={val_mse:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = checkpoint_dir / "joint_best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_mse": val_mse,
                        "config": vars(args),
                    },
                    best_path,
                )
        else:
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"loss={train_loss:.4f} p_acc={train_acc:.4f} v_mse={train_mse:.4f}"
            )

    final_path = checkpoint_dir / "joint_last.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
        },
        final_path,
    )
    print(f"Saved final checkpoint: {final_path}")
    if val_loader is not None:
        print(f"Best validation checkpoint: {checkpoint_dir / 'joint_best.pt'}")


if __name__ == "__main__":
    main()
