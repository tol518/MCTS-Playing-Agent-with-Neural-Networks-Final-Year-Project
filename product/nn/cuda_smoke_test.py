#!/usr/bin/env python3
"""Tiny CNN forward-pass benchmark to validate GPU throughput."""

import argparse
import sys
import time

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """Small CNN for NxN board input, output logits for board positions + pass."""

    def __init__(
        self,
        in_channels: int = 8,
        board_size: int = 9,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.board_size = board_size
        out_actions = board_size * board_size + 1  # positions + pass

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(hidden_channels * board_size * board_size, out_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h = self.conv(x)
        h = h.view(B, -1)
        return self.fc(h)


def select_device() -> torch.device:
    """Use CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tiny CNN forward-pass benchmark (GPU throughput validation)"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--channels", type=int, default=8, help="Input channels")
    parser.add_argument("--board-size", type=int, default=9, help="Board size (NxN)")
    args = parser.parse_args()

    device = select_device()
    cuda_available = torch.cuda.is_available()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"Device: {device}")
    if cuda_available:
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Falling back to CPU.")

    board_size = args.board_size
    in_channels = args.channels
    out_actions = board_size * board_size + 1

    model = TinyCNN(in_channels=in_channels, board_size=board_size).to(device)
    model.eval()
    x = torch.randn(args.batch_size, in_channels, board_size, board_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed passes
    times = []
    with torch.no_grad():
        for _ in range(args.iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    samples_per_sec = (args.batch_size * args.iters) / (sum(times) / 1000)

    print(f"\n--- Results ---")
    print(f"Batch size: {args.batch_size}, Iters: {args.iters}")
    print(f"Avg latency: {avg_ms:.3f} ms/iter")
    print(f"Throughput: {samples_per_sec:.0f} samples/sec")
    print(f"Model output size: {out_actions} logits")

    sys.exit(0)


if __name__ == "__main__":
    main()
