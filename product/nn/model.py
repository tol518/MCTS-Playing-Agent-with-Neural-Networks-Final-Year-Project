import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Standard Pre-Activation / Post-Activation Residual Block.
    Using Post-Activation (Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU)
    as standard in AlphaGo Zero.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaGoZeroNetwork(nn.Module):
    """
    AlphaGo Zero style Convolutional Neural Network for Go (9x9).
    
    Architecture:
    - Shared Convolutional Trunk (Initial Conv + N Residual Blocks)
    - Policy Head: Outputs raw logits for 82 actions (81 board positions + 1 pass)
    - Value Head: Outputs a single scalar in [-1, 1] representing the expected outcome of the game.
    """
    def __init__(self, in_channels: int = 16, num_res_blocks: int = 5, channels: int = 128, board_size: int = 9):
        super().__init__()
        self.board_size = board_size
        self.action_space = board_size * board_size + 1  # 81 + 1 (pass)
        
        # Initial convolutional block
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        # Residual blocks (the "trunk")
        self.res_blocks = nn.ModuleList([
            ResBlock(channels) for _ in range(num_res_blocks)
        ])
        
        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, self.action_space)
            # Raw logits are returned. 
            # Apply F.log_softmax(logits, dim=1) if log probabilities are needed,
            # or use nn.CrossEntropyLoss(logits, targets) during training.
        )
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, board_size, board_size)
            
        Returns:
            tuple containing:
                - policy_logits (torch.Tensor): Policy head output of shape (batch_size, 82)
                - value (torch.Tensor): Value head output of shape (batch_size, 1) in range [-1, 1]
        """
        # Shared trunk
        x = self.initial_conv(x)
        for block in self.res_blocks:
            x = block(x)
            
        # Branching into two heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value
