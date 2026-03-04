import torch
import torch.nn as nn
import time

class SimpleGoCNN(nn.Module):
    def __init__(self, in_channels=16, hidden_channels=64):
        super(SimpleGoCNN, self).__init__()
        # 3 Convolutional layers mimicking a simple AlphaGo-like trunk for a 9x9 board
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        # Policy head: outputs probabilities for each intersection (81) + 1 for pass
        self.policy_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.policy_flatten = nn.Flatten()
        self.policy_fc = nn.Linear(2 * 9 * 9, 9 * 9 + 1)
        
        # Value head: outputs scalar evaluating position
        self.value_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.value_flatten = nn.Flatten()
        self.value_fc1 = nn.Linear(1 * 9 * 9, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        
        # Policy head
        p = self.policy_conv(x)
        p = self.policy_flatten(p)
        p = self.policy_fc(p)
        
        # Value head
        v = self.value_conv(x)
        v = self.value_flatten(v)
        v = self.value_fc1(v)
        v = torch.relu(v)
        v = self.value_fc2(v)
        v = torch.tanh(v)
        
        return p, v

def run_throughput_test():
    # Parameters
    batch_size = 128
    channels = 16
    height = 9
    width = 9
    iterations = 100
    warmup_iterations = 10
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = SimpleGoCNN(in_channels=channels).to(device)
    model.eval()
    
    # Create dummy input batch
    dummy_input = torch.randn(batch_size, channels, height, width, device=device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Input shape: {dummy_input.shape}")
    
    # Warmup
    print(f"\nPerforming {warmup_iterations} warmup iterations...")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    # Measurement
    print(f"Running {iterations} iterations for measurement...")
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_latency = total_time / iterations
    samples_processed = batch_size * iterations
    throughput = samples_processed / total_time
    
    # Output results
    print("\n" + "=" * 40)
    print("Throughput Test Results")
    print("=" * 40)
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Iterations: {iterations}")
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Average Batch Latency: {avg_latency * 1000:.2f} ms")
    print(f"Average Sample Latency: {(avg_latency / batch_size) * 1000:.4f} ms")
    print(f"Throughput: {throughput:.2f} samples/second")
    print("=" * 40)

if __name__ == "__main__":
    run_throughput_test()
