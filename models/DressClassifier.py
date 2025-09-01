import torch
from torch import nn


class DressClassifierNN(nn.Module):
    """Creates a neural network with 4 layers, one input layer of 28*28
    pixels, """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # 28 x 28 Pixels input -> 512 Neuron output
            nn.Linear(28*28, 512),
            # Sigmoid function
            nn.ReLU(),
            # Linear 512 Neurons to 512 Neurons
            nn.Linear(512, 512),
            nn.ReLU(),
            # 512 to 10 options
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x  = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu' 
    print(f"Using device: {device}")
    model = DressClassifierNN().to(device)
    print(model)

