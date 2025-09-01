import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib
import matplotlib.pyplot as plt

def display_dataset(d: DataLoader, figsize =(8,8), cols=4, rows=4):
    figure = plt.figure(figsize=figsize)
    for i in range(1, cols * rows + 1):
        # returns a tensor of random integers from low (inclusive) to high (exclusive)
        # size is the shape of the output tensor, here a single integer
        sample_idx = torch.randint(len(d), size=(1,)).item() # Why did we use torch here?
        img, label = d[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show() 

def main():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    display_dataset(training_data)

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available. Running on GPU.")
    else:
        print("CUDA is not available. Running on CPU.")
    main()