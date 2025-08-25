import torch


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available. Running on GPU.")
    else:
        print("CUDA is not available. Running on CPU.")