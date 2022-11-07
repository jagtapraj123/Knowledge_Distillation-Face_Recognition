import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        # Initiate model here
        pass
        # You can club layers in nn.Sequential
        # Also, you can have multiple nn.Sequential/layers and connect them in forward function

    def forward(self, x):
        # connect model layers
        # consider layers as functions
        pass
        # return last output
