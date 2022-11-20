import numpy as np
import torch
import torch.nn as nn


def _to_tensor(x: torch.Tensor):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x


def _to_4d(x: torch.Tensor):
    if not x.ndim == 4:
        x = torch.unsqueeze(x, 0)
    return x


# class Teacher(nn.Module):
#     def __init__(self, model: nn.Module) -> None:
#         super().__init__()
#         self.model = model

#     def forward(self, x):
#         return self.model(x)


def teacher_function(model: nn.Module, x: np.array):
    x = _to_tensor(x)
    x = _to_4d(x)
    with torch.no_grad():
        out = model(x)
    # print(out.shape)
    return out
