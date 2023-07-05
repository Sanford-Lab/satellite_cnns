

import torch
from typing import AnyType

class Normalization(torch.nn.Module):
    """Preprocessing normalization layer with z-score."""

    def __init__(self, mean: AnyType, std: AnyType) -> None:
        super().__init__()
        self.mean = torch.nn.Parameter(torch.as_tensor(mean))
        self.std = torch.nn.Parameter(torch.as_tensor(std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
    
    
class MoveDim(torch.nn.Module):
    """Moves a dimension axis to another position."""

    def __init__(self, src: int, dest: int) -> None:
        super().__init__()
        self.src = src
        self.dest = dest

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.moveaxis(self.src, self.dest)