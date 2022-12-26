"""Identity module. """
import torch
from ..module import Module

class Identity(Module):
    def forward(
            self,
            v: torch.Tensor,
            e: torch.Tensor,
            x: torch.Tensor,
            *args, **kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return v, e, x
