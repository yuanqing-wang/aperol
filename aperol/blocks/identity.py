"""Identity module. """
import torch
from ..module import Block

__all__ = ["Identity"]

class Identity(Block):
    def forward(
            self,
            v: torch.Tensor,
            e: torch.Tensor,
            x: torch.Tensor,
            *args, **kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return v, e, x
