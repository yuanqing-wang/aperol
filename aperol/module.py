import torch

class Module(torch.nn.Module):
    """Base module for `aperol` building blocks. """
    def __init__(self, *args, **kwargs):
        super().__init__()

    
