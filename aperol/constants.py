"""Constants fixed in architecture search. """

MAX_IN = MAX_OUT = 16
MIN_IN = MIN_OUT = 8
MAX_DEPTH = 32
NUM_BASIS = 50
CUTOFF_LOWER = 0.0
CUTOFF_UPPER = 5.0
import torch; ACTIVATION = torch.nn.SiLU()
