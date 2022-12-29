"""Building blocks of `aperol`"""

from . import identity, vv, ve, vx, ev, ee, ex, xv, xe, xx
from .identity import *
from .vv import *
from .ve import *
from .vx import *
from .ev import *
from .ee import *
from .ex import *
from .xv import *
from .xe import *
from .xx import *

__all__ = sum(
    [
        vv.__all__,
        ve.__all__,
        vx.__all__,
        ev.__all__,
        ee.__all__,
        ex.__all__,
        xv.__all__,
        xe.__all__,
        xx.__all__,
    ],
    [],
)

all_blocks = {name: globals()[name] for name in __all__}
