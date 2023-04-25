"""Building blocks of `aperol`"""

from . import identity
from . import ee, ev
from . import pe, pp, pv
from . import ve, vp, vv
from . import xe, xp
from .identity import *

from .ee import *
from .ev import *
from .pe import *
from .pp import *
from .pv import *
from .ve import *
from .vp import *
from .vv import *
from .xe import *
from .xp import *

__all__ = sum(
    [
        identity.__all__,
        ee.__all__,
        ev.__all__,
        pe.__all__,
        pp.__all__,
        pv.__all__,
        ve.__all__,
        vp.__all__,
        vv.__all__,
        xe.__all__,
        xp.__all__,
    ],
    [],
)

all_blocks = {name: globals()[name] for name in __all__}

