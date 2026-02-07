"""Common type definitions for visualbase."""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# BGR image array type
BGRImage: TypeAlias = npt.NDArray[np.uint8]
