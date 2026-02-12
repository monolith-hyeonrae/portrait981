"""Type definitions for frame scoring."""

from typing import Callable, Optional, Tuple

from vpx.sdk import Observation

# Type alias for filter function
FilterFunc = Callable[[Optional[Observation], Optional[Observation], Optional[Observation]], Tuple[bool, str]]
