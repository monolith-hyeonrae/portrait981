"""Error policy for module execution.

ErrorPolicy controls retry, timeout, and error-handling behavior
for individual modules. Attached to a module as a class attribute.

Example:
    >>> class SlowDetector(Module):
    ...     error_policy = ErrorPolicy(
    ...         max_retries=2,
    ...         timeout_sec=5.0,
    ...         on_timeout="skip",
    ...     )
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ErrorPolicy:
    """Declarative error-handling policy for a module.

    Attributes:
        max_retries: Number of retries on error (0 = no retry).
        timeout_sec: Maximum processing time in seconds (0 = no timeout).
        on_timeout: Action on timeout: "skip", "raise", or "fallback".
        on_error: Action on error: "skip", "raise", or "fallback".
        fallback_signals: Default signals to return in "fallback" mode.
    """

    max_retries: int = 0
    timeout_sec: float = 0.0
    on_timeout: str = "skip"
    on_error: str = "skip"
    fallback_signals: Dict[str, Any] = field(default_factory=dict)


__all__ = ["ErrorPolicy"]
