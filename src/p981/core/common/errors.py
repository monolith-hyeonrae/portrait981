class StageError(Exception):
    """Base error for stage orchestration issues."""


class StageValidationError(StageError):
    """Raised when stage inputs fail validation."""
