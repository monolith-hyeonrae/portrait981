"""Timestamp conversion utilities."""

NS_PER_SECOND = 1_000_000_000


def pts_to_ns(pts_ms: float) -> int:
    """Convert PTS in milliseconds to nanoseconds.

    Args:
        pts_ms: Presentation timestamp in milliseconds (from OpenCV)

    Returns:
        Timestamp in nanoseconds
    """
    return int(pts_ms * 1_000_000)


def ns_to_pts(t_ns: int) -> float:
    """Convert nanoseconds to PTS in milliseconds.

    Args:
        t_ns: Timestamp in nanoseconds

    Returns:
        Presentation timestamp in milliseconds (for OpenCV)
    """
    return t_ns / 1_000_000


def ns_to_seconds(t_ns: int) -> float:
    """Convert nanoseconds to seconds."""
    return t_ns / NS_PER_SECOND


def seconds_to_ns(seconds: float) -> int:
    """Convert seconds to nanoseconds."""
    return int(seconds * NS_PER_SECOND)
