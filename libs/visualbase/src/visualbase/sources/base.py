"""Base source abstract class."""

from abc import ABC, abstractmethod
from typing import Optional

from visualbase.core.frame import Frame


class BaseSource(ABC):
    """Abstract base class for video sources.

    All video sources must implement this interface to be compatible
    with VisualBase.
    """

    @abstractmethod
    def open(self) -> None:
        """Open the video source.

        Raises:
            IOError: If the source cannot be opened.
        """
        ...

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Read the next frame from the source.

        Returns:
            Frame object if available, None if end of stream.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the video source and release resources."""
        ...

    @abstractmethod
    def seek(self, t_ns: int) -> bool:
        """Seek to a specific timestamp.

        Args:
            t_ns: Target timestamp in nanoseconds.

        Returns:
            True if seek was successful, False otherwise.
        """
        ...

    @property
    @abstractmethod
    def is_seekable(self) -> bool:
        """Whether this source supports seeking."""
        ...

    @property
    @abstractmethod
    def fps(self) -> float:
        """Original frames per second of the source."""
        ...

    @property
    @abstractmethod
    def width(self) -> int:
        """Original frame width in pixels."""
        ...

    @property
    @abstractmethod
    def height(self) -> int:
        """Original frame height in pixels."""
        ...

    def __enter__(self) -> "BaseSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
