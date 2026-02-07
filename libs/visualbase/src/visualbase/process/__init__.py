"""Process module for A-B*-C architecture.

This module provides the Ingest process (A module) that:
- Captures video from camera/RTSP
- Maintains a RingBuffer for clip extraction
- Distributes proxy streams to extractors
- Receives TRIG messages and extracts clips
"""

from visualbase.process.ingest import IngestProcess

__all__ = [
    "IngestProcess",
]
