"""High-level API for momentscan.

    >>> import momentscan as ms
    >>> result = ms.run("video.mp4")

Version routing:
    v1: Legacy — BatchHighlightEngine + CollectionEngine (기존)
    v2: Simplified — visualpath FlowGraph + visualbind (신규)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

logger = logging.getLogger(__name__)

DEFAULT_FPS = 10
DEFAULT_BACKEND = "simple"
DEFAULT_VERSION = "v2"


def run(
    video: Union[str, Path],
    *,
    version: str = DEFAULT_VERSION,
    # v2 params
    expression_model: Optional[str] = None,
    pose_model: Optional[str] = None,
    fps: int = DEFAULT_FPS,
    top_k: int = 10,
    # v1 params (legacy)
    analyzers: Optional[Sequence[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 1,
    backend: str = DEFAULT_BACKEND,
    profile: Optional[str] = None,
    isolation: Optional[Any] = None,
    on_frame: Optional[Any] = None,
    collection_path: Optional[Union[str, Path]] = None,
    member_id: Optional[str] = None,
    bind_model: Optional[Union[str, Path]] = None,
) -> Any:
    """Process a video.

    Args:
        video: Path to video file.
        version: "v1" (legacy) or "v2" (simplified). Default: v2.
        expression_model: v2 expression model path.
        pose_model: v2 pose model path.
        fps: Frames per second.
        top_k: v2 top K frames to select.
        **kwargs: v1 legacy parameters.

    Returns:
        v1: Result with highlights, collection.
        v2: list[FrameResult].
    """
    if version == "v2":
        from momentscan.v2 import MomentscanV2
        app = MomentscanV2(
            expression_model=expression_model,
            pose_model=pose_model,
        )
        return app.run(video, fps=fps)

    else:
        # v1 legacy
        from momentscan.v1 import MomentscanApp, Result
        app = MomentscanApp(
            analyzers=analyzers, output_dir=output_dir,
            collection_path=str(collection_path) if collection_path else None,
            member_id=member_id,
            bind_model=str(bind_model) if bind_model else None,
        )
        return app.run(
            video,
            modules=list(analyzers) if analyzers else None,
            fps=fps,
            batch_size=batch_size,
            backend=backend,
            isolation=isolation,
            profile=profile,
            on_frame=on_frame,
        )


# Re-exports for backward compatibility
from momentscan.v1 import MomentscanApp, Result  # noqa: E402, F401
