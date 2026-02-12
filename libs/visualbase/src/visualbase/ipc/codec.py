"""Frame codec for IPC transmission.

Provides encode/decode functions for serializing Frame objects
to JSON-compatible dicts using JPEG compression + base64 encoding.
"""

import base64
from typing import Any, Dict

import numpy as np


def encode_frame(frame: "Frame", jpeg_quality: int = 95) -> Dict[str, Any]:
    """Encode a Frame to a JSON-serializable dict.

    Uses JPEG compression for efficient transmission.

    Args:
        frame: Frame to encode.
        jpeg_quality: JPEG compression quality (0-100).

    Returns:
        Dict with keys: frame_id, t_src_ns, width, height, data_b64.
    """
    import cv2

    _, jpeg_data = cv2.imencode(
        ".jpg", frame.data,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )

    return {
        "frame_id": frame.frame_id,
        "t_src_ns": frame.t_src_ns,
        "width": frame.width,
        "height": frame.height,
        "data_b64": base64.b64encode(jpeg_data.tobytes()).decode("ascii"),
    }


def decode_frame(data: Dict[str, Any]) -> "Frame":
    """Decode a dict back into a Frame object.

    Args:
        data: Dict produced by encode_frame().

    Returns:
        Reconstructed Frame.

    Raises:
        ValueError: If image data cannot be decoded.
    """
    import cv2
    from visualbase import Frame

    jpeg_bytes = base64.b64decode(data["data_b64"])
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode frame image data")

    return Frame.from_array(
        img,
        frame_id=data["frame_id"],
        t_src_ns=data["t_src_ns"],
    )
