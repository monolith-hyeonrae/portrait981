"""Hardware-accelerated video decoder support.

Provides utilities for selecting and configuring video decoders
with optional GPU acceleration.

Supported decoders:
- auto: Automatic selection (tries GPU first, falls back to CPU)
- nvdec: NVIDIA hardware decoder (requires CUDA)
- vaapi: Intel/AMD hardware decoder (Linux)
- cpu: Software decoder (always available)

Example:
    >>> source = FileSource("video.mp4", decoder="nvdec")
    >>> source.open()
"""

import logging
import os
from enum import Enum
from typing import Optional, Tuple, List

import cv2

logger = logging.getLogger(__name__)


class DecoderType(Enum):
    """Video decoder types."""
    AUTO = "auto"
    NVDEC = "nvdec"      # NVIDIA CUDA decoder
    VAAPI = "vaapi"      # Intel/AMD VA-API (Linux)
    CPU = "cpu"          # Software decoder


# OpenCV hardware acceleration constants
# These may not exist in all OpenCV versions
try:
    HW_ACCELERATION_ANY = cv2.VIDEO_ACCELERATION_ANY
    HW_ACCELERATION_NONE = cv2.VIDEO_ACCELERATION_NONE
except AttributeError:
    HW_ACCELERATION_ANY = 0
    HW_ACCELERATION_NONE = 0

# FFmpeg codec names for hardware decoders
NVDEC_CODECS = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "h265": "hevc_cuvid",
    "vp8": "vp8_cuvid",
    "vp9": "vp9_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "mpeg2": "mpeg2_cuvid",
}

VAAPI_CODECS = {
    "h264": "h264_vaapi",
    "hevc": "hevc_vaapi",
    "h265": "hevc_vaapi",
    "vp8": "vp8_vaapi",
    "vp9": "vp9_vaapi",
}


def check_nvdec_available() -> bool:
    """Check if NVIDIA NVDEC is available."""
    try:
        # Try to detect CUDA
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible == "-1":
            return False

        # Check if nvidia-smi exists (basic NVIDIA driver check)
        import shutil
        if shutil.which("nvidia-smi") is None:
            return False

        return True
    except Exception:
        return False


def check_vaapi_available() -> bool:
    """Check if VA-API is available."""
    try:
        # Check for VA-API device
        import os
        vaapi_devices = [
            "/dev/dri/renderD128",
            "/dev/dri/renderD129",
        ]
        return any(os.path.exists(d) for d in vaapi_devices)
    except Exception:
        return False


def get_available_decoders() -> List[str]:
    """Get list of available decoders.

    Returns:
        List of decoder names (e.g., ["nvdec", "cpu"])
    """
    available = ["cpu"]  # CPU always available

    if check_nvdec_available():
        available.insert(0, "nvdec")

    if check_vaapi_available():
        available.insert(0, "vaapi")

    return available


def configure_capture(
    path: str,
    decoder: str = "auto",
) -> Tuple[cv2.VideoCapture, str]:
    """Create and configure a VideoCapture with the specified decoder.

    Args:
        path: Video file path or URL.
        decoder: Decoder type ("auto", "nvdec", "vaapi", "cpu").

    Returns:
        Tuple of (VideoCapture, actual_decoder_used)
    """
    decoder = decoder.lower()
    actual_decoder = "cpu"

    if decoder == "auto":
        # Try hardware decoders in order
        for hw_decoder in ["nvdec", "vaapi"]:
            cap, used = configure_capture(path, hw_decoder)
            if cap.isOpened():
                return cap, used
            cap.release()
        # Fall back to CPU
        decoder = "cpu"

    if decoder == "nvdec":
        if check_nvdec_available():
            cap = _create_nvdec_capture(path)
            if cap.isOpened():
                logger.info(f"Using NVDEC hardware decoder for {path}")
                return cap, "nvdec"
            cap.release()
        logger.debug("NVDEC not available, falling back to CPU")
        decoder = "cpu"

    if decoder == "vaapi":
        if check_vaapi_available():
            cap = _create_vaapi_capture(path)
            if cap.isOpened():
                logger.info(f"Using VA-API hardware decoder for {path}")
                return cap, "vaapi"
            cap.release()
        logger.debug("VA-API not available, falling back to CPU")
        decoder = "cpu"

    # CPU decoder (default)
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        logger.debug(f"Using CPU software decoder for {path}")
    return cap, "cpu"


def _create_nvdec_capture(path: str) -> cv2.VideoCapture:
    """Create VideoCapture with NVDEC hardware acceleration."""
    # Method 1: Use CAP_PROP_HW_ACCELERATION
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)

    try:
        # Try to enable hardware acceleration
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, HW_ACCELERATION_ANY)
        cap.set(cv2.CAP_PROP_HW_DEVICE, 0)  # Use first GPU
    except Exception as e:
        logger.debug(f"Failed to set HW acceleration properties: {e}")

    return cap


def _create_vaapi_capture(path: str) -> cv2.VideoCapture:
    """Create VideoCapture with VA-API hardware acceleration."""
    # Set environment for VA-API
    os.environ.setdefault("LIBVA_DRIVER_NAME", "iHD")  # Intel

    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)

    try:
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, HW_ACCELERATION_ANY)
    except Exception as e:
        logger.debug(f"Failed to set VA-API acceleration: {e}")

    return cap


def get_decoder_info() -> dict:
    """Get information about available decoders.

    Returns:
        Dict with decoder availability and details.
    """
    return {
        "available": get_available_decoders(),
        "nvdec": {
            "available": check_nvdec_available(),
            "description": "NVIDIA CUDA video decoder",
        },
        "vaapi": {
            "available": check_vaapi_available(),
            "description": "Intel/AMD VA-API decoder (Linux)",
        },
        "cpu": {
            "available": True,
            "description": "Software decoder (FFmpeg)",
        },
    }
