"""Shared serialization utilities for IPC communication.

Provides JSON-safe serialization and Observation (de)serialization
used by both the worker subprocess and the launcher (main process).
"""

import json
from typing import Any, Dict, List, Optional

from visualpath.core.extractor import Observation


# ---------------------------------------------------------------------------
# AttrDict: transparent dict→attribute bridge for deserialized data
# ---------------------------------------------------------------------------

class _AttrDict(dict):
    """Dict subclass providing attribute access for deserialized JSON data.

    After ZMQ round-trip, dataclass objects (``FaceObservation``,
    ``FaceDetectOutput``, …) become plain dicts.  Code written for
    those objects uses attribute access (``face.bbox``, ``data.faces``).
    Wrapping dicts in ``_AttrDict`` makes both ``d.key`` and ``d["key"]``
    work transparently, so the rest of the codebase requires no changes.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _wrap_value(value: Any) -> Any:
    """Recursively wrap dicts as ``_AttrDict`` for attribute access."""
    if isinstance(value, dict):
        return _AttrDict({k: _wrap_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap_value(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Serialize helpers
# ---------------------------------------------------------------------------

def serialize_value(value: Any) -> Any:
    """Recursively serialize a value for JSON transmission.

    Handles dataclasses, lists, tuples, dicts, and objects with __dict__.
    Falls back to repr() or str() for unserializable types.

    Args:
        value: Any value to serialize.

    Returns:
        JSON-serializable representation.
    """
    if value is None:
        return None

    # Fast path: already JSON-serializable
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        pass

    # Handle dataclasses
    if hasattr(value, "__dataclass_fields__"):
        return {
            k: serialize_value(getattr(value, k))
            for k in value.__dataclass_fields__
        }

    # Handle lists/tuples
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]

    # Handle dicts
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}

    # Handle objects with __dict__
    if hasattr(value, "__dict__"):
        return repr(value)

    return str(value)


def serialize_observation(obs: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Serialize an Observation for ZMQ transmission.

    Works with any object that has the standard Observation attributes
    (source, frame_id, t_ns, signals, metadata, timing, data, faces).

    Args:
        obs: Observation to serialize.

    Returns:
        JSON-serializable dict, or None if obs is None.
    """
    if obs is None:
        return None

    result: Dict[str, Any] = {
        "source": getattr(obs, "source", "unknown"),
        "frame_id": getattr(obs, "frame_id", -1),
        "t_ns": getattr(obs, "t_ns", 0),
        "signals": getattr(obs, "signals", {}),
        "metadata": getattr(obs, "metadata", {}),
        "timing": getattr(obs, "timing", None),
    }

    if hasattr(obs, "data") and obs.data is not None:
        result["data"] = serialize_value(obs.data)

    # faces field (extractors-base Observation has this, core does not)
    faces = getattr(obs, "faces", [])
    if faces:
        result["faces"] = serialize_value(faces)

    return result


def deserialize_observation(data: Optional[Dict[str, Any]]) -> Optional[Observation]:
    """Deserialize an Observation from a ZMQ message dict.

    Nested dicts are wrapped in ``_AttrDict`` so that attribute access
    (``obs.data.faces``, ``face.bbox``) works the same as on the
    original typed objects.

    Args:
        data: Dict containing serialized observation data.

    Returns:
        Reconstructed Observation object, or None.
    """
    if data is None:
        return None

    # Wrap data so nested attribute access works (e.g. obs.data.faces[0].bbox)
    obs_data = data.get("data")
    if obs_data is not None:
        obs_data = _wrap_value(obs_data)

    obs = Observation(
        source=data["source"],
        frame_id=data["frame_id"],
        t_ns=data["t_ns"],
        signals=data.get("signals", {}),
        data=obs_data,
        metadata=data.get("metadata", {}),
        timing=data.get("timing"),
    )

    # Attach faces (extractors-base Observation declares this field;
    # core Observation does not, but Python allows extra attributes).
    faces_raw = data.get("faces", [])
    obs.faces = [_wrap_value(f) for f in faces_raw] if faces_raw else []

    return obs


__all__ = [
    "serialize_value",
    "serialize_observation",
    "deserialize_observation",
]
