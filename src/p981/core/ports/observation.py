from __future__ import annotations

from typing import Protocol, _GenericAlias

from loguru import logger

from ..common import ObservationEvent


class ObservationPort(Protocol):
    def emit(self, event: ObservationEvent) -> None:
        """Handle observation events from core domains."""


class NoopObservationPort:
    def emit(self, event: ObservationEvent) -> None:
        return


class LoguruObservationPort:
    def emit(self, event: ObservationEvent) -> None:
        payload = event.payload
        frame_index = payload.get("frame_index")
        timestamp_ms = payload.get("timestamp_ms")
        avg_luma = payload.get("avg_luma")
        logger.info(
            "obs {} | frame={} ts_ms={} avg_luma={}",
            event.kind,
            frame_index,
            timestamp_ms,
            avg_luma,
        )


class _WarningObservationPort:
    def __init__(self, name: str) -> None:
        self._name = name
        self._warned = False

    def emit(self, event: ObservationEvent) -> None:
        if not self._warned:
            logger.warning("{} observation adapter is not implemented yet.", self._name)
            self._warned = True


class PixeltableObservationPort:
    def __init__(self, table_name: str = "p981_observations") -> None:
        self._table_name = table_name
        self._pxt = self._load_pixeltable()
        self._table = self._get_or_create_table()

    def emit(self, event: ObservationEvent) -> None:
        row = self._event_to_row(event)
        self._table.insert([row])

    def _load_pixeltable(self):
        try:
            import pixeltable as pxt  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pixeltable is required for PixeltableObservationPort") from exc
        return pxt

    def _resolve_type(self, names: list[str], default: object) -> object:
        for name in names:
            attr = getattr(self._pxt, name, None)
            if attr is None:
                continue
            if isinstance(attr, (_GenericAlias, type)):
                return attr
            try:
                return attr()
            except TypeError:
                return attr
        return default

    def _schema(self) -> dict[str, object]:
        string_t = self._resolve_type(["String"], str)
        int_t = self._resolve_type(["Int"], int)
        float_t = self._resolve_type(["Float"], float)
        json_t = self._resolve_type(["Json"], str)
        opt_string_t = self._optional(string_t)
        opt_int_t = self._optional(int_t)
        opt_float_t = self._optional(float_t)
        return {
            "kind": string_t,
            "video_ref": opt_string_t,
            "frame_index": opt_int_t,
            "timestamp_ms": opt_int_t,
            "avg_luma": opt_float_t,
            "frame_path": opt_string_t,
            "asset_ref": opt_string_t,
            "tags": json_t,
            "payload": json_t,
        }

    def _get_or_create_table(self):
        create_table = getattr(self._pxt, "create_table", None)
        if callable(create_table):
            return create_table(
                self._table_name,
                self._schema(),
                if_exists="replace_force",
            )
        raise RuntimeError("Unsupported pixeltable API; missing create_table/get_table")

    def _event_to_row(self, event: ObservationEvent) -> dict[str, object]:
        payload = dict(event.payload)
        tags = event.tags or {}
        timestamp_ms = event.timestamp_ms
        if timestamp_ms is None:
            timestamp_ms = payload.get("timestamp_ms")
        return {
            "kind": event.kind,
            "video_ref": payload.get("video_ref"),
            "frame_index": payload.get("frame_index"),
            "timestamp_ms": timestamp_ms,
            "avg_luma": payload.get("avg_luma"),
            "frame_path": payload.get("frame_path"),
            "asset_ref": event.asset_ref,
            "tags": tags or {},
            "payload": payload,
        }

    @staticmethod
    def _optional(base_type: object) -> object:
        try:
            return base_type | None
        except TypeError:
            return base_type


class RerunObservationPort(_WarningObservationPort):
    def __init__(self) -> None:
        super().__init__("rerun")


class MultiObservationPort:
    def __init__(self, ports: list[ObservationPort]) -> None:
        self._ports = ports

    def emit(self, event: ObservationEvent) -> None:
        for port in self._ports:
            try:
                port.emit(event)
            except Exception as exc:
                logger.warning("ObservationPort emit failed: {}", exc)
