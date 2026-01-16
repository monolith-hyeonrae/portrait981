"""OpenCV 기반 observer 백엔드."""

from __future__ import annotations

from p981.core.application.protocols.observer import ObserverEvent


class OpenCvObserverBackend:
    """OpenCV 창에 프레임을 표시하는 간단한 디버그 백엔드."""

    def __init__(
        self,
        window_name: str = "p981",
        delay_ms: int = 1,
        print_events: bool = True,
    ) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for OpenCvObserverBackend") from exc

        self._cv2 = cv2
        self._window_name = window_name
        self._delay_ms = delay_ms
        self._print_events = print_events
        self._enabled = True

    def emit(self, event: ObserverEvent) -> None:
        if not self._enabled:
            return

        payload = event.payload
        frame_path = payload.get("frame_path")
        if not isinstance(frame_path, str) or not frame_path:
            return

        frame = self._cv2.imread(frame_path)
        if frame is None:
            return

        if self._print_events:
            frame_index = payload.get("frame_index")
            timestamp_ms = payload.get("timestamp_ms")
            avg_luma = payload.get("avg_luma")
            print(
                f"[opencv] {event.kind} frame={frame_index} ts_ms={timestamp_ms} avg_luma={avg_luma}"
            )

        self._cv2.imshow(self._window_name, frame)
        key = self._cv2.waitKey(self._delay_ms) & 0xFF
        if key in (ord("q"), 27):
            self._cv2.destroyAllWindows()
            self._enabled = False
