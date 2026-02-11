"""Gesture analyzer for hand gesture recognition."""

from __future__ import annotations

from typing import Optional, List, Dict, TYPE_CHECKING
import logging

import numpy as np

if TYPE_CHECKING:
    from visualbase import Frame

from vpx.sdk import (
    Module,
    Observation,
    ProcessingStep,
    processing_step,
    get_processing_steps,
    Capability,
    ModuleCapabilities,
)
from vpx.hand_gesture.types import GestureType, HandLandmarkIndex
from vpx.hand_gesture.output import GestureOutput
from vpx.hand_gesture.backends.base import HandLandmarkBackend, HandLandmarks

logger = logging.getLogger(__name__)


class GestureAnalyzer(Module):
    """Analyzer for hand gesture recognition using MediaPipe Hands.

    Detects specific hand gestures useful for gokart scenario:
    - V-sign (peace sign)
    - Thumbs up
    - OK sign
    - Open palm
    - Fist
    - Pointing

    Features:
    - Uses MediaPipe Hands for 21-landmark hand detection
    - Rule-based gesture classification
    - Per-hand gesture detection
    - Confidence scoring

    Args:
        hand_backend: Hand landmark backend (default: MediaPipeHandsBackend).
        device: Device for inference (default: "cpu").
        min_gesture_confidence: Minimum confidence for gesture (default: 0.6).

    Example:
        >>> analyzer = GestureAnalyzer()
        >>> with analyzer:
        ...     obs = analyzer.process(frame)
        ...     if obs.signals.get("gesture_detected", 0) > 0:
        ...         print(f"Gesture: {obs.metadata.get('gesture_type')}")
    """

    def __init__(
        self,
        hand_backend: Optional[HandLandmarkBackend] = None,
        device: str = "cpu",
        min_gesture_confidence: float = 0.6,
    ):
        self._device = device
        self._hand_backend = hand_backend
        self._min_gesture_confidence = min_gesture_confidence
        self._initialized = False

        # Step timing tracking (auto-populated by @processing_step decorator)
        self._step_timings: Optional[Dict[str, float]] = None

    @property
    def name(self) -> str:
        return "hand.gesture"

    @property
    def capabilities(self) -> ModuleCapabilities:
        return ModuleCapabilities(
            flags=Capability.STATEFUL,
            gpu_memory_mb=0,
            init_time_sec=1.0,
            required_extras=frozenset({"vpx-hand-gesture"}),
        )

    @property
    def processing_steps(self) -> List[ProcessingStep]:
        """Get the list of internal processing steps (auto-extracted from decorators)."""
        return get_processing_steps(self)

    def initialize(self) -> None:
        """Initialize hand landmark backend."""
        if self._initialized:
            return

        if self._hand_backend is None:
            from vpx.hand_gesture.backends.mediapipe_hands import (
                MediaPipeHandsBackend,
            )

            self._hand_backend = MediaPipeHandsBackend()

        self._hand_backend.initialize(self._device)
        self._initialized = True
        logger.info("GestureAnalyzer initialized")

    def cleanup(self) -> None:
        """Release backend resources."""
        if self._hand_backend is not None:
            self._hand_backend.cleanup()

        self._initialized = False
        logger.info("GestureAnalyzer cleaned up")

    # ========== Processing Steps (decorated methods) ==========

    @processing_step(
        name="hand_detection",
        description="Detect hands with 21 landmarks each",
        backend="MediaPipeHandsBackend",
        input_type="Frame (BGR image)",
        output_type="List[HandLandmarks]",
    )
    def _detect_hands(self, image) -> List:
        """Detect hands using backend."""
        return self._hand_backend.detect(image)

    @processing_step(
        name="gesture_classification",
        description="Classify gesture for each hand (V-sign, thumbs up, OK, etc.)",
        backend="Rule-based classifier",
        input_type="List[HandLandmarks]",
        output_type="List[GestureResult]",
        depends_on=["hand_detection"],
    )
    def _classify_gestures(self, hands: List) -> List[Dict]:
        """Classify gestures for all hands."""
        all_gestures = []
        for hand in hands:
            gesture, confidence = self._classify_gesture(hand)
            all_gestures.append({
                "handedness": hand.handedness,
                "gesture": gesture,
                "confidence": confidence,
            })
        return all_gestures

    @processing_step(
        name="aggregation",
        description="Aggregate gesture signals for all hands",
        input_type="List[GestureResult]",
        output_type="Dict (signals, metadata)",
        depends_on=["gesture_classification"],
    )
    def _aggregate_gestures(self, hands: List, all_gestures: List[Dict], image_size: tuple) -> Dict:
        """Aggregate gesture results into signals."""
        w, h = image_size

        # Find best gesture
        best_gesture = GestureType.NONE
        best_confidence = 0.0

        for g in all_gestures:
            if g["confidence"] > best_confidence:
                best_gesture = g["gesture"]
                best_confidence = g["confidence"]

        gesture_detected = (
            1.0
            if best_gesture != GestureType.NONE
            and best_confidence >= self._min_gesture_confidence
            else 0.0
        )

        signals = {
            "hand_count": float(len(hands)),
            "gesture_detected": gesture_detected,
            "gesture_confidence": best_confidence if gesture_detected else 0.0,
        }

        # Add per-gesture type signals
        for gesture_type in GestureType:
            if gesture_type != GestureType.NONE:
                signals[f"gesture_{gesture_type.value}"] = (
                    1.0 if best_gesture == gesture_type else 0.0
                )

        # Build hand landmarks data for visualization
        hand_landmarks_data = []
        for i, hand in enumerate(hands):
            hand_landmarks_data.append({
                "handedness": hand.handedness,
                "landmarks": hand.landmarks.tolist(),
                "confidence": hand.confidence,
                "gesture": all_gestures[i]["gesture"].value if i < len(all_gestures) else "",
                "gesture_confidence": all_gestures[i]["confidence"] if i < len(all_gestures) else 0.0,
                "image_size": (w, h),
            })

        return {
            "signals": signals,
            "hand_landmarks_data": hand_landmarks_data,
            "best_gesture": best_gesture,
            "gesture_detected": gesture_detected > 0,
            "all_gestures": [
                {"handedness": g["handedness"], "gesture": g["gesture"].value, "confidence": g["confidence"]}
                for g in all_gestures
            ],
        }

    # MediaPipe 21-point hand connections
    _HAND_CONNECTIONS = (
        # Thumb
        (HandLandmarkIndex.WRIST, HandLandmarkIndex.THUMB_CMC),
        (HandLandmarkIndex.THUMB_CMC, HandLandmarkIndex.THUMB_MCP),
        (HandLandmarkIndex.THUMB_MCP, HandLandmarkIndex.THUMB_IP),
        (HandLandmarkIndex.THUMB_IP, HandLandmarkIndex.THUMB_TIP),
        # Index finger
        (HandLandmarkIndex.WRIST, HandLandmarkIndex.INDEX_FINGER_MCP),
        (HandLandmarkIndex.INDEX_FINGER_MCP, HandLandmarkIndex.INDEX_FINGER_PIP),
        (HandLandmarkIndex.INDEX_FINGER_PIP, HandLandmarkIndex.INDEX_FINGER_DIP),
        (HandLandmarkIndex.INDEX_FINGER_DIP, HandLandmarkIndex.INDEX_FINGER_TIP),
        # Middle finger
        (HandLandmarkIndex.WRIST, HandLandmarkIndex.MIDDLE_FINGER_MCP),
        (HandLandmarkIndex.MIDDLE_FINGER_MCP, HandLandmarkIndex.MIDDLE_FINGER_PIP),
        (HandLandmarkIndex.MIDDLE_FINGER_PIP, HandLandmarkIndex.MIDDLE_FINGER_DIP),
        (HandLandmarkIndex.MIDDLE_FINGER_DIP, HandLandmarkIndex.MIDDLE_FINGER_TIP),
        # Ring finger
        (HandLandmarkIndex.WRIST, HandLandmarkIndex.RING_FINGER_MCP),
        (HandLandmarkIndex.RING_FINGER_MCP, HandLandmarkIndex.RING_FINGER_PIP),
        (HandLandmarkIndex.RING_FINGER_PIP, HandLandmarkIndex.RING_FINGER_DIP),
        (HandLandmarkIndex.RING_FINGER_DIP, HandLandmarkIndex.RING_FINGER_TIP),
        # Pinky
        (HandLandmarkIndex.WRIST, HandLandmarkIndex.PINKY_MCP),
        (HandLandmarkIndex.PINKY_MCP, HandLandmarkIndex.PINKY_PIP),
        (HandLandmarkIndex.PINKY_PIP, HandLandmarkIndex.PINKY_DIP),
        (HandLandmarkIndex.PINKY_DIP, HandLandmarkIndex.PINKY_TIP),
        # Palm connections
        (HandLandmarkIndex.INDEX_FINGER_MCP, HandLandmarkIndex.MIDDLE_FINGER_MCP),
        (HandLandmarkIndex.MIDDLE_FINGER_MCP, HandLandmarkIndex.RING_FINGER_MCP),
        (HandLandmarkIndex.RING_FINGER_MCP, HandLandmarkIndex.PINKY_MCP),
    )

    def annotate(self, obs):
        """Return KeypointsMark + LabelMark for each detected hand."""
        if obs is None or obs.data is None:
            return []
        from vpx.sdk.marks import KeypointsMark, LabelMark

        if hasattr(obs.data, "hand_landmarks"):
            hand_landmarks = obs.data.hand_landmarks
        elif isinstance(obs.data, dict) and "hand_landmarks" in obs.data:
            hand_landmarks = obs.data["hand_landmarks"]
        else:
            return []

        marks = []
        for hand in hand_landmarks:
            landmarks = hand.get("landmarks", [])
            if len(landmarks) < 21:
                continue
            marks.append(KeypointsMark(
                points=tuple(tuple(l) for l in landmarks),
                connections=self._HAND_CONNECTIONS,
                normalized=True,
                line_color=(200, 100, 0),
                point_radius=3,
            ))
            gesture = hand.get("gesture", "none")
            if gesture and gesture != "none":
                wrist = landmarks[HandLandmarkIndex.WRIST]
                handedness = hand.get("handedness", "")
                prefix = handedness[0] if handedness else "?"
                marks.append(LabelMark(
                    text=f"{prefix}:{gesture}",
                    x=wrist[0], y=wrist[1] + 0.03,
                    normalized=True,
                    color=(0, 255, 0),
                    background=(40, 40, 40),
                ))
        return marks

    # ========== Main process method ==========

    def process(self, frame: Frame, deps=None) -> Optional[Observation]:
        """Extract gesture observations from a frame.

        Args:
            frame: Input frame to analyze.
            deps: Not used (no dependencies).

        Returns:
            Observation with gesture signals.
        """
        if not self._initialized or self._hand_backend is None:
            raise RuntimeError("Analyzer not initialized. Call initialize() first.")

        image = frame.data
        h, w = image.shape[:2]
        t_ns = frame.t_src_ns

        # Enable step timing collection
        self._step_timings = {}

        # Execute processing steps (timing auto-tracked by decorators)
        hands = self._detect_hands(image)

        if not hands:
            # Collect timing data
            timing = self._step_timings.copy() if self._step_timings else None
            self._step_timings = None

            return Observation(
                source=self.name,
                frame_id=frame.frame_id,
                t_ns=t_ns,
                signals={
                    "hand_count": 0,
                    "gesture_detected": 0.0,
                    "gesture_confidence": 0.0,
                },
                data=GestureOutput(gestures=[], hand_landmarks=[]),
                metadata={
                    "gesture_type": "",
                    "hands_detected": 0,
                    "_metrics": {
                        "hands_detected": 0,
                        "gestures_recognized": 0,
                    },
                },
                timing=timing,
            )

        # Execute classification and aggregation steps
        all_gestures = self._classify_gestures(hands)
        result = self._aggregate_gestures(hands, all_gestures, (w, h))

        # Collect timing data
        timing = self._step_timings.copy() if self._step_timings else None
        self._step_timings = None

        gestures_recognized = sum(
            1 for g in result["all_gestures"] if g["gesture"] != "none"
        )

        return Observation(
            source=self.name,
            frame_id=frame.frame_id,
            t_ns=t_ns,
            signals=result["signals"],
            data=GestureOutput(
                gestures=[g["gesture"] for g in result["all_gestures"]],
                hand_landmarks=result["hand_landmarks_data"],
            ),
            metadata={
                "gesture_type": result["best_gesture"].value if result["gesture_detected"] else "",
                "hands_detected": len(hands),
                "all_gestures": result["all_gestures"],
                "_metrics": {
                    "hands_detected": len(hands),
                    "gestures_recognized": gestures_recognized,
                },
            },
            timing=timing,
        )

    def _classify_gesture(self, hand: HandLandmarks) -> tuple[GestureType, float]:
        """Classify gesture from hand landmarks.

        Args:
            hand: Hand landmarks from detection.

        Returns:
            Tuple of (gesture_type, confidence).
        """
        landmarks = hand.landmarks  # Shape: (21, 3)

        # Get finger states (up/down)
        fingers_up = self._get_fingers_up(landmarks, hand.handedness)

        # Classify based on finger states
        gesture, confidence = self._classify_from_fingers(fingers_up, landmarks)

        # Combine with hand detection confidence
        final_confidence = confidence * hand.confidence

        return gesture, final_confidence

    def _get_fingers_up(
        self, landmarks: np.ndarray, handedness: str
    ) -> List[bool]:
        """Determine which fingers are extended.

        Args:
            landmarks: Hand landmarks array (21, 3).
            handedness: "Left" or "Right".

        Returns:
            List of 5 booleans [thumb, index, middle, ring, pinky].
        """
        fingers_up = []

        # Thumb: Compare tip x to IP joint x
        # For right hand: thumb up if tip.x > ip.x
        # For left hand: thumb up if tip.x < ip.x
        thumb_tip = landmarks[HandLandmarkIndex.THUMB_TIP]
        thumb_ip = landmarks[HandLandmarkIndex.THUMB_IP]

        if handedness == "Right":
            thumb_up = thumb_tip[0] > thumb_ip[0]
        else:
            thumb_up = thumb_tip[0] < thumb_ip[0]
        fingers_up.append(thumb_up)

        # Other fingers: Compare tip y to PIP joint y
        # Finger is up if tip.y < pip.y (y increases downward)
        finger_tips = [
            HandLandmarkIndex.INDEX_FINGER_TIP,
            HandLandmarkIndex.MIDDLE_FINGER_TIP,
            HandLandmarkIndex.RING_FINGER_TIP,
            HandLandmarkIndex.PINKY_TIP,
        ]
        finger_pips = [
            HandLandmarkIndex.INDEX_FINGER_PIP,
            HandLandmarkIndex.MIDDLE_FINGER_PIP,
            HandLandmarkIndex.RING_FINGER_PIP,
            HandLandmarkIndex.PINKY_PIP,
        ]

        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            finger_up = tip[1] < pip[1]
            fingers_up.append(finger_up)

        return fingers_up

    def _classify_from_fingers(
        self, fingers_up: List[bool], landmarks: np.ndarray
    ) -> tuple[GestureType, float]:
        """Classify gesture from finger states.

        Args:
            fingers_up: List of finger up states [thumb, index, middle, ring, pinky].
            landmarks: Hand landmarks for additional checks.

        Returns:
            Tuple of (gesture_type, confidence).
        """
        thumb, index, middle, ring, pinky = fingers_up

        # V-sign: Index and middle up, others down
        if not thumb and index and middle and not ring and not pinky:
            return GestureType.V_SIGN, 0.9

        # Thumbs up: Only thumb up
        if thumb and not index and not middle and not ring and not pinky:
            return GestureType.THUMBS_UP, 0.9

        # Open palm: All fingers up
        if all(fingers_up):
            return GestureType.OPEN_PALM, 0.85

        # Fist: All fingers down
        if not any(fingers_up):
            return GestureType.FIST, 0.85

        # Pointing: Only index up
        if not thumb and index and not middle and not ring and not pinky:
            return GestureType.POINTING, 0.85

        # OK sign: Thumb and index touching, others up
        # Check distance between thumb tip and index tip
        if self._is_ok_sign(landmarks, fingers_up):
            return GestureType.OK_SIGN, 0.8

        return GestureType.NONE, 0.0

    def _is_ok_sign(self, landmarks: np.ndarray, fingers_up: List[bool]) -> bool:
        """Check if hand is making OK sign.

        OK sign: Thumb and index tips close together, middle/ring/pinky up.

        Args:
            landmarks: Hand landmarks.
            fingers_up: Finger up states.

        Returns:
            True if OK sign detected.
        """
        thumb, index, middle, ring, pinky = fingers_up

        # Middle, ring, pinky should be up
        if not (middle and ring and pinky):
            return False

        # Thumb and index tips should be close
        thumb_tip = landmarks[HandLandmarkIndex.THUMB_TIP]
        index_tip = landmarks[HandLandmarkIndex.INDEX_FINGER_TIP]

        # Calculate distance (normalized coordinates)
        distance = np.linalg.norm(thumb_tip[:2] - index_tip[:2])

        # Threshold for "touching" (normalized units)
        return distance < 0.05

