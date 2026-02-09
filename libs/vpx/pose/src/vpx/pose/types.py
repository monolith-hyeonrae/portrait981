"""Pose estimation domain types."""


class KeypointIndex:
    """COCO 17 keypoint indices for pose estimation.

    Used with YOLOv8-Pose and similar backends.

    Example:
        >>> kpts = pose.keypoints
        >>> left_wrist = kpts[KeypointIndex.LEFT_WRIST]
        >>> if left_wrist[2] > 0.5:  # confidence threshold
        ...     print(f"Left wrist at ({left_wrist[0]}, {left_wrist[1]})")
    """

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# COCO keypoint names in order (for reference)
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


__all__ = ["KeypointIndex", "COCO_KEYPOINT_NAMES"]
