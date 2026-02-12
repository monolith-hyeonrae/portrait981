"""InsightFace SCRFD backend for face detection."""

import contextlib
import io
from pathlib import Path
from typing import List, Optional
import logging

import numpy as np

from vpx.face_detect.backends.base import DetectedFace

logger = logging.getLogger(__name__)


class InsightFaceSCRFD:
    """Face detection backend using InsightFace SCRFD.

    SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
    provides fast and accurate face detection with optional landmark detection.

    Args:
        model_name: Model variant (default: "buffalo_l" for best accuracy).
        det_size: Detection input size (width, height).
        det_thresh: Detection confidence threshold.

    Example:
        >>> backend = InsightFaceSCRFD()
        >>> backend.initialize("cuda:0")
        >>> faces = backend.detect(image)
        >>> backend.cleanup()
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        det_thresh: float = 0.5,
        models_dir: Optional[Path] = None,
    ):
        self._model_name = model_name
        self._det_size = det_size
        self._det_thresh = det_thresh
        self._models_dir = models_dir
        self._app: Optional[object] = None
        self._initialized = False
        self._device = "cpu"
        self._actual_provider = "unknown"
        # Batch inference session (created in _setup_batch_session)
        self._batch_session = None
        self._batch_input_name: Optional[str] = None
        self._batch_output_names: Optional[List[str]] = None
        self._batch_calibrated = False  # first-batch sanity check done

    def initialize(self, device: str = "cuda:0") -> None:
        """Initialize InsightFace app with SCRFD detector."""
        if self._initialized:
            return  # Already initialized

        try:
            from insightface.app import FaceAnalysis
            import onnxruntime as ort

            # [5] stdout 억제 — see facemoment/cli/utils.py module docstring
            ort.set_default_logger_severity(3)  # ONNX Python 로거 ERROR only

            # Check available ONNX providers
            available_providers = ort.get_available_providers()
            logger.debug(f"Available ONNX providers: {available_providers}")

            # Parse device (insightface uses ctx_id: 0 for GPU, -1 for CPU)
            if device.startswith("cuda"):
                ctx_id = int(device.split(":")[-1]) if ":" in device else 0
                if "CUDAExecutionProvider" in available_providers:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    self._actual_provider = "CUDA"
                else:
                    providers = ["CPUExecutionProvider"]
                    self._actual_provider = "CPU (CUDA unavailable)"
                    logger.warning("CUDAExecutionProvider not available, falling back to CPU")
            else:
                ctx_id = -1
                providers = ["CPUExecutionProvider"]
                self._actual_provider = "CPU"

            self._device = device

            # [5] redirect_stdout — find model, set det-size, Applied providers
            fa_kwargs = dict(name=self._model_name, providers=providers)
            if self._models_dir is not None:
                fa_kwargs["root"] = str(self._models_dir / "insightface")
            with contextlib.redirect_stdout(io.StringIO()):
                self._app = FaceAnalysis(**fa_kwargs)
                self._app.prepare(ctx_id=ctx_id, det_size=self._det_size)
            self._initialized = True
            logger.info(f"InsightFace SCRFD initialized (provider={self._actual_provider})")

            # Try to create batch-capable session for detect_batch()
            self._setup_batch_session()

        except ImportError:
            raise ImportError(
                "insightface is required for InsightFaceSCRFD backend. "
                "Install with: pip install insightface"
            )
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            raise

    def _setup_batch_session(self) -> None:
        """Create a batch-capable ONNX session from the SCRFD model.

        The default buffalo_l SCRFD model has input shape ``[1, 3, ?, ?]``
        (batch=1 fixed). This method loads the model, changes the batch
        dimension to dynamic, and creates a new InferenceSession that
        accepts ``[N, 3, H, W]`` inputs for true GPU batch inference.

        Key fix: SCRFD's detection heads use
        ``Conv [N,C,H,W] → Transpose perm=[2,3,0,1] → [H,W,N,C] → Reshape [-1,feat]``
        which interleaves batch data by spatial position when N > 1.
        We change the Transpose perm to ``[0,2,3,1]`` so the output
        is ``[N,H,W,C]``, keeping the batch dimension first and
        producing per-image contiguous blocks after Reshape.
        """
        try:
            import onnx
            from onnx import numpy_helper
            import onnxruntime as ort

            scrfd = self._app.det_model
            model_file = getattr(scrfd, 'model_file', None)
            if model_file is None:
                logger.debug("SCRFD model_file not available, batch session skipped")
                return

            # Already batch-capable
            if getattr(scrfd, 'batched', False):
                self._batch_session = scrfd.session
                self._batch_input_name = scrfd.input_name
                self._batch_output_names = scrfd.output_names
                logger.info("SCRFD already supports batch, reusing session")
                return

            model = onnx.load(model_file)

            # Make input batch dimension dynamic
            for inp in model.graph.input:
                shape = inp.type.tensor_type.shape
                if shape and len(shape.dim) >= 1:
                    shape.dim[0].dim_param = "batch"

            # Make output batch dimensions dynamic
            for out in model.graph.output:
                shape = out.type.tensor_type.shape
                if shape and len(shape.dim) >= 1:
                    shape.dim[0].dim_param = "batch"

            # Collect Reshape data-input names to identify upstream Transpose nodes
            reshape_data_inputs = set()
            for node in model.graph.node:
                if node.op_type == "Reshape" and len(node.input) >= 1:
                    reshape_data_inputs.add(node.input[0])

            # Fix Transpose nodes feeding into Reshape.
            # SCRFD detection heads: Conv→Transpose(perm=[2,3,0,1])→Reshape
            # perm=[2,3,0,1] maps [N,C,H,W]→[H,W,N,C], interleaving
            # batch data by spatial position. Change to [0,2,3,1] so
            # output is [N,H,W,C] — batch-first, per-image contiguous.
            # For batch=1 these produce identical results.
            transpose_fixed = 0
            for node in model.graph.node:
                if node.op_type != "Transpose":
                    continue
                if not any(o in reshape_data_inputs for o in node.output):
                    continue
                for attr in node.attribute:
                    if attr.name == "perm" and list(attr.ints) == [2, 3, 0, 1]:
                        attr.ints[:] = [0, 2, 3, 1]
                        transpose_fixed += 1

            # Fix Reshape nodes with hardcoded batch=1 (for models using
            # [1, -1, feat] pattern instead of [-1, feat]).
            init_map = {init.name: init for init in model.graph.initializer}
            reshape_fixed = 0
            for node in model.graph.node:
                if node.op_type == "Reshape" and len(node.input) >= 2:
                    shape_name = node.input[1]
                    if shape_name not in init_map:
                        continue
                    shape_tensor = init_map[shape_name]
                    shape_data = numpy_helper.to_array(shape_tensor).copy()
                    if (len(shape_data) >= 2
                            and shape_data[0] == 1
                            and shape_data[1] == -1):
                        shape_data[0] = 0
                        new_tensor = numpy_helper.from_array(
                            shape_data, name=shape_tensor.name,
                        )
                        shape_tensor.CopyFrom(new_tensor)
                        reshape_fixed += 1

            # Clear stale intermediate shape info (computed for batch=1)
            del model.graph.value_info[:]

            if transpose_fixed > 0 or reshape_fixed > 0:
                logger.debug(
                    "Batch model fix: %d Transpose, %d Reshape nodes patched",
                    transpose_fixed, reshape_fixed,
                )

            model_bytes = model.SerializeToString()

            providers = scrfd.session.get_providers()
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3

            with contextlib.redirect_stdout(io.StringIO()):
                self._batch_session = ort.InferenceSession(
                    model_bytes, sess_options, providers=providers,
                )

            self._batch_input_name = self._batch_session.get_inputs()[0].name
            batch_out_names = {o.name for o in self._batch_session.get_outputs()}
            orig_out_names = set(scrfd.output_names)

            if batch_out_names != orig_out_names:
                logger.warning(
                    "Batch session output names differ from original: "
                    "batch=%s, original=%s — skipping batch session",
                    sorted(batch_out_names), sorted(orig_out_names),
                )
                self._batch_session = None
                return

            # Use original SCRFD output name order so _decode_single() indexing
            # matches the expected layout (scores[0..fmc-1], bbox[fmc..2fmc-1],
            # kps[2fmc..3fmc-1]).  The batch session may return outputs in a
            # different order after onnx.load() + SerializeToString(), but
            # session.run(output_names, ...) returns results in the requested
            # order, so passing scrfd.output_names guarantees correctness.
            self._batch_output_names = list(scrfd.output_names)
            logger.info("Batch ONNX session created (dynamic batch dim)")

        except Exception as e:
            logger.debug("Batch session setup failed (will use sequential): %s", e)
            self._batch_session = None

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """Detect faces using SCRFD.

        Args:
            image: BGR image as numpy array (H, W, 3).

        Returns:
            List of detected faces with bounding boxes, landmarks, and pose.
        """
        if not self._initialized or self._app is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        faces = self._app.get(image)
        results = []

        for face in faces:
            if face.det_score < self._det_thresh:
                continue

            # Extract bounding box (x1, y1, x2, y2 -> x, y, w, h)
            bbox = face.bbox.astype(int)
            x, y = int(bbox[0]), int(bbox[1])
            w, h = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])

            # Extract pose angles if available
            yaw, pitch, roll = 0.0, 0.0, 0.0
            if hasattr(face, "pose") and face.pose is not None:
                pose = face.pose
                yaw = float(pose[1]) if len(pose) > 1 else 0.0
                pitch = float(pose[0]) if len(pose) > 0 else 0.0
                roll = float(pose[2]) if len(pose) > 2 else 0.0

            # Extract landmarks (5-point kps)
            landmarks = None
            if hasattr(face, "kps") and face.kps is not None:
                landmarks = face.kps.astype(np.float32)

            results.append(
                DetectedFace(
                    bbox=(x, y, w, h),
                    confidence=float(face.det_score),
                    landmarks=landmarks,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                )
            )

        return results

    def detect_batch(self, images: List[np.ndarray]) -> List[List[DetectedFace]]:
        """Detect faces in a batch of images using ONNX session batch inference.

        Uses a dedicated batch-capable ONNX session (created at init time
        by modifying the model's batch dimension to dynamic). Falls back
        to sequential ``detect()`` when no batch session is available.

        Args:
            images: List of BGR images as numpy arrays (H, W, 3).

        Returns:
            List of detection results, one per image.
        """
        if not self._initialized or self._app is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        if len(images) == 0:
            return []
        if len(images) == 1:
            return [self.detect(images[0])]

        if self._batch_session is None:
            return [self.detect(img) for img in images]

        try:
            results = self._detect_batch_onnx(images, self._app.det_model)

            # First-batch calibration: compare batch vs sequential on first image
            if not self._batch_calibrated:
                self._batch_calibrated = True
                seq_check = self.detect(images[0])
                batch_count = len(results[0])
                seq_count = len(seq_check)
                if seq_count > 0 and batch_count == 0:
                    logger.warning(
                        "Batch calibration failed: batch found 0 faces but "
                        "sequential found %d — disabling batch",
                        seq_count,
                    )
                    self._batch_session = None
                    return [self.detect(img) for img in images]
                if seq_count > 0 and batch_count > 0:
                    # Compare first face bbox overlap (IoU)
                    sb = seq_check[0].bbox  # (x, y, w, h)
                    bb = results[0][0].bbox
                    iou = self._bbox_iou(sb, bb)
                    if iou < 0.5:
                        logger.warning(
                            "Batch calibration failed: bbox IoU=%.2f < 0.5 "
                            "(seq=%s, batch=%s) — disabling batch",
                            iou, sb, bb,
                        )
                        self._batch_session = None
                        return [self.detect(img) for img in images]
                    logger.debug(
                        "Batch calibration OK: %d faces, IoU=%.2f",
                        batch_count, iou,
                    )

            return results
        except Exception as e:
            logger.warning("Batch inference failed, falling back to sequential: %s", e)
            return [self.detect(img) for img in images]

    def _detect_batch_onnx(
        self,
        images: List[np.ndarray],
        scrfd,
    ) -> List[List[DetectedFace]]:
        """Run true ONNX session batch inference via SCRFD internals.

        Replicates SCRFD.detect() preprocessing and forward() postprocessing
        for multiple images in a single session.run() call.

        Args:
            images: List of BGR images.
            scrfd: SCRFD model instance (``self._app.det_model``).

        Returns:
            Per-image list of DetectedFace.
        """
        import cv2

        input_size = scrfd.input_size  # (width, height)
        n = len(images)

        # --- Preprocess: resize + letterbox per image ---
        blobs = []
        det_scales: List[float] = []

        for img in images:
            im_ratio = float(img.shape[0]) / img.shape[1]
            model_ratio = float(input_size[1]) / input_size[0]
            if im_ratio > model_ratio:
                new_height = input_size[1]
                new_width = int(new_height / im_ratio)
            else:
                new_width = input_size[0]
                new_height = int(new_width * im_ratio)

            det_scales.append(float(new_height) / img.shape[0])

            resized = cv2.resize(img, (new_width, new_height))
            det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
            det_img[:new_height, :new_width, :] = resized

            blob = cv2.dnn.blobFromImage(
                det_img,
                1.0 / scrfd.input_std,
                (input_size[0], input_size[1]),
                (scrfd.input_mean, scrfd.input_mean, scrfd.input_mean),
                swapRB=True,
            )
            blobs.append(blob)

        # --- Single ONNX session.run with batch (dynamic batch session) ---
        batch_blob = np.concatenate(blobs, axis=0)  # [N, 3, H, W]
        net_outs = self._batch_session.run(
            self._batch_output_names,
            {self._batch_input_name: batch_blob},
        )

        # --- Per-image postprocessing ---
        all_results: List[List[DetectedFace]] = []
        for img_idx in range(n):
            scores_list, bboxes_list, kpss_list = self._decode_single(
                scrfd, net_outs, img_idx,
            )

            if not scores_list or all(len(s) == 0 for s in scores_list):
                all_results.append([])
                continue

            scores = np.vstack(scores_list)
            bboxes = np.vstack(bboxes_list) / det_scales[img_idx]

            kpss = None
            if scrfd.use_kps and kpss_list:
                kpss = np.vstack(kpss_list) / det_scales[img_idx]

            pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
            order = scores.ravel().argsort()[::-1]
            pre_det = pre_det[order, :]
            if kpss is not None:
                kpss = kpss[order, :]

            keep = scrfd.nms(pre_det)
            det = pre_det[keep, :]
            if kpss is not None:
                kpss = kpss[keep, :]

            all_results.append(self._det_to_faces(det, kpss))

        return all_results

    @staticmethod
    def _decode_single(scrfd, net_outs, img_idx):
        """Decode network outputs for a single image in a batch.

        Handles two possible output layouts:
        - 2D flattened: ``[N * anchors, features]`` — slice by anchor offset
        - 3D batched: ``[N, anchors, features]`` — index by batch dim

        Which layout is produced depends on the ONNX model's internal
        reshape operations. We detect the layout from ``ndim`` and use
        the appropriate indexing.

        Args:
            scrfd: SCRFD/RetinaFace model instance.
            net_outs: Raw outputs from ``session.run()``.
            img_idx: Index of the image in the batch.

        Returns:
            Tuple of (scores_list, bboxes_list, kpss_list) per FPN level.
        """
        from insightface.model_zoo.retinaface import distance2bbox, distance2kps

        input_height, input_width = scrfd.input_size[1], scrfd.input_size[0]
        fmc = scrfd.fmc
        threshold = scrfd.det_thresh
        feat_stride_fpn = scrfd._feat_stride_fpn
        num_anchors = scrfd._num_anchors

        scores_list, bboxes_list, kpss_list = [], [], []

        for idx, stride in enumerate(feat_stride_fpn):
            height = input_height // stride
            width = input_width // stride
            anchors_per_image = height * width * num_anchors

            # Extract per-image data based on output layout
            score_out = net_outs[idx]
            bbox_out = net_outs[idx + fmc]

            if score_out.ndim == 3:
                # 3D batched: [N, anchors, features]
                scores = score_out[img_idx]
                bbox_preds = bbox_out[img_idx] * stride
                if scrfd.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][img_idx] * stride
            else:
                # 2D flattened: [N*anchors, features]
                start = img_idx * anchors_per_image
                end = start + anchors_per_image
                scores = score_out[start:end]
                bbox_preds = bbox_out[start:end] * stride
                if scrfd.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][start:end] * stride

            # Anchor centers (cached by SCRFD)
            key = (height, width, stride)
            if key in scrfd.center_cache:
                anchor_centers = scrfd.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1,
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * num_anchors, axis=1,
                    ).reshape((-1, 2))
                if len(scrfd.center_cache) < 100:
                    scrfd.center_cache[key] = anchor_centers

            # Filter by threshold
            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])

            if scrfd.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                kpss_list.append(kpss[pos_inds])

        return scores_list, bboxes_list, kpss_list

    def _det_to_faces(self, det: np.ndarray, kpss) -> List[DetectedFace]:
        """Convert raw detection arrays to DetectedFace list.

        Args:
            det: Shape (N, 5) with columns [x1, y1, x2, y2, score].
            kpss: Shape (N, K, 2) keypoints or None.

        Returns:
            List of DetectedFace (filtered by det_thresh).
        """
        results = []
        for i in range(det.shape[0]):
            score = float(det[i, 4])
            if score < self._det_thresh:
                continue

            x1, y1, x2, y2 = det[i, :4].astype(int)
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)

            landmarks = None
            yaw, pitch, roll = 0.0, 0.0, 0.0
            if kpss is not None:
                landmarks = kpss[i].astype(np.float32)
                yaw, pitch, roll = self._estimate_pose_from_kps(landmarks)

            results.append(
                DetectedFace(
                    bbox=(x, y, w, h),
                    confidence=score,
                    landmarks=landmarks,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                )
            )
        return results

    @staticmethod
    def _estimate_pose_from_kps(kps: np.ndarray) -> tuple:
        """Estimate head pose (yaw, pitch, roll) from 5-point face landmarks.

        Uses geometric relationships between the 5 canonical landmarks
        (right_eye, left_eye, nose, right_mouth, left_mouth) to approximate
        head pose angles. This provides reasonable accuracy for thresholds
        used in trigger detection (±10-25°).

        InsightFace landmark order:
            0: right eye, 1: left eye, 2: nose tip,
            3: right mouth corner, 4: left mouth corner

        Args:
            kps: (5, 2) array of landmark (x, y) positions in pixel coords.

        Returns:
            (yaw, pitch, roll) in degrees.
        """
        if kps.shape[0] < 5:
            return 0.0, 0.0, 0.0

        right_eye = kps[0]
        left_eye = kps[1]
        nose = kps[2]
        right_mouth = kps[3]
        left_mouth = kps[4]

        # Eye midpoint and inter-eye distance (reference scale)
        eye_mid = (right_eye + left_eye) * 0.5
        eye_dist = float(np.linalg.norm(left_eye - right_eye))
        if eye_dist < 1e-6:
            return 0.0, 0.0, 0.0

        # Mouth midpoint
        mouth_mid = (right_mouth + left_mouth) * 0.5

        # --- Yaw ---
        # Nose horizontal offset from eye midpoint, normalized by eye distance.
        # Positive = looking right (from camera's view).
        nose_offset_x = (nose[0] - eye_mid[0]) / eye_dist
        yaw = float(np.degrees(np.arcsin(np.clip(nose_offset_x, -1.0, 1.0)))) * 2.0

        # --- Pitch ---
        # Vertical position of nose relative to eye-to-mouth span.
        # A frontal face has nose at ~55% from eyes to mouth (5-point landmarks).
        face_height = float(mouth_mid[1] - eye_mid[1])
        if face_height > 1e-6:
            expected_ratio = 0.55
            actual_ratio = (nose[1] - eye_mid[1]) / face_height
            pitch_offset = actual_ratio - expected_ratio
            pitch = float(np.degrees(np.arcsin(np.clip(pitch_offset * 2.5, -1.0, 1.0))))
        else:
            pitch = 0.0

        # --- Roll ---
        # Angle of the line connecting the two eyes.
        dy = float(left_eye[1] - right_eye[1])
        dx = float(left_eye[0] - right_eye[0])
        roll = float(np.degrees(np.arctan2(dy, dx)))

        return yaw, pitch, roll

    @staticmethod
    def _bbox_iou(a: tuple, b: tuple) -> float:
        """Compute IoU between two (x, y, w, h) bounding boxes."""
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def cleanup(self) -> None:
        """Release InsightFace resources."""
        self._app = None
        self._batch_session = None
        self._initialized = False
        logger.info("InsightFace SCRFD cleaned up")

    def get_provider_info(self) -> str:
        """Get actual provider being used."""
        return self._actual_provider
