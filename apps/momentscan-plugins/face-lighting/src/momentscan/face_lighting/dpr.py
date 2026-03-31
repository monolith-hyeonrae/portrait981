"""DPR (Deep Portrait Relighting) wrapper for lighting SH estimation.

SH 9계수를 단일 얼굴 이미지에서 추출.
690K parameters, CPU 추론.

⚠️ SH 좌표계 (이미지 기준, DPR convention):
  SH[0] = ambient
  SH[1] = Y (depth, 안쪽 양수)
  SH[2] = Z (상하, 이미지 위가 양수)
  SH[3] = X (좌우, 이미지 우측이 양수)
  SH[4-8] = 2nd order

  X > 0 = 이미지 우측이 밝음
  X < 0 = 이미지 좌측이 밝음
  Z > 0 = 이미지 상단이 밝음

Usage:
    from visualbind.dpr import DPRLighting

    dpr = DPRLighting()
    dpr.initialize()
    desc = dpr.estimate_lighting_descriptor(face_crop_bgr)
    # desc["sh_dir_x"] > 0 = 이미지 우측 밝음
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger("visualbind.dpr")


def _default_model_path() -> Path:
    cwd_path = Path("models/dpr_v1.t7")
    if cwd_path.exists():
        return cwd_path
    try:
        from vpx.sdk.paths import get_models_dir
        sdk_path = get_models_dir() / "dpr_v1.t7"
        if sdk_path.exists():
            return sdk_path
    except ImportError:
        pass
    return cwd_path


class DPRLighting:
    """DPR pretrained model wrapper for SH lighting estimation."""

    def __init__(self, model_path: Optional[Path] = None):
        self._model_path = model_path or _default_model_path()
        self._model = None

    def initialize(self):
        if self._model is not None:
            return

        from momentscan.face_lighting.dpr_model import HourglassNet

        model = HourglassNet()
        model.load_state_dict(torch.load(str(self._model_path), map_location="cpu"))
        model.cpu()
        model.eval()

        self._model = model
        logger.info("DPR loaded: %s (%.1fMB, %dK params)",
                    self._model_path.name,
                    sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6,
                    sum(p.numel() for p in model.parameters()) // 1000)

    def estimate(self, face_bgr: np.ndarray) -> np.ndarray:
        """Extract 9 SH coefficients from face image.

        Args:
            face_bgr: BGR face image (any size, resized to 512x512)

        Returns:
            (9,) float array — SH coefficients [amb, Y, Z, X, ...]
        """
        if self._model is None:
            self.initialize()

        # DPR input: L channel of LAB, 512x512
        img_512 = cv2.resize(face_bgr, (512, 512))
        Lab = cv2.cvtColor(img_512, cv2.COLOR_BGR2LAB)
        inputL = Lab[:, :, 0].astype(np.float32) / 255.0
        inputL = torch.from_numpy(inputL[None, None, ...])

        sh_input = torch.zeros(1, 9, 1, 1).float()

        with torch.no_grad():
            _, outputSH = self._model(inputL, sh_input, 0)

        return outputSH.squeeze().numpy()

    def estimate_lighting_descriptor(self, face_bgr: np.ndarray) -> dict:
        """Extract structured lighting descriptor.

        ⚠️ 이미지 좌표계 (반전 불필요):
          sh_dir_x > 0 = 이미지 우측이 밝음
          sh_dir_x < 0 = 이미지 좌측이 밝음
          sh_dir_y > 0 = 이미지 상단이 밝음

        Returns:
            dict with sh_ambient, sh_dir_x, sh_dir_y, sh_dir_strength, sh_coefficients
        """
        sh_9 = self.estimate(face_bgr)

        ambient = float(sh_9[0])
        dir_x = float(sh_9[3])    # X: 좌우 (이미지 우측 양수)
        dir_y = float(sh_9[2])    # Z: 상하 (이미지 상단 양수)
        dir_z = float(sh_9[1])    # Y: 전후 (depth)
        dir_strength = float(np.sqrt(dir_x**2 + dir_y**2 + dir_z**2))

        return {
            "sh_ambient": ambient,
            "sh_dir_x": dir_x,
            "sh_dir_y": dir_y,
            "sh_dir_strength": dir_strength,
            "sh_coefficients": sh_9.tolist(),
        }
