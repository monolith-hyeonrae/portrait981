"""합성 도메인 서비스 인터페이스를 정의한다."""

from __future__ import annotations

from typing import Protocol

from ..types import AssetRef


class SynthesisService(Protocol):
    def synthesize_base(self, keyframe_pack_ref: AssetRef) -> AssetRef:
        """keyframe_pack_ref로 base_portrait_ref를 생성한다."""

    def synthesize_closeup(self, base_portrait_ref: AssetRef) -> AssetRef:
        """base_portrait_ref로 closeup_image_ref를 생성한다."""

    def synthesize_fullbody(self, base_portrait_ref: AssetRef) -> AssetRef:
        """base_portrait_ref로 fullbody_image_ref를 생성한다."""

    def synthesize_cinematic(self, closeup_image_ref: AssetRef, fullbody_image_ref: AssetRef) -> AssetRef:
        """closeup + fullbody로 cinematic_video_ref를 생성한다."""


class StubSynthesisService:
    def synthesize_base(self, keyframe_pack_ref: AssetRef) -> AssetRef:
        raise NotImplementedError("SynthesisService.synthesize_base is not implemented")

    def synthesize_closeup(self, base_portrait_ref: AssetRef) -> AssetRef:
        raise NotImplementedError("SynthesisService.synthesize_closeup is not implemented")

    def synthesize_fullbody(self, base_portrait_ref: AssetRef) -> AssetRef:
        raise NotImplementedError("SynthesisService.synthesize_fullbody is not implemented")

    def synthesize_cinematic(self, closeup_image_ref: AssetRef, fullbody_image_ref: AssetRef) -> AssetRef:
        raise NotImplementedError("SynthesisService.synthesize_cinematic is not implemented")
