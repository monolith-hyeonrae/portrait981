from __future__ import annotations

from typing import Protocol

from ..types import AssetRef


class SynthesisService(Protocol):
    def synthesize_base(self, keyframe_pack_ref: AssetRef) -> AssetRef:
        """Generate base_portrait_ref from keyframe_pack_ref."""

    def synthesize_closeup(self, base_portrait_ref: AssetRef) -> AssetRef:
        """Generate closeup_image_ref from base_portrait_ref."""

    def synthesize_fullbody(self, base_portrait_ref: AssetRef) -> AssetRef:
        """Generate fullbody_image_ref from base_portrait_ref."""

    def synthesize_cinematic(self, closeup_image_ref: AssetRef, fullbody_image_ref: AssetRef) -> AssetRef:
        """Generate cinematic_video_ref from closeup + fullbody."""


class StubSynthesisService:
    def synthesize_base(self, keyframe_pack_ref: AssetRef) -> AssetRef:
        raise NotImplementedError("SynthesisService.synthesize_base is not implemented")

    def synthesize_closeup(self, base_portrait_ref: AssetRef) -> AssetRef:
        raise NotImplementedError("SynthesisService.synthesize_closeup is not implemented")

    def synthesize_fullbody(self, base_portrait_ref: AssetRef) -> AssetRef:
        raise NotImplementedError("SynthesisService.synthesize_fullbody is not implemented")

    def synthesize_cinematic(self, closeup_image_ref: AssetRef, fullbody_image_ref: AssetRef) -> AssetRef:
        raise NotImplementedError("SynthesisService.synthesize_cinematic is not implemented")
