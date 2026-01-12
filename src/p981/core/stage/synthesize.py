from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..domain import AssetService, SynthesisService
from ..common import ProgressHandler, ProgressReporter
from ..types import SynthesizeInput, SynthesizeOutput
from .runner import StageRunner, StageStep


@dataclass(frozen=True)
class SynthesizeStageDeps:
    asset: AssetService
    synthesis: SynthesisService


class SynthesizeStage(Protocol):
    def run(self, request: SynthesizeInput, progress: ProgressHandler | None = None) -> SynthesizeOutput:
        """Execute synthesize stage and return its outputs."""


class StubSynthesizeStage:
    def __init__(self, deps: SynthesizeStageDeps) -> None:
        self._deps = deps

    def run(self, request: SynthesizeInput, progress: ProgressHandler | None = None) -> SynthesizeOutput:
        raise NotImplementedError("SynthesizeStage.run is not implemented")


class SimpleSynthesizeStage:
    def __init__(self, deps: SynthesizeStageDeps) -> None:
        self._deps = deps

    def run(self, request: SynthesizeInput, progress: ProgressHandler | None = None) -> SynthesizeOutput:
        reporter = ProgressReporter("synthesize", progress)
        runner = StageRunner(reporter)
        style = request.style
        if style == "base":
            if not request.moment_ref:
                raise ValueError("moment_ref is required for base style")
            moment_meta: dict[str, object] = {}
            blob_ref: str | None = None

            def load_moment() -> None:
                nonlocal moment_meta
                moment_meta = self._safe_meta(request.moment_ref)

            def synthesize_base() -> None:
                nonlocal blob_ref
                keyframe_pack_ref = moment_meta.get("keyframe_pack_ref", request.moment_ref)
                blob_ref = self._deps.synthesis.synthesize_base(keyframe_pack_ref)

            runner.run(
                [
                    StageStep(
                        name="asset.load_moment",
                        message="load moment metadata",
                        progress=0.1,
                        action=load_moment,
                    ),
                    StageStep(
                        name="synthesis.base",
                        message="generate base portrait",
                        progress=0.45,
                        action=synthesize_base,
                    ),
                ]
            )
            customer_id = moment_meta.get("customer_id", "unknown")
            generated_ref = runner.run_step(
                StageStep(
                    name="asset.save_base",
                    message="save base portrait",
                    progress=0.75,
                    action=lambda: self._deps.asset.save_asset(
                        asset_type="base_portrait",
                        customer_id=customer_id,
                        source_ref=request.moment_ref,
                        blob_ref=blob_ref,
                        meta={"style": "base", "moment_ref": request.moment_ref},
                    ),
                )
            )
        elif style == "closeup":
            if not request.base_portrait_ref:
                raise ValueError("base_portrait_ref is required for closeup style")
            base_meta: dict[str, object] = {}
            blob_ref: str | None = None

            def load_base() -> None:
                nonlocal base_meta
                base_meta = self._safe_meta(request.base_portrait_ref)

            def synthesize_closeup() -> None:
                nonlocal blob_ref
                blob_ref = self._deps.synthesis.synthesize_closeup(request.base_portrait_ref)

            runner.run(
                [
                    StageStep(
                        name="asset.load_base",
                        message="load base portrait metadata",
                        progress=0.1,
                        action=load_base,
                    ),
                    StageStep(
                        name="synthesis.closeup",
                        message="generate closeup image",
                        progress=0.45,
                        action=synthesize_closeup,
                    ),
                ]
            )
            customer_id = base_meta.get("customer_id", "unknown")
            generated_ref = runner.run_step(
                StageStep(
                    name="asset.save_closeup",
                    message="save closeup image",
                    progress=0.75,
                    action=lambda: self._deps.asset.save_asset(
                        asset_type="closeup_image",
                        customer_id=customer_id,
                        source_ref=request.base_portrait_ref,
                        blob_ref=blob_ref,
                        meta={"style": "closeup", "base_portrait_ref": request.base_portrait_ref},
                    ),
                )
            )
        elif style == "fullbody":
            if not request.base_portrait_ref:
                raise ValueError("base_portrait_ref is required for fullbody style")
            base_meta: dict[str, object] = {}
            blob_ref: str | None = None

            def load_base() -> None:
                nonlocal base_meta
                base_meta = self._safe_meta(request.base_portrait_ref)

            def synthesize_fullbody() -> None:
                nonlocal blob_ref
                blob_ref = self._deps.synthesis.synthesize_fullbody(request.base_portrait_ref)

            runner.run(
                [
                    StageStep(
                        name="asset.load_base",
                        message="load base portrait metadata",
                        progress=0.1,
                        action=load_base,
                    ),
                    StageStep(
                        name="synthesis.fullbody",
                        message="generate fullbody image",
                        progress=0.45,
                        action=synthesize_fullbody,
                    ),
                ]
            )
            customer_id = base_meta.get("customer_id", "unknown")
            generated_ref = runner.run_step(
                StageStep(
                    name="asset.save_fullbody",
                    message="save fullbody image",
                    progress=0.75,
                    action=lambda: self._deps.asset.save_asset(
                        asset_type="fullbody_image",
                        customer_id=customer_id,
                        source_ref=request.base_portrait_ref,
                        blob_ref=blob_ref,
                        meta={"style": "fullbody", "base_portrait_ref": request.base_portrait_ref},
                    ),
                )
            )
        elif style == "cinematic":
            if not request.closeup_image_ref or not request.fullbody_image_ref:
                raise ValueError("closeup_image_ref and fullbody_image_ref are required for cinematic")
            closeup_meta: dict[str, object] = {}
            blob_ref: str | None = None

            def load_closeup() -> None:
                nonlocal closeup_meta
                closeup_meta = self._safe_meta(request.closeup_image_ref)

            def synthesize_cinematic() -> None:
                nonlocal blob_ref
                blob_ref = self._deps.synthesis.synthesize_cinematic(
                    request.closeup_image_ref, request.fullbody_image_ref
                )

            runner.run(
                [
                    StageStep(
                        name="asset.load_closeup",
                        message="load closeup metadata",
                        progress=0.1,
                        action=load_closeup,
                    ),
                    StageStep(
                        name="synthesis.cinematic",
                        message="generate cinematic video",
                        progress=0.55,
                        action=synthesize_cinematic,
                    ),
                ]
            )
            customer_id = closeup_meta.get("customer_id", "unknown")
            generated_ref = runner.run_step(
                StageStep(
                    name="asset.save_cinematic",
                    message="save cinematic video",
                    progress=0.8,
                    action=lambda: self._deps.asset.save_asset(
                        asset_type="cinematic_video",
                        customer_id=customer_id,
                        source_ref=f"{request.closeup_image_ref}+{request.fullbody_image_ref}",
                        blob_ref=blob_ref,
                        meta={
                            "style": "cinematic",
                            "closeup_image_ref": request.closeup_image_ref,
                            "fullbody_image_ref": request.fullbody_image_ref,
                        },
                    ),
                )
            )
        else:
            raise ValueError(f"Unsupported style: {style}")

        reporter.emit("synthesize.complete", "synthesize complete", 1.0)
        return SynthesizeOutput(generated_asset_ref=generated_ref, reused_existing=False)

    def _safe_meta(self, asset_ref: str) -> dict[str, object]:
        try:
            return self._deps.asset.get_asset_meta(asset_ref)
        except KeyError:
            return {}
