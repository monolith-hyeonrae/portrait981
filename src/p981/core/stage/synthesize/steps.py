"""Synthesize 스테이지 전용 스텝 구성 요소."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ...domain import AssetService, SynthesisService
from ...executor.step_runner import StageRunner, StageStep
from ...types import AssetRef, Style


class SynthesizeDeps(Protocol):
    """Synthesize 스텝 실행에 필요한 의존성."""

    asset: AssetService
    synthesis: SynthesisService


class SynthesizeRequest(Protocol):
    """Synthesize 입력에 필요한 필드 모음."""

    style: Style
    moment_ref: AssetRef | None
    base_portrait_ref: AssetRef | None
    closeup_image_ref: AssetRef | None
    fullbody_image_ref: AssetRef | None


@dataclass
class SynthesizeStepBuilder:
    """Synthesize 스텝 구성과 실행을 돕는다."""

    deps: SynthesizeDeps

    def run(self, request: SynthesizeRequest, runner: StageRunner) -> AssetRef:
        handlers = {
            "base": self._run_base,
            "closeup": self._run_closeup,
            "fullbody": self._run_fullbody,
            "cinematic": self._run_cinematic,
        }
        handler = handlers.get(request.style)
        if handler is None:
            raise ValueError(f"Unsupported style: {request.style}")
        return handler(request, runner)

    def _run_base(self, request: SynthesizeRequest, runner: StageRunner) -> AssetRef:
        if not request.moment_ref:
            raise ValueError("moment_ref is required for base style")

        # 입력 메타를 불러온다.
        moment_meta = self._load_meta(
            runner,
            step_name="asset.load_moment",
            message="load moment metadata",
            progress=0.1,
            asset_ref=request.moment_ref,
        )

        # 합성 실행.
        keyframe_pack_ref = moment_meta.get("keyframe_pack_ref", request.moment_ref)
        blob_ref = runner.run_step(
            StageStep(
                name="synthesis.base",
                message="generate base portrait",
                progress=0.45,
                action=lambda: self.deps.synthesis.synthesize_base(keyframe_pack_ref),
            )
        )

        # 결과 자산 저장.
        member_id = moment_meta.get("member_id", "unknown")
        return self._save_asset(
            runner,
            step_name="asset.save_base",
            message="save base portrait",
            progress=0.75,
            asset_type="base_portrait",
            member_id=member_id,
            source_ref=request.moment_ref,
            blob_ref=blob_ref,
            meta={"style": "base", "moment_ref": request.moment_ref},
        )

    def _run_closeup(self, request: SynthesizeRequest, runner: StageRunner) -> AssetRef:
        if not request.base_portrait_ref:
            raise ValueError("base_portrait_ref is required for closeup style")

        # 입력 메타를 불러온다.
        base_meta = self._load_meta(
            runner,
            step_name="asset.load_base",
            message="load base portrait metadata",
            progress=0.1,
            asset_ref=request.base_portrait_ref,
        )

        # 합성 실행.
        blob_ref = runner.run_step(
            StageStep(
                name="synthesis.closeup",
                message="generate closeup image",
                progress=0.45,
                action=lambda: self.deps.synthesis.synthesize_closeup(request.base_portrait_ref),
            )
        )

        # 결과 자산 저장.
        member_id = base_meta.get("member_id", "unknown")
        return self._save_asset(
            runner,
            step_name="asset.save_closeup",
            message="save closeup image",
            progress=0.75,
            asset_type="closeup_image",
            member_id=member_id,
            source_ref=request.base_portrait_ref,
            blob_ref=blob_ref,
            meta={"style": "closeup", "base_portrait_ref": request.base_portrait_ref},
        )

    def _run_fullbody(self, request: SynthesizeRequest, runner: StageRunner) -> AssetRef:
        if not request.base_portrait_ref:
            raise ValueError("base_portrait_ref is required for fullbody style")

        # 입력 메타를 불러온다.
        base_meta = self._load_meta(
            runner,
            step_name="asset.load_base",
            message="load base portrait metadata",
            progress=0.1,
            asset_ref=request.base_portrait_ref,
        )

        # 합성 실행.
        blob_ref = runner.run_step(
            StageStep(
                name="synthesis.fullbody",
                message="generate fullbody image",
                progress=0.45,
                action=lambda: self.deps.synthesis.synthesize_fullbody(request.base_portrait_ref),
            )
        )

        # 결과 자산 저장.
        member_id = base_meta.get("member_id", "unknown")
        return self._save_asset(
            runner,
            step_name="asset.save_fullbody",
            message="save fullbody image",
            progress=0.75,
            asset_type="fullbody_image",
            member_id=member_id,
            source_ref=request.base_portrait_ref,
            blob_ref=blob_ref,
            meta={"style": "fullbody", "base_portrait_ref": request.base_portrait_ref},
        )

    def _run_cinematic(self, request: SynthesizeRequest, runner: StageRunner) -> AssetRef:
        if not request.closeup_image_ref or not request.fullbody_image_ref:
            raise ValueError("closeup_image_ref and fullbody_image_ref are required for cinematic")

        # 입력 메타를 불러온다.
        closeup_meta = self._load_meta(
            runner,
            step_name="asset.load_closeup",
            message="load closeup metadata",
            progress=0.1,
            asset_ref=request.closeup_image_ref,
        )

        # 합성 실행.
        blob_ref = runner.run_step(
            StageStep(
                name="synthesis.cinematic",
                message="generate cinematic video",
                progress=0.55,
                action=lambda: self.deps.synthesis.synthesize_cinematic(
                    request.closeup_image_ref, request.fullbody_image_ref
                ),
            )
        )

        # 결과 자산 저장.
        member_id = closeup_meta.get("member_id", "unknown")
        return self._save_asset(
            runner,
            step_name="asset.save_cinematic",
            message="save cinematic video",
            progress=0.8,
            asset_type="cinematic_video",
            member_id=member_id,
            source_ref=f"{request.closeup_image_ref}+{request.fullbody_image_ref}",
            blob_ref=blob_ref,
            meta={
                "style": "cinematic",
                "closeup_image_ref": request.closeup_image_ref,
                "fullbody_image_ref": request.fullbody_image_ref,
            },
        )

    def _load_meta(
        self,
        runner: StageRunner,
        *,
        step_name: str,
        message: str,
        progress: float,
        asset_ref: AssetRef,
    ) -> dict[str, object]:
        return runner.run_step(
            StageStep(
                name=step_name,
                message=message,
                progress=progress,
                action=lambda: self._safe_meta(asset_ref),
            )
        )

    def _save_asset(
        self,
        runner: StageRunner,
        *,
        step_name: str,
        message: str,
        progress: float,
        asset_type: str,
        member_id: object,
        source_ref: str,
        blob_ref: AssetRef | None,
        meta: dict[str, object],
    ) -> AssetRef:
        return runner.run_step(
            StageStep(
                name=step_name,
                message=message,
                progress=progress,
                action=lambda: self.deps.asset.save_asset(
                    asset_type=asset_type,
                    member_id=member_id,
                    source_ref=source_ref,
                    blob_ref=blob_ref,
                    meta=meta,
                ),
            )
        )

    def _safe_meta(self, asset_ref: str) -> dict[str, object]:
        try:
            return self.deps.asset.get_asset_meta(asset_ref)
        except KeyError:
            return {}
