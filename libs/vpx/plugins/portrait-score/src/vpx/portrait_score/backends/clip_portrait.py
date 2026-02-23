"""CLIP Portrait Quality Scorer.

CLIP ViT-B/32 image embedding과 포트레이트 품질 텍스트 프롬프트 간
cosine similarity를 계산하여 [0, 1] 점수를 반환한다.

텍스트 프롬프트를 통해 "얼굴이 멋지게 나오는 순간"에 특화된 점수를 산출.
프롬프트를 조정하면 평가 기준을 코드 변경 없이 튜닝 가능.

Axis-based scoring:
    N개 프롬프트 임베딩을 평균하여 **하나의 축 벡터**로 압축.
    축 단위로 홀리스틱 개념을 판정한다.
    (논문 "Image Aesthetic Assessment with CLIP" 방식 응용)

Usage:
    scorer = CLIPPortraitScorer(device="cpu")
    scorer.initialize()
    if scorer.available:
        score = scorer.score(image_bgr)  # float in [0, 1]
        print(scorer.last_breakdown)     # per-prompt similarities
        print(scorer.last_axes)          # axis scores
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_CLIP_MODEL = "ViT-B-32"
_CLIP_PRETRAINED = "openai"

# ── Default prompts ──
# Positive: 포트레이트 사진에서 "좋아 보이는" 특징
_DEFAULT_POSITIVE = [
    "a portrait photo of a person with a natural smile",
    "an attractive well-lit face looking at camera",
    "a sharp clear portrait photo with good expression",
]

# Negative: 포트레이트 사진에서 "안 좋아 보이는" 특징
_DEFAULT_NEGATIVE = [
    "a blurry out of focus face",
    "a person with closed eyes and awkward expression",
    "a dark poorly lit unflattering photo of a face",
]


# ── Axis definitions ──
# N개 프롬프트를 평균 → 하나의 축 벡터로 압축.
# 전문 모델(HSEmotion, LibreFace AU, 6DRepNet)이 커버 못하는 홀리스틱 품질만 남김.


@dataclass(frozen=True)
class AxisDefinition:
    """축 정의: positive/negative 프롬프트 그룹 → 양극 축 벡터.

    score = sigmoid(logit_scale × (cos_sim(img, pos) - cos_sim(img, neg)))
    positive만으로는 절대 유사도에 의존해 프레임 간 차이가 작지만,
    negative를 정의하면 축 위 상대 위치를 측정하여 변별력이 높아진다.
    """

    name: str
    prompts: Tuple[str, ...]      # positive 방향
    neg_prompts: Tuple[str, ...]  # negative 반대축
    action: str  # "select" | "quirky"
    threshold: float = 0.5  # sigmoid score threshold for activation


_AXES: Tuple[AxisDefinition, ...] = (
    # ── disney_smile: 미소의 질감 ──
    # 입 다문 채 눈웃음 + 따뜻한 온기. Duchenne 미소의 동화적 질감.
    # negative: 크게 벌린 웃음(hearty_laugh), 시크함(cold_charisma), 무표정 구분.
    AxisDefinition(
        name="disney_smile",
        prompts=(
            "a person with a warm gentle closed-mouth Duchenne smile with crinkled eyes",
            "a radiant angelic smile like a Disney princess or prince",
            "a sweet tender smile with soft sparkling eyes full of warmth",
            "a person smiling gently with pure innocent joy and kindness",
        ),
        neg_prompts=(
            "a person laughing hard with mouth wide open",
            "a person with a cool smirk and no warmth",
            "a person with a blank neutral unsmiling face",
        ),
        action="select",
    ),
    # ── model_aura: 시크한 카리스마 ──
    # 웃지 않지만 멋짐이 폭발하는 순간. 시크/도도/당당/자신감.
    # negative로 웃는 얼굴을 명시하여, 미소 없는 쿨함만 잡는다.
    AxisDefinition(
        name="charisma",
        prompts=(
            "a person with a cool confident smirk not smiling",
            "a person with a chic aloof expression and sharp gaze",
            "a person posing with bold confidence and no smile",
            "a person with a fierce unbothered look",
            "a brooding mysterious portrait with intense eyes",
        ),
        neg_prompts=(
            "a person smiling warmly and friendly",
            "a person laughing with a big open smile",
            "a cheerful happy person with bright expression",
        ),
        action="select",
    ),
    # ── playful_cute: 장난기 ──
    # 혀내밀기/볼부풀리기/입삐죽 — 전문 모델 감지 불가, 검증됨.
    # negative: 입 다문 일반적 표정으로 명확히 대비.
    AxisDefinition(
        name="playful_cute",
        prompts=(
            "a person sticking tongue out playfully",
            "a person puffing cheeks like a blowfish",
            "a person making an exaggerated duck face pout",
            "a person crossing eyes and making a goofy face",
        ),
        neg_prompts=(
            "a person with mouth closed and normal expression",
            "a person with a neutral relaxed face",
            "a person frowning with a displeased scowling expression",
        ),
        action="select",
    ),
    # ── wild_roar: 포효/돌격 함성 ──
    # 입 크게 벌리고 소리 지르듯 호탕한 순간. 웃음+함성+흥분 전부 포함.
    # negative: 입 다문 잔잔한 상태로 확실히 분리.
    AxisDefinition(
        name="wild_roar",
        prompts=(
            "a person screaming and yelling with mouth wide open in excitement",
            "a person roaring with laughter and head thrown back",
            "a person shouting with intense energy and a wide open mouth",
            "a person with an explosive expression of pure thrill and adrenaline",
        ),
        neg_prompts=(
            "a person with a gentle closed-mouth smile",
            "a person sitting quietly with a calm still expression",
            "a person with lips together and a composed face",
        ),
        action="select",
    ),
)


@dataclass(frozen=True)
class CompositeDefinition:
    """복합 축: 기본 축 점수의 기하평균으로 정의."""

    name: str
    components: Tuple[str, ...]  # 조합할 기본 축 이름
    action: str  # "select" | "quirky"
    threshold: float = 0.5


_COMPOSITES: Tuple[CompositeDefinition, ...] = ()


@dataclass
class AxisScore:
    """단일 축 스코어링 결과."""

    name: str
    score: float  # 0~1 정규화된 점수
    raw_sim: float  # raw cosine similarity
    active: bool  # score > threshold
    action: str  # "select" | "quirky"
    composite: bool = False  # True면 기본 축 조합으로 산출된 복합 축


# cosine similarity 관측 범위 → 0~1 선형 변환
_SIM_FLOOR = 0.15
_SIM_CEIL = 0.35


def _normalize_sim(raw: float) -> float:
    """Raw cosine similarity → 0~1 linear scale."""
    return max(0.0, min(1.0, (raw - _SIM_FLOOR) / (_SIM_CEIL - _SIM_FLOOR)))


@dataclass
class PromptBreakdown:
    """프롬프트별 cosine similarity 분석 결과."""

    score: float = 0.0
    positive: Dict[str, float] = field(default_factory=dict)  # prompt → sim
    negative: Dict[str, float] = field(default_factory=dict)  # prompt → sim
    avg_pos: float = 0.0
    avg_neg: float = 0.0
    raw_diff: float = 0.0


class CLIPPortraitScorer:
    """CLIP text-image similarity 기반 포트레이트 품질 점수.

    score = avg(cos_sim(img, positive)) - avg(cos_sim(img, negative))
    → [0, 1] 정규화.

    Linear head 불필요 — CLIP 모델만 로드하면 동작.
    """

    def __init__(
        self,
        device: str = "auto",
        positive_prompts: Optional[List[str]] = None,
        negative_prompts: Optional[List[str]] = None,
        embed_ema_alpha: float = 0.2,
        enable_caption: bool = False,
    ):
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._device = device
        self._positive_prompts = positive_prompts or list(_DEFAULT_POSITIVE)
        self._negative_prompts = negative_prompts or list(_DEFAULT_NEGATIVE)

        self._clip_model = None
        self._preprocess = None
        self._tokenizer = None
        self._dtype = None  # torch.float32 or torch.float16
        self._logit_scale: float = 1.0  # CLIP learned temperature
        self._pos_embeds = None  # (N_pos, embed_dim) cached
        self._neg_embeds = None  # (N_neg, embed_dim) cached
        self._axis_pos_embeds = None  # (N_axes, embed_dim) positive axis
        self._axis_neg_embeds = None  # (N_axes, embed_dim) negative axis
        self._embed_ema_alpha = embed_ema_alpha  # EMA 가중치 (1.0=스무딩 없음)
        self._embed_ema = None  # (1, embed_dim) EMA 상태
        self._enable_caption = enable_caption
        self._coca_model = None
        self._coca_transform = None
        self.available: bool = False
        self.last_breakdown: Optional[PromptBreakdown] = None
        self.last_axes: Optional[List[AxisScore]] = None
        self.last_caption: Optional[str] = None

    def initialize(self) -> bool:
        """Load CLIP model and pre-encode text prompts.

        Returns True if successfully loaded, False otherwise.
        """
        try:
            import torch
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                _CLIP_MODEL, pretrained=_CLIP_PRETRAINED,
            )
            model = model.to(self._device)
            model.eval()

            # fp16 on CUDA for ~2x speedup
            import torch as _torch
            use_fp16 = "cuda" in self._device
            if use_fp16:
                model = model.half()
                self._dtype = _torch.float16
                logger.info("CLIPPortraitScorer: fp16 enabled on %s", self._device)
            else:
                self._dtype = _torch.float32

            tokenizer = open_clip.get_tokenizer(_CLIP_MODEL)

            # Pre-encode text prompts
            with torch.no_grad():
                pos_tokens = tokenizer(self._positive_prompts).to(self._device)
                neg_tokens = tokenizer(self._negative_prompts).to(self._device)
                pos_embeds = model.encode_text(pos_tokens)
                neg_embeds = model.encode_text(neg_tokens)
                # L2 normalize
                pos_embeds = pos_embeds / pos_embeds.norm(dim=-1, keepdim=True)
                neg_embeds = neg_embeds / neg_embeds.norm(dim=-1, keepdim=True)

                # Axis embeddings: pos/neg 프롬프트 한번에 인코딩 → 축별 평균 → L2 정규화
                all_prompts: List[str] = []
                pos_slices: List[Tuple[int, int]] = []
                neg_slices: List[Tuple[int, int]] = []
                for axis_def in _AXES:
                    ps = len(all_prompts)
                    all_prompts.extend(axis_def.prompts)
                    pos_slices.append((ps, len(all_prompts)))
                    ns = len(all_prompts)
                    all_prompts.extend(axis_def.neg_prompts)
                    neg_slices.append((ns, len(all_prompts)))

                all_tokens = tokenizer(all_prompts).to(self._device)
                all_embeds = model.encode_text(all_tokens)
                all_embeds = all_embeds / all_embeds.norm(dim=-1, keepdim=True)

                def _mean_norm(embeds, start, end):
                    m = embeds[start:end].mean(dim=0, keepdim=True)
                    return m / m.norm(dim=-1, keepdim=True)

                axis_pos_list = [_mean_norm(all_embeds, s, e) for s, e in pos_slices]
                axis_neg_list = [_mean_norm(all_embeds, s, e) for s, e in neg_slices]

                axis_pos_embeds = torch.cat(axis_pos_list, dim=0)  # (N_axes, D)
                axis_neg_embeds = torch.cat(axis_neg_list, dim=0)  # (N_axes, D)

            # CLIP learned logit_scale — exp(log_scale) → scalar
            self._logit_scale = float(model.logit_scale.exp().item())

            self._clip_model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            self._pos_embeds = pos_embeds
            self._neg_embeds = neg_embeds
            self._axis_pos_embeds = axis_pos_embeds
            self._axis_neg_embeds = axis_neg_embeds
            self.available = True

            n_pos_prompts = sum(len(a.prompts) for a in _AXES)
            n_neg_prompts = sum(len(a.neg_prompts) for a in _AXES)
            logger.info(
                "CLIPPortraitScorer: loaded (device=%s, logit_scale=%.1f, "
                "score_pos=%d, score_neg=%d, axes=%d, axis_prompts=%d+%d)",
                self._device, self._logit_scale,
                len(self._positive_prompts),
                len(self._negative_prompts), len(_AXES),
                n_pos_prompts, n_neg_prompts,
            )

            # Optional CoCa captioner
            if self._enable_caption:
                try:
                    # open_clip.coca_model imports BeamSearchScorer from
                    # top-level transformers, but transformers >=4.47 moved it
                    # to transformers.generation.beam_search.  Patch the
                    # re-export so open_clip sees _has_transformers = True.
                    try:
                        import transformers
                        if not hasattr(transformers, "BeamSearchScorer"):
                            from transformers.generation.beam_search import BeamSearchScorer
                            transformers.BeamSearchScorer = BeamSearchScorer
                        # Reload coca_model so it picks up the fix
                        import importlib, open_clip.coca_model as _cm
                        importlib.reload(_cm)
                    except Exception:
                        pass

                    coca_model, _, coca_transform = open_clip.create_model_and_transforms(
                        "coca_ViT-B-32",
                        pretrained="mscoco_finetuned_laion2b_s13b_b90k",
                    )
                    coca_model = coca_model.to(self._device)
                    coca_model.eval()
                    if use_fp16:
                        coca_model = coca_model.half()
                    self._coca_model = coca_model
                    self._coca_transform = coca_transform
                    logger.info("CLIPPortraitScorer: CoCa captioner enabled")
                except Exception as e:
                    logger.info("CLIPPortraitScorer: CoCa unavailable — %s", e)

            return True

        except Exception as e:
            logger.warning("CLIPPortraitScorer: initialization failed — %s", e)
            self.available = False
            return False

    def _encode_image(self, image_bgr: np.ndarray):
        """BGR image → L2-normalized CLIP embedding (with EMA smoothing)."""
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        with torch.no_grad():
            tensor = self._preprocess(pil_img).unsqueeze(0)
            tensor = tensor.to(device=self._device, dtype=self._dtype)
            embed = self._clip_model.encode_image(tensor)
            embed = embed / embed.norm(dim=-1, keepdim=True)

            # EMA smoothing on embedding
            alpha = self._embed_ema_alpha
            if self._embed_ema is None:
                self._embed_ema = embed
            else:
                self._embed_ema = alpha * embed + (1.0 - alpha) * self._embed_ema
                self._embed_ema = self._embed_ema / self._embed_ema.norm(dim=-1, keepdim=True)
            embed = self._embed_ema

        return embed

    def score(self, image_bgr: np.ndarray) -> float:
        """Return portrait quality score in [0, 1] for a BGR image.

        Per-prompt similarities are stored in self.last_breakdown.
        Axis scores are stored in self.last_axes.
        """
        if not self.available:
            self.last_breakdown = None
            self.last_axes = None
            return 0.0

        try:
            img_embed = self._encode_image(image_bgr)

            # Cosine similarity (raw 0~1) + logit_scale for sigmoid
            ls = self._logit_scale
            raw_pos = (img_embed @ self._pos_embeds.T).squeeze(0)  # (N_pos,)
            raw_neg = (img_embed @ self._neg_embeds.T).squeeze(0)  # (N_neg,)

            avg_pos = float(raw_pos.mean().item())
            avg_neg = float(raw_neg.mean().item())
            raw_diff = avg_pos - avg_neg

            # Sigmoid(logit_scale × diff) → [0, 1]
            import math
            final = float(1.0 / (1.0 + math.exp(-ls * raw_diff)))

            # Store per-prompt breakdown (normalized 0~1)
            self.last_breakdown = PromptBreakdown(
                score=final,
                positive={
                    p: _normalize_sim(float(raw_pos[i].item()))
                    for i, p in enumerate(self._positive_prompts)
                },
                negative={
                    p: _normalize_sim(float(raw_neg[i].item()))
                    for i, p in enumerate(self._negative_prompts)
                },
                avg_pos=_normalize_sim(avg_pos),
                avg_neg=_normalize_sim(avg_neg),
                raw_diff=raw_diff,
            )

            # Axis scoring
            self._compute_axes(img_embed)

            # CoCa caption (optional)
            self._generate_caption(image_bgr)

            return final

        except Exception as e:
            logger.debug("CLIPPortraitScorer.score failed: %s", e)
            self.last_breakdown = None
            self.last_axes = None
            return 0.0

    def _compute_axes(self, img_embed) -> None:
        """Compute axis scores from pos-neg axis embedding pairs.

        score = sigmoid(logit_scale × (cos_sim(img, pos) - cos_sim(img, neg)))
        """
        if self._axis_pos_embeds is None:
            self.last_axes = None
            return

        import math

        ls = self._logit_scale
        pos_sims = (img_embed @ self._axis_pos_embeds.T).squeeze(0)  # (N_axes,)
        neg_sims = (img_embed @ self._axis_neg_embeds.T).squeeze(0)  # (N_axes,)

        results = []
        score_map: Dict[str, float] = {}
        for i, axis_def in enumerate(_AXES):
            raw_pos = float(pos_sims[i].item())
            raw_neg = float(neg_sims[i].item())
            raw_diff = raw_pos - raw_neg
            score = 1.0 / (1.0 + math.exp(-ls * raw_diff))
            score_map[axis_def.name] = score
            results.append(AxisScore(
                name=axis_def.name,
                score=score,
                raw_sim=raw_diff,
                active=score > axis_def.threshold,
                action=axis_def.action,
            ))

        # Composite axes: 기본 축 점수의 기하평균
        for comp in _COMPOSITES:
            vals = [score_map[c] for c in comp.components if c in score_map]
            if vals:
                geo_mean = math.prod(vals) ** (1.0 / len(vals))
            else:
                geo_mean = 0.0
            results.append(AxisScore(
                name=comp.name,
                score=geo_mean,
                raw_sim=0.0,
                active=geo_mean > comp.threshold,
                action=comp.action,
                composite=True,
            ))

        self.last_axes = results

    def _generate_caption(self, image_bgr: np.ndarray) -> None:
        """Generate a short caption using CoCa (optional)."""
        if self._coca_model is None:
            self.last_caption = None
            return

        try:
            import torch
            from PIL import Image

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            with torch.no_grad():
                tensor = self._coca_transform(pil_img).unsqueeze(0)
                # CoCa generate는 fp32가 안정적 — fp16 텐서를 fp32로 변환
                tensor = tensor.to(device=self._device, dtype=torch.float32)
                generated = self._coca_model.float().generate(
                    tensor, seq_len=20,
                )
                # fp16 모드면 다시 half로 복원
                if self._dtype == torch.float16:
                    self._coca_model.half()

                import open_clip
                caption = open_clip.decode(generated[0]).strip()
                for token in ("<start_of_text>", "<end_of_text>"):
                    caption = caption.replace(token, "")
                self.last_caption = caption.strip()
        except Exception as e:
            if not getattr(self, "_caption_warned", False):
                logger.warning("CoCa caption failed: %s", e)
                self._caption_warned = True
            self.last_caption = None

    def cleanup(self) -> None:
        self._clip_model = None
        self._preprocess = None
        self._tokenizer = None
        self._dtype = None
        self._logit_scale = 1.0
        self._pos_embeds = None
        self._neg_embeds = None
        self._axis_pos_embeds = None
        self._axis_neg_embeds = None
        self._embed_ema = None
        self._coca_model = None
        self._coca_transform = None
        self._caption_warned = False
        self.available = False
        self.last_breakdown = None
        self.last_axes = None
        self.last_caption = None
