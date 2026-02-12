# identity_memory — Temporal Identity Memory Bank

> **momentbank** 핵심 구현 상세. 마스터 플랜: [momentbank.md](momentbank.md)
>
> 인물별 Face-ID 임베딩을 시간 경과에 따라 누적·관리하는 메모리 뱅크.
> 단일 prototype(medoid)의 한계를 극복하여 **상태별 다중 centroid**를 유지하고,
> reportrait 디퓨전 생성 시 **query 기반 reference image 선택**을 제공하는 시스템.

## 1. 목적

- identity_builder가 생성한 이미지/임베딩을 **지속적으로 축적**
- 단일 prototype 대신 **k개 memory node**로 인물의 다양한 상태를 표현
- 동일인 판정(`match`)과 reference 선택(`select_refs`) 두 가지 API 제공
- ComfyUI에 **이미지 경로** 전달 — 임베딩 직접 주입 금지

## 2. 입출력

### Input

- identity_builder 출력 (anchor/coverage/challenge 이미지 + meta)
- 프레임별 Face-ID 임베딩 (`e_id`)
- 품질 점수, 버킷 메타 (yaw/pitch/expression/lighting)

### Output

```
output/{video_id}/identity_memory/
├── person_0/
│   └── memory_bank.json
├── person_1/
│   └── memory_bank.json
└── _version.json
```

### memory_bank.json schema

```json
{
  "person_id": 0,
  "k": 6,
  "nodes": [
    {
      "node_id": 0,
      "vec_id": [0.12, -0.34, ...],
      "rep_images": [
        "../../identity_builder/person_0/anchors/anchor_001.jpg",
        "../../identity_builder/person_0/coverage/cov_yaw0_pitch0_neutral.jpg"
      ],
      "meta_hist": {
        "yaw_bins": {"[-5,5]": 12, "[5,15]": 3},
        "pitch_bins": {"neutral": 10, "slight-up": 5},
        "expression_bins": {"neutral": 8, "smile": 7},
        "quality_best": 0.95,
        "quality_mean": 0.82,
        "hit_count": 15,
        "last_updated_ms": 85200.0
      }
    }
  ],
  "_version": {
    "app": "identity_memory",
    "app_version": "0.1.0",
    "embed_model": "arcface-r100",
    "gate_version": "v1",
    "created_at": "2026-02-12T15:30:00Z"
  }
}
```

## 3. Memory Node 구조

```python
@dataclass
class MemoryNode:
    node_id: int
    vec_id: np.ndarray          # Face-ID embedding, L2 normalized
    rep_images: list[str]       # 1~3 대표 이미지 경로 (quality 순)
    meta_hist: NodeMeta         # 버킷 분포 + 품질 통계

@dataclass
class NodeMeta:
    yaw_bins: dict[str, int]        # bin → hit count
    pitch_bins: dict[str, int]
    expression_bins: dict[str, int]
    quality_best: float
    quality_mean: float
    hit_count: int
    last_updated_ms: float
```

Memory bank `M = {m_0, m_1, ..., m_{k-1}}` where `k <= k_max`.
각 노드는 인물의 **특정 상태 영역**(예: "정면 나", "어두운 조명 나", "옆얼굴 나")을 대표.

## 4. 핵심 API (3개)

### 4.1 `update(e_id, quality, meta, image_path)`

메모리 뱅크에 새 관측을 반영.

```python
def update(self, e_id: np.ndarray, quality: float, meta: dict, image_path: str):
    # (A) Quality gate
    if quality < self.q_update_min:
        return  # 매칭만, 뱅크 수정 금지

    # (B) Nearest node 탐색
    sims = [cos(e_id, m.vec_id) for m in self.nodes]
    i_star = argmax(sims)
    best_sim = sims[i_star]

    # (C) EMA merge (close match)
    if best_sim >= self.tau_merge:
        m = self.nodes[i_star]
        m.vec_id = normalize((1 - self.alpha) * m.vec_id + self.alpha * e_id)
        m.meta_hist.update(meta, quality)
        m.meta_hist.hit_count += 1
        m.update_rep_images(image_path, quality)  # quality 상위 3장 유지
        return

    # (D) New node creation (distant embedding + high quality)
    if best_sim < self.tau_new and quality >= self.q_new_min:
        new_node = MemoryNode(
            node_id=self.next_id(),
            vec_id=e_id,
            rep_images=[image_path],
            meta_hist=NodeMeta.from_single(meta, quality),
        )
        self.nodes.append(new_node)

    # (E) Bank management
    if len(self.nodes) > self.k_max:
        self._evict_or_merge()
```

### 4.2 `match(e_id) → MatchResult`

동일인 판정 (identity_builder의 `stable(t)` 대체).

```python
def match(self, e_id: np.ndarray) -> MatchResult:
    sims = [cos(e_id, m.vec_id) for m in self.nodes]
    i_star = argmax(sims)

    return MatchResult(
        stable_score=sims[i_star],
        best_node_id=self.nodes[i_star].node_id,
        top3=sorted(zip(range(len(sims)), sims), key=lambda x: -x[1])[:3],
    )
```

### 4.3 `select_refs(query) → RefSelection`

ComfyUI reference image 선택.

```python
def select_refs(self, query: RefQuery) -> RefSelection:
    # Query embedding 구성
    q = query.to_embedding()  # state vector 또는 text embedding

    # Softmax weighted selection
    sims = [cos(q, m.vec_id) for m in self.nodes]
    weights = softmax(np.array(sims) / self.temperature)

    # Anchor 최소 가중치 보장
    for i, node in enumerate(self.nodes):
        if node.is_anchor:
            weights[i] = max(weights[i], self.anchor_min_weight)
    weights /= weights.sum()  # re-normalize

    # Top-p node 선택
    top_indices = argsort(weights)[-self.top_p:]

    # Reference 조합
    anchor_refs = self._get_anchor_refs()          # 1~2장 (항상 포함)
    coverage_refs = self._get_coverage_refs(top_indices)  # 1~3장
    challenge_refs = self._get_challenge_refs(top_indices) # 0~1장

    return RefSelection(
        anchor_refs=anchor_refs,
        coverage_refs=coverage_refs,
        weights=weights[top_indices].tolist(),
        paths_for_comfy=anchor_refs + coverage_refs + challenge_refs,
    )
```

## 5. Bank Management

### Eviction (k > k_max 시)

우선순위:
1. `hit_count` 최저 + `quality_best` 최저 노드 제거
2. `last_updated_ms`가 가장 오래된 노드 제거 (LRU)

### Node Merge (bank 압축)

```python
def _compact(self):
    for a, b in combinations(self.nodes, 2):
        if cos(a.vec_id, b.vec_id) > self.tau_close:
            merged = self._merge_nodes(a, b)
            self.nodes.remove(a)
            self.nodes.remove(b)
            self.nodes.append(merged)
            break  # 한 번에 하나씩
```

### rep_images 관리

- 노드당 최대 3장
- quality 기준 정렬, 새 이미지가 기존 최저보다 높으면 교체
- 이미지 경로는 identity_builder 출력 기준 상대 경로

## 6. Query Embedding 구성

`select_refs()`의 query는 **생성하려는 이미지의 맥락**을 표현한다.

### 방식 1: State Vector (기본)

```python
# 포즈/표정/조명 조건을 수치 벡터로 인코딩
q = concat(
    pose_vector,        # yaw, pitch, roll (정규화)
    expression_vector,  # mouth_open, eye_open, smile (0~1)
    lighting_vector,    # brightness, contrast (정규화)
)
# → 저차원 (8~16D), Face-ID 공간과 직접 비교 불가
# → 별도 projection layer 또는 nearest-bucket 매칭으로 변환
```

### 방식 2: Text Query (실험적)

```python
# CLIP text encoder로 자연어 쿼리 → embedding
q = clip_text_encode("smiling person in dark lighting, side view")
# → CLIP 공간이므로 Face-ID와 직접 비교 불가, cross-modal 매칭 필요
```

### 방식 3: Bucket 기반 매칭 (실용적)

```python
# query 조건과 각 노드의 meta_hist 분포 매칭
# embedding 비교 없이 메타데이터 기반 선택
scores = [node.meta_hist.coverage_score(target_buckets) for node in self.nodes]
```

초기 구현은 **방식 3 (bucket 기반)**으로 시작. embedding 기반 query는 데이터 확보 후 실험.

### Anchor 최소 가중치 정책

Anchor 노드(정면 고품질)는 query에 관계없이 최소 가중치(`anchor_min_weight=0.15`)를 보장.
디퓨전 모델의 ID 유지에 anchor가 핵심이므로 query 조건이 극단적이어도 anchor가 빠지지 않도록 한다.

## 7. Configuration

| 파라미터 | 초기값 | 설명 |
|---------|--------|------|
| `k_max` | 10 | 인물당 최대 memory node 수 |
| `alpha` | 0.1 | EMA 업데이트 계수 (낮을수록 안정적) |
| `tau_merge` | 0.5 | merge 임계값 (이 이상이면 기존 노드에 합침) |
| `tau_new` | 0.3 | 새 노드 생성 임계값 (이 이하면 새 노드) |
| `tau_close` | 0.8 | 노드 간 merge 임계값 (이 이상이면 두 노드 합침) |
| `q_update_min` | TBD | 뱅크 업데이트 최소 품질 (미만이면 매칭만) |
| `q_new_min` | TBD | 새 노드 생성 최소 품질 |
| `temperature` | 0.1 | select_refs softmax 온도 |
| `top_p` | 3 | select_refs 상위 노드 수 |
| `anchor_min_weight` | 0.15 | Anchor 노드 최소 가중치 |

모든 threshold는 데이터 기반 튜닝 대상.

## 8. ComfyUI 연동 방식

### 원칙: 이미지 경로 전달 (Option A)

```
identity_memory.select_refs(query)
        │ paths_for_comfy = [img1.jpg, img2.jpg, ...]
        ▼
comfy_bridge
        │ workflow.json 내 reference image 경로 교체
        ▼
ComfyUI (InstantID / IP-Adapter / PuLID 노드)
```

임베딩 직접 주입 금지 이유:
- Memory bank 임베딩(ArcFace)과 ComfyUI 내부 임베딩(CLIP-Vision)은 공간/차원이 다름
- 직접 주입은 효과 없거나 깨짐

### Reference 조합 비율

```
Anchor : Coverage : Challenge = 20 : 70 : 10 (기본값)
```

| 상황 | 조정 |
|------|------|
| ID drift 잦음 | Anchor 비율 ↑ (30:60:10) |
| 표정/각도 다양성 부족 | Coverage 비율 ↑ (15:75:10) |
| 어두운 조건 빈번 | Challenge 비율 ↑ (20:65:15) |

간접 가중: ComfyUI가 weight tensor를 직접 받지 못하는 경우,
이미지 수/반복으로 가중 (예: anchor 2장, coverage 1장).

## 9. identity_builder와의 관계

### Builder → Memory 데이터 흐름

```python
# builder가 이미지를 저장할 때마다
memory.update(
    e_id=frame.face_id_embedding,
    quality=frame.quality_score,
    meta={"yaw_bin": frame.yaw_bin, "pitch_bin": frame.pitch_bin, ...},
    image_path=saved_path,
)
```

### Memory → Builder 안정성 점수

```python
# builder의 ID stability check
result = memory.match(frame.face_id_embedding)
if result.stable_score < tau_id:
    exclude_from_candidates(frame)
```

초기 버전에서는 builder의 medoid prototype 사용,
후기 버전에서 memory bank의 `match()` API로 대체.

## 10. vpx 의존성

| 패키지 | 용도 |
|--------|------|
| vpx-sdk | 공유 타입 |
| vpx-face-detect | Face-ID 임베딩 (ArcFace recognition model) |

identity_memory 자체는 **임베딩 모델을 직접 실행하지 않음**.
builder가 추출한 임베딩을 받아서 저장/검색만 수행.

## 11. Prompt Encoder 진화 (장기 로드맵)

현재 memory bank는 **이미지 경로 선택**(Level 1)만 수행하지만,
장기적으로 디퓨전 모델의 conditioning에 더 깊이 관여할 수 있다.

### Level 1: Retrieval-conditioned Generation (현재 목표)

```
memory_bank.select_refs(query)
    → image paths
    → ComfyUI InstantID/IP-Adapter/PuLID 노드에 전달
```

- Memory bank는 **어떤 이미지를 보여줄지**만 결정
- 디퓨전 모델 내부에 개입하지 않음
- 구현이 단순하고 ComfyUI 생태계와 호환

### Level 2: Condition Token Cross-attention (실험적)

```
memory_bank → memory embeddings → cross-attention layer → diffusion U-Net
```

- Memory node의 임베딩을 디퓨전 모델의 **추가 conditioning token**으로 주입
- IP-Adapter의 확장 버전: 단일 이미지 대신 다중 memory node embedding 사용
- 각 노드의 weight를 attention score에 반영
- **요구 사항**: 커스텀 ComfyUI 노드 또는 모델 수정 필요

### Level 3: MoE Expert Selection (연구)

```
memory_bank → router → expert selection → mixture generation
```

- 각 memory node를 하나의 "expert"로 취급
- Router가 생성 조건에 따라 expert 조합을 결정
- Mixture of Experts 패턴으로 다양한 상태의 이미지 생성

**현재는 Level 1에만 집중. Level 2 이상은 Level 1의 한계가 명확해진 후 탐색.**

## 12. 구현 계획

### Phase 1: Memory Node + Update Logic

- [ ] MemoryNode, NodeMeta 데이터클래스 정의
- [ ] MemoryBank 클래스 (load/save JSON)
- [ ] `update()` 구현 (quality gate, nearest search, EMA merge, new node)
- [ ] Bank management (eviction, node merge)
- [ ] 단위 테스트 (merge/create/evict 시나리오)

### Phase 2: Match API

- [ ] `match()` 구현 (stable_score, top-3)
- [ ] identity_builder 연동 포인트 정의
- [ ] 단위 테스트 (stability 판정 정확도)

### Phase 3: select_refs API

- [ ] RefQuery 타입 정의 (state vector, text embedding)
- [ ] `select_refs()` 구현 (softmax selection, ref 조합)
- [ ] Anchor/Coverage/Challenge 조합 로직
- [ ] 단위 테스트 (query → 올바른 ref 선택 확인)

### Phase 4: 통합 + 고도화

- [ ] identity_builder와 end-to-end 연동 테스트
- [ ] comfy_bridge 연동 (workflow JSON 경로 교체)
- [ ] 버전 마이그레이션 (임베딩 모델 변경 시 뱅크 재구축)
- [ ] 운영 메트릭: 노드 수 안정성, stable_score 분포
