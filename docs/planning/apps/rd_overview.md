# 탑승 영상 비전 분석 및 맥락 결합 AI 개인화 콘텐츠 자동 생성 시스템

> 981파크가 고객의 탑승 경험을 바라보고 재해석한다

## 영상에서 의미 있는 순간을 찾는 '지능형 비전 분석'

탑승 영상 속 고객의 얼굴·표정·동작을 읽고, 2분 전체에서 가장 의미 있는 순간만 자동으로 골라내는 기술

- **멀티모달 특징 추출:** 얼굴 검출·추적(Face Detection & Tracking)과 시맨틱 파싱(Semantic Parsing)으로 눈·입·머리카락을 픽셀 단위로 분리하고, Facial Action Unit 분석과 6-DoF 헤드포즈 추정(Head Pose Estimation)으로 감정 변화와 시선을 측정하며, Vision-Language Model 기반 미학 평가(Aesthetic Assessment)로 사진 품질을 다차원으로 판단하는 기술
- **적응형 평가 & 다중 품질 게이트:** 영상별 통계적 정규화(Statistical Normalization)로 환경 편차를 보정하고, 선명도·노출·대비·세그멘테이션 커버리지 등 10가지 이상 조건을 계층적으로 검사하여 품질 기준 미달 프레임을 사전 필터링하는 Multi-gate Quality Control 기술
- **시계열 하이라이트 선별 & 임베딩 검증:** 품질·감정 점수를 시간축 평활화(Temporal Smoothing)한 뒤 통계적 피크 탐지(Peak Detection)로 절정 구간을 추출하고, Feature Embedding의 프레임 간 변화량 분석으로 극적인 순간을 교차 검증하는 기술

## 사람을 기억하고 맥락을 담는 '개인화 생성 시스템'

같은 사람의 다양한 모습을 체계적으로 기억하고, 탑승 맥락을 반영하여 안전하게 콘텐츠를 생성하는 기술

- **듀얼 임베딩 메모리 & 다양성 관리:** Face Recognition Embedding으로 동일 인물을 판단하고 General Vision Embedding(CLIP/DINOv2 계열)으로 상황 차이를 측정하여, 각도·회전·표정·조명의 다차원 버킷(Multi-dimensional Bucketing)으로 이미지를 분류하고 생성 모델 레퍼런스의 조건별 커버리지를 정량 관리하는 기술
- **맥락 파라미터 결합 & 참조 기반 생성:** 주행 속도·날씨·파크 이벤트·동승자 반응을 Diffusion Model의 Text/Context Conditioning으로 변환하고, ControlNet 및 IP-Adapter 계열 참조 기반 생성(Reference-guided Generation)으로 실제 얼굴을 참조하여 인물 동일성(Identity Preservation)을 유지하면서 배경·분위기만 개인화하는 기술
- **얼굴 마스킹 & 유사도 검증:** Semantic Segmentation으로 얼굴 영역을 생성 대상에서 분리(Face Masking)하고, Cosine Similarity를 Embedding Space에서 측정하여 기준 미달 시 즉시 폐기하는 얼굴 왜곡 방지 안전장치
