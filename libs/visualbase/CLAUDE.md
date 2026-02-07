# VisualBase - Claude Session Context

> 최종 업데이트: 2026-02-02
> 상태: **Phase 8 완료** (151 tests)

## 프로젝트 역할

**범용 미디어 I/O 라이브러리** (재사용 가능):
- 카메라/파일/RTSP 소스에서 프레임 스트리밍
- Ring Buffer로 메모리 효율적 버퍼링
- FFmpeg 기반 클립 추출
- IPC (FIFO, UDS, ZMQ) 통신

## 아키텍처 위치

```
┌─────────────────────────────────────────────────────────┐
│  범용 레이어                                             │
│  ┌─────────────┐      ┌─────────────┐                   │
│  │ visualbase  │ ───→ │ visualpath  │                   │
│  │ (미디어 I/O)│      │ (분석 프레임워크)               │
│  └─────────────┘      └─────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

## 핵심 제공 기능

| 모듈 | 제공 | 사용처 |
|------|------|--------|
| `Frame` | 비디오 프레임 + 타임스탬프 | visualpath, facemoment |
| `RingBuffer` | 메모리 효율적 버퍼링 | 클립 추출 |
| `Clipper` | 프레임 범위 추출 | facemoment Action |
| `Trigger` | 이벤트 신호 타입 | visualpath Fusion |
| `IPC` | 프로세스 간 통신 | 분산 처리 |

## 디렉토리 구조

```
visualbase/
├── sources/       # FileSource, CameraSource, RTSPSource
├── core/          # Frame, RingBuffer, Sampler
├── packaging/     # Trigger, Clipper
├── ipc/           # FIFO, UDS, ZMQ (VideoReader/Writer)
├── streaming/     # ProxyFanout
└── daemon.py      # ZMQ 데몬 모드
```

## CLI 명령어

```bash
visualbase play <source>           # 통합 재생
visualbase daemon <source>         # ZMQ 데몬
visualbase clip <path> --time N    # 클립 추출
visualbase webrtc <source>         # 브라우저 스트리밍
```

## 테스트

```bash
cd ~/repo/monolith/visualbase
uv run pytest tests/ -v            # 151 tests
```

## 의존성

- 코어: opencv-python, numpy
- 옵션: pyzmq (zmq), aiortc (webrtc)

## 관련 패키지

- **visualpath**: visualbase를 기반으로 분석 프레임워크 제공
- **facemoment**: visualbase + visualpath를 사용하는 앱
