# VisualBase - Claude Session Context

> 최종 업데이트: 2026-02-12
> 상태: **Phase 9 완료** (182 tests)

## 프로젝트 역할

**범용 미디어 I/O + 프로세스 인프라 미들웨어** (재사용 가능):
- 카메라/파일/RTSP 소스에서 프레임 스트리밍
- Ring Buffer로 메모리 효율적 버퍼링
- FFmpeg 기반 클립 추출
- IPC (FIFO, UDS, ZMQ PUB/SUB, ZMQ RPC) 통신 — 모든 프로세스 간 통신의 단일 소스

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
| `Frame` | 비디오 프레임 + 타임스탬프 | visualpath, momentscan |
| `RingBuffer` | 메모리 효율적 버퍼링 | 클립 추출 |
| `Clipper` | 프레임 범위 추출 | momentscan Action |
| `Trigger` | 이벤트 신호 타입 | visualpath Fusion |
| `IPC` | 프로세스 간 통신 | 분산 처리 |

## 디렉토리 구조

```
visualbase/
├── sources/       # FileSource, CameraSource, RTSPSource
├── core/          # Frame, RingBuffer, Sampler
├── packaging/     # Trigger, Clipper
├── ipc/           # FIFO, UDS, ZMQ PUB/SUB, ZMQ RPC
│   ├── interfaces.py   # ABCs: VideoReader/Writer, MessageSender/Receiver, RPCServer/Client
│   ├── factory.py      # TransportFactory (런타임 transport 선택)
│   ├── codec.py        # Frame ↔ dict (base64 JPEG) 직렬화
│   ├── _util.py        # check_zmq_available(), generate_ipc_address()
│   ├── fifo.py         # Named pipe transport
│   ├── uds.py          # Unix Domain Socket transport
│   ├── zmq_transport.py # ZMQ PUB/SUB transport
│   ├── zmq_rpc.py      # ZMQ REQ-REP RPC transport
│   └── messages.py     # OBS/TRIG message parsing
├── streaming/     # ProxyFanout
└── daemon.py      # ZMQ 데몬 모드
```

## IPC 아키텍처 (프로세스 간 통신)

visualbase.ipc는 모든 프로세스 간 통신의 단일 소스:

| 패턴 | 구현 | 용도 |
|------|------|------|
| Video streaming (1:N) | FIFO, ZMQ PUB/SUB | Ingest → Extractor 프레임 분배 |
| Message passing | UDS, ZMQ PUB/SUB | OBS/TRIG 메시지 교환 |
| RPC (1:1 동기) | ZMQ REQ-REP | Worker subprocess 격리 실행 |

### Transport 추상화

- ABC 인터페이스: VideoReader/Writer, MessageSender/Receiver, RPCServer/Client
- TransportFactory: 런타임 transport 선택
- Frame codec: Frame ↔ dict (base64 JPEG) 직렬화

### 경계

- **visualbase**: transport 계층 (소켓, 직렬화, 생명주기)
- **visualpath-isolation**: 애플리케이션 프로토콜 (ping/analyze/shutdown, Observation 직렬화)

## CLI 명령어

```bash
visualbase play <source>           # 통합 재생
visualbase daemon <source>         # ZMQ 데몬
visualbase clip <path> --time N    # 클립 추출
visualbase webrtc <source>         # 브라우저 스트리밍
```

## 테스트

```bash
cd ~/repo/monolith/portrait981
uv run pytest libs/visualbase/tests/ -v            # 182 tests
```

## 의존성

- 코어: opencv-python, numpy
- 옵션: pyzmq (zmq), aiortc (webrtc)

## 관련 패키지

- **visualpath**: visualbase를 기반으로 분석 프레임워크 제공
- **momentscan**: visualbase + visualpath를 사용하는 앱
