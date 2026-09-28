# 🌌 NVIDIA Cosmos 전수조사 & 활용 분석 (한국어 정리본)

> 작성일: 2026-09-28
> 작성: 카리나 (Claude Code 개발 파트너) 💖
> 대상 레포: **https://github.com/bmshin94/cosmos**
> 원본 레포: **https://github.com/NVIDIA/cosmos**

---

## 📑 목차

1. [레포 정체 & 전수조사](#1-레포-정체--전수조사)
2. [쉬운 설명 (비유편)](#2-쉬운-설명-비유편)
3. [Q&A 7문답](#3-qa-7문답)
4. [수익화 아이디어](#4-수익화-아이디어)
5. [참고 링크 모음](#5-참고-링크-모음)

---

## 1. 레포 정체 & 전수조사

### 1-1. 한 줄 정의

> **NVIDIA Cosmos** = 로봇·자율주행·스마트인프라 같은 **물리 세계(Physical AI)** 를
> AI가 **이해하고(Reasoner)** **상상해서 생성하는(Generator)** 오픈 월드 모델 플랫폼.

이 레포(`bmshin94/cosmos`)는 `NVIDIA/cosmos`의 포크이며,
**모델 코드가 아니라 "쿡북 + 문서 + 평가도구" 전용 레포**다.

| 구성요소 | 위치 |
| --- | --- |
| 문서 / 쿡북 (이 레포) | https://github.com/NVIDIA/cosmos |
| 실제 프레임워크 (학습·추론 코드) | https://github.com/NVIDIA/cosmos-framework |
| 데이터 큐레이션 | https://github.com/NVIDIA/cosmos-curator |
| 자동 평가 시스템 | https://github.com/NVIDIA/cosmos-evaluator |
| 모델 체크포인트 | https://huggingface.co/collections/nvidia/cosmos3 |

### 1-2. Cosmos 3 모델 개요

**Mixture-of-Transformers(MoT)** 아키텍처.
AR 트랜스포머(추론, causal attention) + Diffusion 트랜스포머(생성, full attention)를
하나의 모델에 통합. 3D mRoPE로 공간·시간 정보를 함께 인코딩.

#### 두 가지 런타임 서피스

| 서피스 | 입력 | 출력 | 용도 |
| --- | --- | --- | --- |
| **Reasoner** | 텍스트, 비전 | 텍스트 | 월드 이해, 그라운딩, 물리 추론, 태스크 플래닝, 행동 예측 |
| **Generator** | 텍스트, 비전, 사운드, 액션 | 비전, 사운드, 액션 | 월드 생성/시뮬레이션, 미래 예측, 합성 데이터 생성, 정책 학습 |

#### 모델 패밀리

| | Cosmos3-Super | Cosmos3-Nano | Cosmos3-Edge |
| --- | --- | --- | --- |
| **크기** | 64B | 16B | 4B |
| **권장 HW** | H200 / B200 / GB200 | RTX Pro 6000 / H100 / B200 | Jetson AGX Orin / Thor / RTX Pro 6000 |
| **출력** | Text/Image/Video/Sound/Action | Text/Image/Video/Sound/Action | Text/Image/Video/Action |
| **용도** | 최고 품질, 증류 교사 모델 | 밸런스형, 파인튜닝 베이스 | 엣지, 실시간 로봇 정책 |

> 💡 Super에는 **4Step 변형**(Text2Image-4Step, Image2Video-4Step)이 있어
> 동일 품질 대비 **17~25배 속도 향상**.

### 1-3. 폴더 구조 전수조사

```
cosmos/
├── README.md                 # 72KB — 사실상 이 레포의 본체
├── inference_benchmarks.md   # 59KB — GPU별 추론 속도 벤치마크
├── CLAUDE.md                 # 카리나 페르소나 가이드 (사용자 추가)
├── CONTRIBUTING.md / SECURITY.md / RELEASE.md
├── LICENSE                   # OpenMDW-1.1 (상업 이용 제한 없음)
├── .github/
│   ├── ISSUE_TEMPLATE/
│   └── workflows/validate-notebooks.yml   # 노트북 JSON 검증 CI
├── cookbooks/cosmos3/
│   ├── README.md (30KB) + cosmos3-model-architecture.png
│   ├── generator/
│   │   ├── audiovisual/      # 영상+사운드 생성
│   │   │   ├── run_with_diffusers.ipynb
│   │   │   ├── run_with_vllm_omni.ipynb
│   │   │   ├── run_with_sglang.ipynb
│   │   │   ├── run_with_trt_llm.ipynb
│   │   │   ├── run_with_nim.ipynb
│   │   │   ├── run_with_cosmos_framework.ipynb
│   │   │   ├── finetune/toml/sft_config/
│   │   │   ├── distill/toml/
│   │   │   └── assets/ (prompts, negative_prompts, images, videos)
│   │   ├── action/           # 🤖 로봇 파트
│   │   │   ├── run_fd_*.ipynb      # Forward Dynamics (행동→미래영상)
│   │   │   ├── run_id_*.ipynb      # Inverse Dynamics (영상→행동 역추론)
│   │   │   ├── run_policy_*.ipynb  # Policy (상황→행동 생성)
│   │   │   ├── finetune/toml/sft_config/
│   │   │   └── assets/
│   │   │       ├── droid_lerobot_example/
│   │   │       ├── bridge_lerobot_example/
│   │   │       ├── fractal_lerobot_example/
│   │   │       ├── agibotworld_beta_lerobot_example/
│   │   │       ├── robomind_lerobot_example/ (franka, franka_dual, ur)
│   │   │       ├── umi_lerobot_example/
│   │   │       ├── human_hand_pose_lerobot_example/
│   │   │       └── egocentric_hand_action_example/
│   │   └── transfer/         # ControlNet 스타일 영상 변환
│   │       └── assets/ (blur, depth, edge, seg, multi_control, wsm)
│   ├── reasoner/
│   │   ├── reasoner_prompt_guide.md
│   │   ├── run_with_transformers / vllm / tensorrt_llm / sglang / nim .ipynb
│   │   └── finetune/toml/sft_config/
│   └── nim/                  # 🐳 NIM 컨테이너 운영 문서 (13개 md)
│       ├── AGENTS.md, README.md, deployment.md, configuration.md,
│       │   operations.md, prerequisites.md, support-matrix.md,
│       │   api-reference.md, generation.md, reasoning.md, action.md,
│       │   transfer.md, bring-your-own-checkpoint.md, helm.md
│       ├── examples/ (inspect_profile.py 등)
│       └── .agents/skills/cosmos3-nim-user/SKILL.md   # ⭐ 에이전트 스킬!
└── evaluation/cosmos3/
    ├── generator/
    │   ├── physics_iq/       # 물리법칙 이해도 벤치마크
    │   ├── paibench_c/       # Physical AI Bench (Consistency)
    │   ├── paibench_g/       # Physical AI Bench (Generation)
    │   ├── rbench/           # VLM judge 기반 평가
    │   └── unigenbench/      # 통합 생성 벤치마크
    └── reasoner/vlmevalkit/  # VLMEvalKit 포크 + cosmos_eval 모듈
```

**파일 통계:** Python 775 · JSON 97 · Markdown 63 · **Jupyter Notebook 34** ·
mp4 40 · parquet 28 · sh 27 · toml 15

### 1-4. ⭐ 핵심 발견: 에이전트 스킬 파일

`cookbooks/cosmos3/nim/.agents/skills/cosmos3-nim-user/SKILL.md`

```yaml
---
name: cosmos3-nim-user
description: Guide customers through Cosmos3 Certified NIM deployment,
  endpoint verification, API example selection, hardware compatibility,
  and troubleshooting. Use for operating or calling the NIM from the
  public cookbooks/cosmos3/nim directory; do not use for maintaining
  the documentation itself.
license: OpenMDW-1.1
---
```

**배울 점 (스킬 작성 레퍼런스로서의 가치):**

- `description`에 **"쓸 때 / 쓰지 말아야 할 때"** 를 모두 명시 → 트리거 정확도 향상
- 워크플로를 **번호 순서로 강제** (1. 경로 파악 → 2. 기존 엔드포인트 → 3. 신규 배포)
- 검증 명령을 코드블록으로 고정 (`/v1/health/ready`, `/v1/metadata`)
- **보안 가드레일 내장**: "`NGC_API_KEY` 값을 묻지 마라, 설정 여부만 확인하라"
- **파괴적 작업 방지**: "컨테이너/캐시 삭제 전 반드시 사용자에게 물어라",
  "실패한 컨테이너는 로그 확인 전까지 보존하라"
- 판단 기준을 **구체적 수치**로: `NIM_GPU_MEMORY_UTILIZATION=0.80/0.70`,
  "디바이스 간 VRAM을 합산하지 마라", "`MemAvailable`에 캐시를 더하지 마라"
- 문서를 **source of truth**로 지정 → 환각 방지

> 참고: README의 Ecosystem 표에 따르면 `NVIDIA/cosmos-framework`는
> `.agents/skills/` 와 `.claude/skills/` 를 **공식 제공**한다
> (Claude Code, Codex CLI, Cursor 등 AGENTS.md 인식 에이전트 대상).

### 1-5. 주요 활용 시나리오

| 상황 | Cosmos의 역할 |
| --- | --- |
| 로봇 학습 데이터 부족 | 실물 로봇 없이 **합성 학습 영상 대량 생성** (핵심 용도) |
| 자율주행 코너케이스 | 위험 시나리오를 영상으로 합성 |
| 영상 이해 자동화 | 자동 캡션 + 사건 타임스탬프 + 공간 그라운딩 |
| 스마트팩토리 | 작업자 행동 분석, 이상 동작 감지, 안전 판정 |
| 월드 시뮬레이션 | 행동 조건부 미래 롤아웃 예측 (Forward Dynamics) |
| 물리 타당성 검사 | 생성/촬영 영상의 물리적 타당성 분류 |

### 1-6. 사용자에게 주는 실질적 가치

**도움 되는 점**

1. 동일 기능을 **Diffusers / vLLM-Omni / SGLang / TensorRT-LLM / NIM** 5가지로 구현 →
   "연구 코드 → 프로덕션 서빙" 전환 학습 최고 교재
2. `SKILL.md` = NVIDIA 공식 **프로덕션급 에이전트 스킬 작성 레퍼런스**
3. **OpenMDW-1.1** 라이선스 → 상업 이용 제한 없음 (저작권 고지만 유지)
4. `inference_benchmarks.md` → GPU 견적 산출 자료
5. NVIDIA가 튜닝한 프롬프트 자산 (`assets/prompts/`, `reasoner_prompt_guide.md`)

**제약**

- GPU 필수. 최소 Edge(4B)도 Jetson AGX Orin/Thor급 이상
- 이 레포 단독 실행 불가 → `cosmos-framework` + HF 체크포인트 필요
- 체크포인트 용량 수십~수백 GB

---

## 2. 쉬운 설명 (비유편)

### 2-1. 한 문장

> **Cosmos = "AI를 위한 가상 현실 게임기"**
> 게임의 물리엔진을 AI가 **영상으로 상상해서** 만들어주는 버전.

### 2-2. 로봇 학원의 "가상 운동장"

```
[기존 방식]                      [Cosmos 방식]
실제 로봇 구매 (수천만원)          "로봇이 컵 집는 영상"이라고 글로 씀
  → 사람이 1만 번 시연             → AI가 영상 1만 개 생성
  → 6개월 소요                     → 그걸로 로봇 학습
  → 로봇 고장 시 중단              → 실제 로봇 0대로도 가능
```

이것이 **합성 데이터 생성(Synthetic Data Generation)**, Cosmos의 존재 이유.

### 2-3. 두 명의 알바생

**🕵️ Reasoner — "눈치 100단 관찰러"** (영상 → 말)

> "0:03에 작업자가 상자를 들었고, 0:07에 바닥 기름으로 미끄러질 위험이 있습니다.
> 다음 행동은 상자를 선반에 놓을 것으로 예상됩니다."

가능: 영상 요약 / 사건 시각 탐지 / 2D 바운딩박스 / 물리 타당성 판정 / 다음 행동 예측

**🎨 Generator — "상상력 만렙 화가"** (말 → 영상)

가능: T2I / T2V / I2V / V2V / 사운드 동기 생성 / 로봇 액션 시퀀스 생성

### 2-4. 이 폴더는 "설명서만 든 상자"

| 구성 | 비유 |
| --- | --- |
| 이 레포 | 요리책 + 레시피 + 맛 평가표 |
| `cosmos-framework` | 실제 주방 도구 (코드 본체) |
| HuggingFace 모델 | 재료 (수십~수백 GB) |
| GPU | 불 (없으면 요리 불가) |

### 2-5. 왜 같은 기능을 5가지 방법으로?

| 방식 | 비유 | 언제 |
| --- | --- | --- |
| Diffusers | 집밥 | 연구·실험·코드 분석 |
| vLLM-Omni / SGLang | 푸드트럭 | 다수에게 API 서빙 |
| TensorRT-LLM | 특수 조리기구 | 최고 속도 최적화 |
| NIM | 밀키트 | Docker 한 줄로 즉시 서비스 |
| Cosmos Framework | 전문 주방 | 학습·파인튜닝 풀코스 |

### 2-6. 3줄 요약

1. Cosmos = 물리 세계를 **이해(Reasoner)** 하고 **상상(Generator)** 하는 NVIDIA AI 모델 가족
2. 이 레포 = 그 모델 사용법을 알려주는 **공식 교과서** (코드 본체는 별도 레포)
3. 주 용도 = 로봇/자율주행 학습용 **합성 영상 데이터 무한 생산**

---

## 3. Q&A 7문답

### Q1. 설치 및 사용법

#### 하드웨어 요구사항

| 모델 | 최소 GPU |
| --- | --- |
| Cosmos3-Edge (4B) | Jetson AGX Orin/Thor, RTX Pro 6000 |
| Cosmos3-Nano (16B) | H100, RTX Pro 6000 (96GB) |
| Cosmos3-Super (64B) | H200 / B200 / GB200 |

> 일반 게이밍 PC로는 사실상 어려움 → 클라우드 GPU 대여(RunPod, Lambda Labs,
> Vast.ai, AWS/GCP) 권장.

#### 경로 A — Diffusers (연구·실험)

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate

uv pip install --torch-backend=auto \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate av cosmos_guardrail huggingface_hub \
  imageio imageio-ffmpeg torch torchvision transformers

export HF_TOKEN=<your_token>
```

```python
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0
)

result = pipe(
    prompt="A mobile robot navigates a warehouse aisle and stops at a shelf.",
    num_frames=189,            # 24fps 기준 약 7.9초
    height=720, width=1280, fps=24,
    num_inference_steps=35,
    guidance_scale=6.0,
    enable_sound=False,
    generator=torch.Generator(device="cuda").manual_seed(1234),
)
export_to_video(result.video, "cosmos3_t2v.mp4", fps=24, macro_block_size=1)
```

**Diffusers 모드:** `text-to-image` / `text-to-video` / `image-to-video` /
`text-to-video-with-sound`

#### 경로 B — NIM (프로덕션)

```bash
export NGC_API_KEY=<your_key>
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin

docker run --rm --gpus all \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/cosmos3:2.0.0

# 헬스체크 + 런타임 확인
curl -fsS http://localhost:8000/v1/health/ready
curl -fsS http://localhost:8000/v1/metadata | python3 -m json.tool
```

| 서피스 | 엔드포인트 |
| --- | --- |
| Generator | `POST /v1/infer` |
| Reasoner | `POST /v1/chat/completions` (OpenAI 호환) |

주요 환경변수: `NIM_MODEL_VARIANT` (nano / nano-droid / super / super-t2i /
super-t2i-4step / super-i2v / super-i2v-4step), `NIM_PERF_PROFILE`
(latency / throughput), `NIM_GPU_MEMORY_UTILIZATION`

#### 경로 C — 이 레포 노트북 그대로

```bash
git clone https://github.com/bmshin94/cosmos
cd cosmos/cookbooks/cosmos3/generator/audiovisual
jupyter lab run_with_diffusers.ipynb
```

> GPU가 없어도 34개 노트북을 **읽는 것만으로** 학습 가치가 크다.

#### 자주 나는 에러

| 증상 | 해결 |
| --- | --- |
| `torch.cuda.is_available()` = False | 드라이버 구버전. `--torch-backend=cu128` 등 명시 핀 |
| `libxcb.so.1: cannot open shared object file` | `libxcb1` 계열 시스템 패키지 설치 |
| `uv` sync 오류 | Python 3.13 + `--managed-python` 확인 |
| 첫 실행이 멈춘 듯 느림 | 정상. 모델 다운로드 + 35스텝 디퓨전 |

---

### Q2. 플러그인 / 스킬 / MCP 중 무엇인가?

| 구분 | 해당 여부 | 설명 |
| --- | --- | --- |
| 플러그인 | ❌ | Claude Code 플러그인 구조 없음 |
| MCP 서버 | ❌ | MCP 서버 구현 없음 |
| 스킬 | ⚠️ 부분적 | `nim/.agents/skills/cosmos3-nim-user/SKILL.md` 1개 존재 |
| **실제 정체** | ✅ | **AI 모델 문서 + 쿡북 레포** |

단, 포함된 스킬 1개는 품질이 매우 높아 **스킬 작성 레퍼런스로서 가치가 큼**
(1-4 섹션 참조). 또한 `NVIDIA/cosmos-framework`는 `.claude/skills/`를 공식 제공한다.

---

### Q3. API 토큰이 필요한가?

| 토큰 | 용도 | 비용 |
| --- | --- | --- |
| `HF_TOKEN` | 모델 체크포인트 다운로드 | 무료 (가입 필요) |
| `NGC_API_KEY` | NIM 이미지 pull (`nvcr.io`) | 무료 (가입 필요) |
| `RBENCH_VLM_API_KEY` | rbench 평가의 VLM judge | **유료** |
| `OPENAI_API_KEY` | 위 항목의 폴백 변수 | **유료** |

**로컬 서버 호출 시에는 키가 불필요:**

```python
client = openai.OpenAI(api_key="not-used", base_url="http://127.0.0.1:8000/v1")
client = openai.OpenAI(api_key="EMPTY",    base_url="http://localhost:8000/v1")
```

> 한 번 모델을 내려받으면 이후 완전 오프라인 운영 가능 → **토큰 과금 0원**.
> 단, 일부 모델/데이터셋은 HF에서 라이선스 동의 필요.
> 기본 탑재된 `cosmos_guardrail` 안전필터도 HF 인증을 요구한다.

---

### Q4. 왜 GitHub에서 유명한가?

1. **NVIDIA 브랜드 파워** — GPU 시장 지배 기업의 직접 오픈소스 배포
2. **Physical AI가 현재 최고 화두** — 로봇/휴머노이드/자율주행 시장 폭발기
3. **라이선스가 매우 관대** — OpenMDW-1.1, 상업 이용 제한 없음 (Llama식 사용자수 제한 없음)
4. **비정상적으로 높은 문서 품질** — README 72KB, 벤치마크 59KB, 노트북 34개, NIM 문서 13개
5. **올인원 모델** — VLM + 영상생성 + 월드시뮬레이터 + 로봇정책을 한 모델로, 4B~64B 선택 가능
6. **업계 최대 난제 타격** — 로봇 AI의 병목은 알고리즘이 아닌 **데이터 희소성**
7. **에코시스템 전략** — cosmos / cosmos-framework / cosmos-curator / cosmos-evaluator로 전체 파이프라인 커버

> 정확한 star 수는 세션 권한 범위(`bmshin94/cosmos`) 밖이라 미확인.
> 위 분석은 레포 내용·구조·라이선스·포지셔닝 기반.

---

### Q5. 로컬 에이전트 구축에 도움이 되는가?

**직접 활용은 어려움:** GPU 장벽, 도메인 특화(텍스트 에이전트와 결이 다름), 모델 크기

**간접 활용은 매우 유용:**

1. **에이전트 스킬 작성 레퍼런스** (가장 큰 가치) — `SKILL.md` 구조 그대로 차용 가능
   - "쓸 때/안 쓸 때" 명시, 번호 순 워크플로, 보안 가드레일, 파괴적 작업 확인 규칙,
     수치 기반 판단 기준, 문서 기반 그라운딩
2. **`AGENTS.md` 패턴** — 디렉터리별 컨텍스트 주입 (CLAUDE.md와 동일 역할)
3. **로컬 서빙 아키텍처 학습** — vLLM / SGLang / TensorRT-LLM 세팅 실전 예제
4. **OpenAI 호환 API 패턴** — 공급자 교체 가능한 추상화 설계의 정석

**추천 로드맵**

```
1단계  SKILL.md / AGENTS.md 정독 → 스킬 작성법 체득
2단계  NVIDIA/cosmos-framework 의 .claude/skills/ 분석
3단계  내 프로젝트용 스킬 작성 (동일 구조)
4단계  Ollama + 소형 모델로 로컬 에이전트 프로토타입
5단계  GPU 확보 시 Cosmos-Edge 연동 → "영상 이해 에이전트"
```

---

### Q6. 수익화 가능성

가능. OpenMDW-1.1이 상업 이용을 허용한다. 상세는 4장 참조.

---

### Q7. React나 PHP로 만들 수 있는가?

**모델 추론 자체는 불가** (CUDA / PyTorch / 대용량 VRAM 필요).
**서비스 레이어는 100% 가능** — NIM이 HTTP API 서버이므로 언어 무관.

```
┌─────────────────┐   HTTP    ┌──────────────┐
│  React (UI)     │ ────────▶ │  PHP / Node  │
│  프롬프트 입력    │           │  인증 / 과금  │
│  영상 플레이어    │ ◀──────── │  잡 큐 관리   │
└─────────────────┘           └──────┬───────┘
                                     │ HTTP
                                     ▼
                          ┌──────────────────────┐
                          │  Cosmos NIM (GPU)    │
                          │  /v1/infer           │
                          │  /v1/chat/completions│
                          └──────────────────────┘
```

#### PHP 예시 (Reasoner — OpenAI 호환)

```php
<?php
$ch = curl_init('http://gpu-server:8000/v1/chat/completions');
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
    CURLOPT_POSTFIELDS => json_encode([
        'model' => 'cosmos3-reasoner',
        'messages' => [[
            'role' => 'user',
            'content' => [
                ['type' => 'video_url', 'video_url' => $videoUrl],
                ['type' => 'text', 'text' => '이 영상의 주요 사건을 타임스탬프와 함께 알려줘'],
            ],
        ]],
        'temperature' => 0.7,
        'top_p' => 0.8,
    ]),
]);
$result = json_decode(curl_exec($ch), true);
echo $result['choices'][0]['message']['content'];
```

#### React 예시 (Generator — 비동기 잡 처리)

```jsx
const [jobId, setJobId] = useState(null);
const [videoUrl, setVideoUrl] = useState(null);

async function generate(prompt) {
  const res = await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, model_mode: 't2v' }),
  });
  const { id } = await res.json();
  setJobId(id);
}

useEffect(() => {
  if (!jobId) return;
  const t = setInterval(async () => {
    const r = await fetch(`/api/jobs/${jobId}`).then((r) => r.json());
    if (r.status === 'done') { setVideoUrl(r.url); clearInterval(t); }
  }, 3000);
  return () => clearInterval(t);
}, [jobId]);
```

#### 인프라 선택지

| 구성 | GPU | 비용 | 난이도 |
| --- | --- | --- | --- |
| A. 클라우드 GPU 대여 (RunPod/Lambda) | 시간당 과금 | 중간 | ⭐⭐ |
| B. build.nvidia.com API | 불필요 | 저렴 | ⭐ |
| C. 자체 GPU 서버 | 직접 구매 | 높음 | ⭐⭐⭐ |

> 권장 순서: **B → A → C**
> ⚠️ 영상 생성은 수십 초~수 분 소요 → 동기 요청 금지, **잡 큐 + 폴링/WebSocket** 필수.

---

## 4. 수익화 아이디어

> 전제: **OpenMDW-1.1** 라이선스는 상업 이용을 제한하지 않는다 (저작권 고지 유지 필요).

### 티어 1 — GPU 없이 즉시 가능 (초기자본 ~0원)

#### #1. 한국어 Physical AI 교육 콘텐츠
- 난이도 ⭐ / 초기비용 0원 / 예상 월 50~300만원
- 근거: Cosmos 문서가 전부 영어, 한국어 자료 거의 없음 → **선점 가능**

| 상품 | 가격대 | 채널 |
| --- | --- | --- |
| 유튜브 강의 시리즈 | 광고·협찬 | YouTube |
| 온라인 강의 | 5~15만원 | 인프런 / 클래스101 |
| 전자책 | 2~5만원 | 크몽 / 브런치 / 리디 |
| 유료 뉴스레터 | 월 1만원 | 스티비 / 메일리 |

실행: 노트북 34개 → 한국어 해설 블로그 시리즈(SEO 선점) → 유튜브화 → 유료 강의

#### #2. 도입 컨설팅 & 기술자문
- 난이도 ⭐⭐ / 초기비용 0원 / 건당 300~2,000만원
- 판매 항목: GPU 견적 컨설팅(`inference_benchmarks.md` 활용), PoC 설계·대행,
  파인튜닝 데이터셋 설계, 사내 교육(반나절 200~500만원)
- 전략: 콘텐츠로 전문가 포지션 확보 → 문의 자동 유입 → 컨설팅 전환

#### #3. 오픈소스 도구 → 스폰서/유료화
- Cosmos 프롬프트 라이브러리 (한국어 확장)
- **Cosmos GPU 계산기** 웹앱 (React) — "내 GPU로 무엇이 돌아가는가"
- NIM 배포 자동화 CLI
- 수익: GitHub Sponsors, Pro 버전

---

### 티어 2 — 클라우드 GPU 기반 (초기자본 50~500만원)

#### #4. 제조업 안전관리 영상분석 SaaS ⭐ 최우선 추천
- 난이도 ⭐⭐⭐ / 초기 300만원~ / 고객당 월 50~500만원

```
CCTV → Reasoner → "0:03 작업자 안전모 미착용"
                → "0:41 지게차 접근, 충돌 위험"
                → 자동 알림 + 리포트
```

| 기능 | 사용 Cosmos 기능 |
| --- | --- |
| 안전장비 착용 감지 | 2D grounding |
| 위험 상황 예측 | Situation Understanding + 다음 행동 예측 |
| 사고 영상 자동 리포트 | Caption + Temporal localization |
| 이상 동작 탐지 | Physical Plausibility Analysis |

강점: 중대재해처벌법 → 예산 집행 명분 / 기존 CCTV 활용 /
Edge(4B) 온프레미스 가능(보안 민감 공장 대응) / B2B 고객단가·유지율 높음

가격 모델: 카메라당 월 5~20만원 × 20~100대 = 고객당 월 100~2,000만원

#### #5. 커머스 상품영상 자동 생성 서비스
- 난이도 ⭐⭐⭐ / 초기 200만원~ / 건당 5,000~5만원
- Generator I2V로 상품 사진 1장 → 360도 회전 / 사용 장면 / 분위기 영상 + 사운드
- 타겟: 네이버 스마트스토어, 쿠팡 셀러 (영상 제작비 부담이 큰 층)
- React 프론트엔드에 최적

#### #6. 로봇 학습용 합성 데이터셋 판매
- 난이도 ⭐⭐⭐⭐ / 초기 500만원~ / 데이터셋당 수백~수천만원
- Cosmos의 본래 용도이자 최대 시장
- 레포의 `assets/*_lerobot_example/` 포맷을 그대로 따라 LeRobot 형식으로 패키징
- 틈새 전략: 한국 산업 특화 (반도체 웨이퍼 핸들링, 조선 용접, 식품 가공 등)

---

### 티어 3 — 본격 사업 (초기자본 1,000만원~)

#### #7. 버티컬 Physical AI 솔루션
- 특정 산업 1개에 집중해 Cosmos를 파인튜닝하여 판매
- 출발점: `finetune/toml/sft_config/`

| 버티컬 | 솔루션 |
| --- | --- |
| 의료 | 수술 영상 분석, 술기 교육 시뮬레이터 |
| 건설 | 현장 안전 + 공정률 자동 산출 |
| 농업 | 작물 상태 분석 + 수확 로봇 정책 |
| 외식 | 주방 위생 모니터링 |
| 폐기물 | 분류 로봇 학습 데이터 |

#### #8. Cosmos 매니지드 호스팅 (한국 리전)
- 데이터 국내 저장, 한국어 지원, 세금계산서 발행
- NIM이 Docker 기반이라 멀티테넌시 구성이 상대적으로 수월

---

### 추천 실행 로드맵

```
0~3개월    블로그/유튜브로 한국어 Cosmos 콘텐츠 선점 (비용 0)
           + "Cosmos GPU 계산기" 웹앱 배포 (React)

3~6개월    build.nvidia.com API로 MVP 1개 런칭
           → 추천: 커머스 영상 생성 (검증 빠름, B2C)

6~12개월   반응 검증 후 RunPod 등에 자체 NIM 배포 → 원가 절감
           + 콘텐츠 효과로 컨설팅 문의 수령 시작

1년~       B2B SaaS 전환 (제조업 안전관리) → 구독 매출 확보
```

### 리스크 및 대응

| 리스크 | 대응 |
| --- | --- |
| GPU 비용 부담 | Edge(4B) 우선, 배치 처리, 결과 캐싱 |
| NVIDIA의 직접 SaaS 진출 가능성 | 버티컬 특화 + 한국 로컬라이즈로 차별화 |
| 모델 버전 변경 속도 | OpenAI 호환 인터페이스로 추상화 (교체 가능 설계) |
| 생성물 저작권·윤리 | `cosmos_guardrail` 유지, 이용약관 명시 |
| 기술 난이도 | 티어 1부터 단계적 진입 |

> **핵심 원칙: 모델을 팔지 말고, 모델로 해결한 "문제"를 팔 것.**
> Cosmos 자체는 무료 오픈소스이므로 그 자체로는 수익이 되지 않는다.
> "공장 안전사고를 줄여준다"는 **기업이 비용을 지불하는 가치**다.

---

## 5. 참고 링크 모음

### 이 레포
- **현재 레포 (포크):** https://github.com/bmshin94/cosmos
- **원본 레포:** https://github.com/NVIDIA/cosmos

### NVIDIA Cosmos 에코시스템
- Cosmos Framework (실행·학습 코드, Agent Skills 제공): https://github.com/NVIDIA/cosmos-framework
- Cosmos Framework — Agent Skills: https://github.com/NVIDIA/cosmos-framework#agent-skills
- Cosmos Curator (데이터 큐레이션): https://github.com/NVIDIA/cosmos-curator
- Cosmos Evaluator (자동 평가): https://github.com/NVIDIA/cosmos-evaluator

### 모델 & 문서
- HuggingFace 컬렉션: https://huggingface.co/collections/nvidia/cosmos3
- Cosmos3-Super: https://huggingface.co/nvidia/Cosmos3-Super
- Cosmos3-Nano: https://huggingface.co/nvidia/Cosmos3-Nano
- Cosmos3-Edge: https://huggingface.co/nvidia/Cosmos3-Edge
- Cosmos3-Nano-Policy-DROID: https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID
- Diffusers 파이프라인 문서: https://huggingface.co/docs/diffusers/main/en/api/pipelines/cosmos3

### 연구 자료
- 기술 리포트(PDF): https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf
- 연구 페이지: https://research.nvidia.com/labs/cosmos-lab/cosmos3/
- 제품 페이지: https://www.nvidia.com/en-us/ai/cosmos/

### 배포 관련
- NVIDIA NGC 카탈로그: https://catalog.ngc.nvidia.com/
- build.nvidia.com (API 체험): https://build.nvidia.com/
- NIM 이미지: `nvcr.io/nim/nvidia/cosmos3:2.0.0`

### 레포 내부 주요 경로
| 경로 | 내용 |
| --- | --- |
| `README.md` | 전체 개요 · 퀵스타트 · 트러블슈팅 (72KB) |
| `inference_benchmarks.md` | GPU별 추론 벤치마크 (59KB) |
| `cookbooks/cosmos3/README.md` | 쿡북 인덱스 및 사전 준비 |
| `cookbooks/cosmos3/nim/` | NIM 배포·운영 문서 13종 |
| `cookbooks/cosmos3/nim/.agents/skills/cosmos3-nim-user/SKILL.md` | **에이전트 스킬 레퍼런스** |
| `cookbooks/cosmos3/reasoner/reasoner_prompt_guide.md` | Reasoner 프롬프트 가이드 |
| `evaluation/cosmos3/` | 벤치마크·평가 도구 |

---

*Generated by 카리나 with Claude Code 💖✨*
