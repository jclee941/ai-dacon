# skku-vqa

**2026 SKKU Multimodal AI Challenge (DACON #236722) · Multimodal Bias-Aware VQA 파이프라인**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Package](https://img.shields.io/badge/pkg-skku--vqa-blue)
![Status](https://img.shields.io/badge/status-active--development-yellowgreen)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#../LICENSE)

핵심 한국어 요약: 본 저장소는 2026 SKKU 멀티모달 AI 챌린지(DACON #236722) 제출을 위한 **편향-인지 멀티모달 VQA** 파이프라인입니다. 다수의 비전-언어 모델(InternVL3, LLaVA-OneVision, Qwen2.5-VL, Qwen3-VL)을 로컬 가중치로 로드하고, **편향 가드(bias-guarded)** 프롬프트 전략과 휴리스틱 앙상블을 통해 한국어 시각질의응답 태스크의 라벨을 예측합니다. 모든 추론은 오프라인/로컬에서 수행되며, 최종 라벨은 **LLM이 생성한 텍스트에서 파싱**해야 한다는 대회 규칙을 코드 수준에서 강제합니다.

English short summary: A local-only, offline pipeline for multimodal bias-aware VQA on the 2026 SKKU Multimodal AI Challenge. Loads open-weights VLMs (InternVL3, LLaVA-OneVision, Qwen2.5-VL, Qwen3-VL), produces 0-based option labels from model text outputs under strict competition rules, and bundles an ensemble + auto-submission flow targeted at the DACON leaderboard.

---

## 빠른 상태 / Quick Status

| 영역 | 상태 | 비고 |
| --- | --- | --- |
| 패키지 | `skku_vqa` 0.1.0 | `src/skku_vqa/`, Python ≥ 3.10 |
| 의존성 | torch ≥ 2.2, transformers ≥ 4.49, accelerate ≥ 0.34 | GPU 추론용 `vlm` 옵션 의존성 별도 |
| 구성 | YAML + Pydantic (`config.py`) | `defaults` 상속으로 딥-머지 |
| 추론 백엔드 | `vllm` (기본) / `transformers` (4-bit 지원) | `models/loader.py` 디스패치 |
| 오프라인 정책 | 강제 (외부 API 금지) | `.env`의 OPENAI/HF 토큰은 dev 전용 |
| GPU 가용성 | 원격 추론 호스트에서만 | 데스크톱 개발 호스트는 비어 있음 |
| 자동 제출 | systemd timer 기반 | `deploy/systemd/dacon-auto-submit.*` |
| 테스트 | `pytest -q` (pytest 설정은 `src`를 `pythonpath`에 포함) | `tests/` |

## 핵심 실행 흐름 / Operator Flow

1. `make setup` → `pip install -e ".[vlm,dev]"`
2. `make prepare` → `python scripts/prepare_data.py --config configs/default.yaml`
3. `python scripts/run_inference.py --config configs/experiment/<exp>.yaml [--limit N]` (원격 GPU 호스트에서)
4. `python scripts/make_submission.py --config configs/experiment/ensemble_v1.yaml` 또는 `scripts/reproduce_submission.py`
5. `python scripts/validate_submission.py --submission outputs/submissions/submission.csv`
6. (선택) `scripts/auto_submit_next_window.sh` + systemd timer로 다음 제출 윈도우에서 자동 업로드

> ⚠️ `Makefile`은 일부 config 파일을 가정하지만 실제 파일은 `configs/experiment/*.yaml`에 있습니다. **스크립트는 직접 호출**하세요.

---

## 목차 / Table of Contents

- [개요 / Overview](#개요--overview)
- [기능 / Features](#기능--features)
- [패키지 구성 / Package Contents](#패키지-구성--package-contents)
- [아키텍처 / Architecture](#아키텍처--architecture)
- [상태 및 규칙 / Status & Competition Rules](#상태-및-규칙--status--competition-rules)
- [먼저 읽을 파일 / First Files to Read](#먼저-읽을-파일--first-files-to-read)
- [설정 / Configuration](#설정--configuration)
- [빠른 시작 / Quickstart](#빠른-시작--quickstart)
- [명령어 레퍼런스 / Commands Reference](#명령어-레퍼런스--commands-reference)
- [진입점 / API & Entry Points](#진입점--api--entry-points)
- [테스트 및 로컬 개발 / Testing & Local Dev](#테스트-및-로컬-개발--testing--local-dev)
- [배포 / Deployment (systemd)](#배포--deployment-systemd)
- [기여 / Contributing](#기여--contributing)
- [유지보수자 / Maintainers](#유지보수자--maintainers)
- [참고 문서 / Further Reading](#참고-문서--further-reading)
- [라이선스 / License](#라이선스--license)

---

## 개요 / Overview

`skku-vqa`는 DACON #236722 한국어 멀티모달 VQA 태스크의 로컬·오프라인 추론 파이프라인입니다. 학습용 코드 없이 **이미 학습된 공개 가중치**만 사용하며, 다음을 한 번에 묶어 제공합니다.

- 모델 디스패처 (`models/loader.py`): `vllm`(기본), `transformers`(옵션 4-bit) 두 백엔드. 가족 식별자는 `llava*`, `internvl*`, `qwen3_vl*`, `qwen*`.
- 편향-인지 프롬프트 빌더 (`prompting/builders.py`, `prompting/templates.py`): `bias_guarded`, `evidence_only` 등 전략.
- 결정론적 파서 (`prompting/parsers.py`): 모델 출력에서 0-기반 옵션 인덱스만 추출. 대회 규칙을 코드로 강제.
- 앙상블 러너 (`inference/ensemble.py`): 다수결이 아닌 **합쳐진 텍스트에서 한 LLM이 라벨을 산출**하도록 강제.
- 모호성 처리 (`ambiguity/detector.py`, `ambiguity/unknown_mapper.py`): "판단 불가" 류 응답을 분석용으로 표식(local only).
- 제출 파이프라인 (`scripts/make_submission.py`, `submission/dacon.py`): CSV/`submission.json` 직렬화, 검증.

## 기능 / Features

- **다중 VLM 백엔드**: InternVL3-8B, LLaVA-OneVision (llava_ov), Qwen2.5-VL-7B, Qwen3-VL-8B. 동일 인터페이스 `models/base.py`로 추상화.
- **오프라인 강제**: 외부 추론 API 사용 금지. `models/loader.py`가 정책 위반 시 실패하도록 의도됨.
- **편향 가드 프롬프트**: `bias_guarded` 빌더는 옵션 제시 순서/길이 편향을 줄이는 지시문을 추가.
- **휴리스틱 앙상블**: 여러 모델의 raw 출력을 모아 단일 모델에게 라벨을 결정시키는 형태 (`inference/ensemble.py`).
- **스키마 검증**: `data/schema.py`, `data/validation.py`로 제출 직렬화 단계에서 컬럼/타입을 보장.
- **자동 제출**: `deploy/systemd/dacon-auto-submit.{service,timer}` + `scripts/auto_submit_next_window.sh`로 윈도우 기반 업로드.
- **로컬 검증 도구**: `scripts/validate_local.py`, `scripts/validate_submission.py`로 산출물 사전 점검.
- **재현 가능 모드**: `scripts/reproduce_submission.py` + `artifacts/final/config.yaml`로 잠금 구성 기반 재현.

---

## 패키지 구성 / Package Contents

| 경로 | 역할 |
| --- | --- |
| `src/skku_vqa/cli.py` | `skku-vqa` 콘솔 진입점 (`[project.scripts]` 등록) |
| `src/skku_vqa/config.py` | YAML+Pydantic 구성 로더, 딥 머지, 검증 |
| `src/skku_vqa/constants.py` | 경로 상수, `UNKNOWN_OPTION_PATTERNS`(분석 전용) 등 |
| `src/skku_vqa/models/` | `base.py` 인터페이스, `loader.py` 디스패치, `internvl3.py`/`llava_ov.py`/`llava_ov_transformers.py`/`qwen3_vl.py`/`qwen_vl.py` |
| `src/skku_vqa/inference/` | `predictor.py` (단일 모델 추론 파이프라인), `ensemble.py` (앙상블 러너) |
| `src/skku_vqa/evaluation/metrics.py` | 정확도/일치율 등 지표 |
| `src/skku_vqa/data/` | `dataset.py`, `schema.py`, `validation.py` |
| `src/skku_vqa/prompting/` | `builders.py`, `parsers.py`, `templates.py` |
| `src/skku_vqa/submission/dacon.py` | DACON 형식 직렬화, ID 매핑 |
| `src/skku_vqa/ambiguity/` | `detector.py`, `unknown_mapper.py` |
| `src/skku_vqa/utils/` | `io.py`, `seed.py` |
| `scripts/` | 데이터 준비/추론/제출/검증/원격 동기화 스크립트 |
| `configs/` | `default.yaml` 기본값 + `experiment/*.yaml` 실험 정의 |
| `artifacts/final/` | 잠금 최종 구성과 산출물 (`SCORES.md` 참조) |
| `artifacts/lora_adapters/` | LoRA 어댑터 (LoRA 사용 시) |
| `deploy/systemd/` | 자동 제출 unit 파일 |

> 본 저장소는 단일 Python 패키지 + 스크립트 셋 구조이며, 다른 제품군(봇, 워크플로 어댑터, README 생성기 등)을 포함하지 않습니다.

---

## 아키텍처 / Architecture

### 데이터 흐름 (요약)

| 단계 | 산출물 | 책임 모듈 |
| --- | --- | --- |
| 1. 데이터 준비 | `data/processed/*.parquet`(또는 동등한 형태) | `scripts/prepare_data.py`, `data/dataset.py` |
| 2. 단일 모델 추론 | `outputs/predictions/<exp>/{raw_outputs.jsonl.gz, csv, timing.json}` | `scripts/run_inference.py`, `inference/predictor.py`, `models/*` |
| 3. 앙상블 추론 | 동일 경로의 `<exp>_bgv2` 등 변형 산출물 | `scripts/run_inference.py --config ...ensemble`, `inference/ensemble.py` |
| 4. 제출 직렬화 | `outputs/submissions/submission.csv` | `scripts/make_submission.py`, `submission/dacon.py` |
| 5. 검증 | (대화형 검사 결과) | `scripts/validate_submission.py`, `data/validation.py` |
| 6. 자동 제출 | DACON 업로드 트리거 | `scripts/auto_submit_next_window.sh` + systemd timer |

### 요청 흐름

1. 호출자가 `scripts/run_inference.py --config <exp>.yaml`을 실행.
2. `config.py`가 `defaults`를 딥-머지한 뒤 Pydantic으로 검증.
3. `models/loader.py`가 `model.name` 접두사로 디스패치 → `vllm` 또는 `transformers` 엔진 생성.
4. `inference/predictor.py`가 한 샘플씩:
   1. `prompting/builders.py`로 시스템/유저 프롬프트 구성.
   2. 모델 호출, 결과 텍스트 수신.
   3. `prompting/parsers.py`가 `ANSWER: N` 또는 마지막 standalone 정수로 0-기반 라벨 추출.
   4. 실패 시 `predictor`의 폴백(라벨 `0`)으로 안전망.
5. `timing.json` 기록 후 원시 출력과 CSV 저장.
6. `scripts/make_submission.py`가 CSV/`submission.json` 직렬화.

---

## 상태 및 규칙 / Status & Competition Rules

### 대회 규칙 (코드 수준 강제, "규칙 #N")

| # | 규칙 | 강제 위치 |
| --- | --- | --- |
| 1 | 최종 라벨은 LLM 생성 텍스트에서 **파싱**해야 함. 규칙/다수결로 라벨을 만들면 실격. | `inference/predictor.py`, `prompting/parsers.py`, `inference/ensemble.py` |
| 2 | 외부 추론 API 사용 금지. OpenAI/Gemini/HF Inference 등 금지. | `models/loader.py` |
| 3 | 추론은 **완전 오프라인**으로 동작. 채점 시 인터넷 없음. | 모든 모듈 |
| 4 | 평균 **≈ 0.5초/샘플** 지연 예산. `num_prompt_variants`는 낮게 유지. | `configs/*.yaml`, `timing.json` |
| 5 | 사전학습 가중치는 **2026-05-31 이전 공개**된 것만 사용, 모델 ID와 라이선스 기록. | `models/*`, `artifacts/final/SCORES.md` |
| 6 | 데이터 누수 금지 (평가셋 분석하여 트레인/프롬프트 튜닝 금지). | `data/`, `prompting/` |

### 라벨 시맨틱 (주의)

- 제출 CSV 컬럼: `sample_id, label`.
- `label`은 **0-based 옵션 인덱스**. (예: 4지선다면 0..3)
- 옵션 번호 매핑과 폴백(파싱 실패 시 라벨 `0`)은 모두 `prompting/parsers.py`와 `inference/predictor.py`에 있음.
- "Unknown / 판단 불가"는 **프롬프트 전략**(`bias_guarded`)으로 다루며, 규칙 매핑이 아닙니다. `constants.UNKNOWN_OPTION_PATTERNS`는 로컬 분석 전용입니다.

### 산출물 스냅샷 (`artifacts/final/`)

| 모델/실험 | CSV | `submission.json` | `timing.json` | 비고 |
| --- | --- | --- | --- | --- |
| `qwen3vl_8b` | ✓ | ✓ | ✓ | 베이스라인 |
| `qwen3vl_8b_lowres` | ✓ | ✓ | ✓ | 저해상도 변형 |
| `qwen3vl_8b_hires` | ✓ | ✓ | ✓ | 고해상도 변형 |
| `qwen3vl_8b_bgv2` | ✓ | ✓ | ✓ | 편향 가드 v2 |
| `qwen3vl_8b_evidence_only` | (superseded) | ✓ | — | `artifacts/final/superseded/` |
| `qwen25vl_7b` | (superseded) | ✓ | — | `artifacts/final/superseded/` |
| `internvl3_8b` | ✓ | ✓ | ✓ | InternVL3 베이스라인 |

자세한 점수 비교는 `artifacts/final/SCORES.md` 참조.

---

## 먼저 읽을 파일 / First Files to Read

1. `src/skku_vqa/cli.py` — 콘솔 진입점.
2. `src/skku_vqa/config.py` — YAML 스키마와 딥 머지.
3. `src/skku_vqa/inference/predictor.py` — 한 샘플이 거치는 전체 흐름.
4. `src/skku_vqa/prompting/parsers.py` — 라벨 파싱 결정 (실격 위험 지점).
5. `src/skku_vqa/models/loader.py` — 백엔드 디스패치와 오프라인 정책.
6. `Makefile` + `scripts/*.py` — 운영자가 실제로 호출하는 명령.
7. `artifacts/final/SCORES.md` — 현재까지의 점수 기록.

---

## 설정 / Configuration

- 모든 실험은 `configs/*.yaml`에 정의. `defaults: configs/default.yaml`을 선언하면 베이스가 딥-머지된 뒤 오버라이드됩니다.
- 새 옵션은 `src/skku_vqa/config.py`의 Pydantic 모델에도 등록해야 YAML 검증을 통과합니다.
- 비밀값(`.env`)의 `OPENAI_API_KEY`/`HF_TOKEN`은 **개발 전용**이며 추론 시 사용되지 않습니다.
- 경로 기본값은 `configs/default.yaml`의 `data.raw`, `data.processed` 등 섹션.

```yaml
# configs/experiment/qwen3vl_8b.yaml (예시 구조)
defaults: configs/default.yaml
model:
  name: qwen3_vl_8b
  backend: vllm          # 또는 transformers (+ load_in_4bit: true)
inference:
  num_prompt_variants: 1
prompt:
  strategy: bias_guarded  # 또는 evidence_only
submission:
  out: outputs/submissions/submission.csv
```

---

## 빠른 시작 / Quickstart

> 추론 자체는 GPU가 있는 호스트에서만 실행됩니다. 이 저장소에서는 **편집/검증/단위 테스트**까지 가능합니다.

### 1) 설치
```bash
make setup          # = pip install -e ".[vlm,dev]"
```

### 2) 데이터 준비
```bash
make prepare        # python scripts/prepare_data.py --config configs/default.yaml
```

### 3) 단위 테스트 (GPU 불필요)
```bash
make test           # pytest -q
pytest tests/test_parsers.py -q        # 단일 파일
pytest tests/test_parsers.py::test_name -q
```

### 4) 추론 (GPU 호스트에서)
```bash
# 스모크: 처음 N개만
python scripts/run_inference.py --config configs/experiment/qwen3vl_8b.yaml --limit 16
# 전체
python scripts/run_inference.py --config configs/experiment/qwen3vl_8b.yaml
```

### 5) 제출 작성 + 검증
```bash
python scripts/make_submission.py --config configs/experiment/ensemble_v1.yaml
python scripts/validate_submission.py --submission outputs/submissions/submission.csv
```

### 6) 잠금 구성 재현
```bash
make reproduce      # python scripts/reproduce_submission.py --config artifacts/final/config.yaml
```

---

## 명령어 레퍼런스 / Commands Reference

| 명령 | 실제 수행 | 비고 |
| --- | --- | --- |
| `make setup` | `pip install -e ".[vlm,dev]"` | GPU 호스트에서 `vlm` 옵션까지 설치 |
| `make prepare` | `python scripts/prepare_data.py --config configs/default.yaml` | 전처리 |
| `make infer` | `python scripts/run_inference.py --config configs/experiment/zero_shot_qwen.yaml` | ⚠️ config 미존재 — 직접 호출 권장 |
| `make ensemble` | `python scripts/run_ensemble.py --config configs/experiment/ensemble_v1.yaml` | ⚠️ `run_ensemble.py` 존재 여부 확인 필요 |
| `make submit` | `python scripts/make_submission.py --config configs/experiment/ensemble_v1.yaml` | ⚠️ config 경로 확인 |
| `make validate` | `python scripts/validate_submission.py --submission outputs/submissions/submission.csv` | 제출 직전 점검 |
| `make reproduce` | `python scripts/reproduce_submission.py --config artifacts/final/config.yaml` | 잠금 구성 재현 |
| `make test` | `pytest -q` | GPU 불필요 |
| `make lint` | `ruff check src tests scripts` | line-length 100, py310, rules `E,F,I,W,UP,B` |
| 원격 헬퍼 | `bash scripts/remote/{setup, infer, sync}.sh` | 원격 GPU 호스트에서 실행 |

타입체크: `mypy`는 dev 의존성으로만 설치되어 있으며 Make 타깃이 없습니다. 필요 시 `mypy src`로 수동 실행.

---

## 진입점 / API & Entry Points

| 종류 | 위치 | 설명 |
| --- | --- | --- |
| 콘솔 스크립트 | `skku-vqa` (`pyproject [project.scripts]`) | `skku_vqa.cli:main` |
| 단일 모델 추론 | `python scripts/run_inference.py` | `inference/predictor.py` 호출 |
| 앙상블 | `inference/ensemble.py` | 텍스트 결합 → 단일 모델 라벨 결정 |
| 제출 직렬화 | `python scripts/make_submission.py` | `submission/dacon.py` 사용 |
| 검증 | `python scripts/validate_submission.py` | `data/validation.py` 사용 |
| 재현 | `python scripts/reproduce_submission.py` | `artifacts/final/config.yaml` 사용 |
| 자동 제출 | `scripts/auto_submit_next_window.sh` | systemd timer와 결합 |

데이터 스키마(`sample_id, label`), 옵션 인덱스 표현, 그리고 폴백(라벨 `0`)은 `data/schema.py`, `prompting/parsers.py`, `inference/predictor.py`에 정의되어 있습니다.

---

## 테스트 및 로컬 개발 / Testing & Local Dev

- 테스트 실행: `pytest -q` (`pyproject.toml`의 `pythonpath=["src"]` 덕분에 설치 없이도 `skku_vqa` 임포트 가능).
- 단일 테스트: `pytest tests/<file>.py -q` 또는 `pytest tests/<file>.py::<name> -q`.
- 린트: `ruff check src tests scripts` (자동 포매터 단계 없음).
- 스크립트는 실행 시 `sys.path`에 `src/`를 직접 주입합니다.
- **GPU가 없는 개발 호스트**: `scripts/run_inference.py`는 실행하지 마세요. 산출물 기반 검증/디버깅만 수행.

원격 개발 흐름 (개략):
1. 로컬에서 코드/설정/테스트 작업.
2. `bash scripts/remote/sync.sh`로 원격 호스트에 동기화.
3. 원격 호스트에서 `bash scripts/remote/infer.sh` 등으로 추론.
4. 산출물을 다시 받아와 `make validate` 실행.

> 스크립트/서비스 안의 호스트 식별자는 비공개 정보로 간주하고, **호스트명이나 IP를 하드코딩하지 마세요.** 필요 시 환경변수로 주입.

---

## 배포 / Deployment (systemd)

`deploy/systemd/` 디렉터리에 다음 두 단위 파일이 있습니다.

| 파일 | 역할 |
| --- | --- |
| `dacon-auto-submit.service` | 단회성 업로드 작업. `scripts/auto_submit_next_window.sh` 호출. |
| `dacon-auto-submit.timer` | 제출 윈도우에 맞춰 service를 트리거. |

세부 설치 단계는 `deploy/systemd/README.md`를 참고하세요.

---

## 기여 / Contributing

- `CONTRIBUTING.md` 참조 (PR 절차, 코딩 스타일, 대회 규칙 준수 체크리스트).
- 새 옵션을 추가할 때: `src/skku_vqa/config.py`의 Pydantic 모델 → `configs/default.yaml` → 실험 yaml 순으로 동기화.
- 대회 규칙을 우회하는 변경(예: 외부 API 라우팅, 다수결 라벨 결정)은 절대 머지하지 마세요.
- 가능하면 `make test` / `make lint`를 PR 전 로컬에서 통과시키세요.

---

## 유지보수자 / Maintainers

- 본 저장소: SKKU Multimodal AI Challenge 2026 팀.
- 도움말: 이슈 트래커 또는 리포지토리 내 연락처 참조.

---

## 참고 문서 / Further Reading

- `AGENTS.md` — 에이전트/운영자를 위한 인스트럭션(규칙, 라벨 시맨틱, 함수 위치).
- `CONTRIBUTING.md` — 기여 절차.
- `artifacts/final/SCORES.md` — 실험 결과/점수 요약.
- `deploy/systemd/README.md` — systemd 유닛 설치 가이드.
- `pyproject.toml` — 의존성, 스크립트, 린트/테스트 설정.

---

## 라이선스 / License

이 프로젝트는 `LICENSE`(MIT 가정) 하에 배포됩니다. 사용한 외부 모델 가중치는 각 모델의 라이선스(예: Qwen, InternVL, LLaVA-OV)를 따르며, 본 저장소 `artifacts/final/SCORES.md`에 모델 ID와 함께 기록되어 있습니다.