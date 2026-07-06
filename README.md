# skku-vqa

2026 SKKU Multimodal AI Challenge · DACON #236722

![Python](https://img.shields.io/badge/python-≥3.10-blue)
![DACON](https://img.shields.io/badge/DACON-236722-blueviolet)
![Status](https://img.shields.io/badge/status-active--development-green)
![Inference](https://img.shields.io/badge/inference-offline--only-orange)
![License](https://img.shields.io/badge/license-project--internal-lightgrey)

> **한 줄 요약**: 다지선다 VQA용 경량 VLM 파이프라인.
> 로컬 가중치로 오프라인 추론하며, 편향 가드 프롬프트와 LLM 파싱으로
> 최종 라벨을 결정합니다 (규칙 기반 덮어쓰기 없음).

## 한눈에 보기 / At a glance

| 항목 | 값 |
|---|---|
| 무엇이 실행되나 | `prepare → infer → ensemble → submit → validate` 파이프라인 |
| 누가 운영하나 | 챌린지 팀 (저장소 메인테이너) |
| 다음에 누를 명령 | `python scripts/run_inference.py --config configs/experiment/<exp>.yaml` |
| 지원 VLM | InternVL3, LLaVA-OneVision, Qwen3-VL, Qwen-VL |
| 백엔드 | vLLM (기본) / Transformers (4bit 가능) |
| 라벨 규약 | 0-based 정수, CSV `sample_id,label` |
| GPU | 필수 (16GB+ 권장, 4bit 8B 모델) |
| 네트워크 | 추론 시 완전 오프라인 필수 |
| 레이턴시 예산 | 평균 ~0.5초/샘플 |

## 파이프라인 흐름 / Pipeline flow

1. **Prepare** — `scripts/prepare_data.py`로 입력을 정규화합니다.
2. **Infer** — `scripts/run_inference.py`가 VLM 디스패처로 단일 모델 추론을 수행합니다.
3. **Ensemble** — 다중 변형·해상도 결과를 `skku_vqa.inference.ensemble`로 통합합니다.
4. **Submit** — `scripts/make_submission.py`가 DACON 형식 CSV를 직렬화합니다.
5. **Validate** — `scripts/validate_submission.py`로 형식 검증을 합니다.

> ⚠️ **Makefile 주의**: `make infer` / `make ensemble` / `make submit`은
> 존재하지 않는 예시 설정 경로를 가리킵니다. 운영 시
> `python scripts/run_inference.py --config configs/experiment/<file>.yaml`로
> 명시적으로 넘겨주세요. 자세한 내용은 [`AGENTS.md`](AGENTS.md).

## 목차 / Contents

- [Purpose / 목적](#purpose--목적)
- [Status / 상태](#status--상태)
- [Package contents / 패키지 구성](#package-contents--패키지-구성)
- [First files to read / 먼저 읽을 파일](#first-files-to-read--먼저-읽을-파일)
- [API or entry points / API 및 진입점](#api-or-entry-points--api-및-진입점)
- [Quickstart / 빠른 시작](#quickstart--빠른-시작)
- [Configuration / 구성](#configuration--구성)
- [Commands / 명령어](#commands--명령어)
- [Local development / 로컬 개발](#local-development--로컬-개발)
- [Hard rules / 대회 규칙 요약](#hard-rules--대회-규칙-요약)
- [Label semantics / 라벨 규약](#label-semantics--라벨-규약)
- [Supported VLMs / 지원 모델](#supported-vlms--지원-모델)
- [Maintainers / 책임자](#maintainers--책임자)
- [Further documentation / 참고 문서](#further-documentation--참고-문서)
- [License](#license)

## Purpose / 목적

`skku-vqa`는 2026 SKKU 멀티모달 AI 챌린지 (DACON #236722)용
경량 VQA 파이프라인입니다. 사용자가 할 수 있는 일은 다음과 같습니다.

- 다지선다 VLM 추론을 단일 스크립트로 실행.
- 여러 VLM과 해상도 변형을 조합해 앙상블.
- `bias_guarded` 프롬프트 전략으로 모호한 응답을 LLM 내부에서 처리.
- DACON 제출 형식 (`sample_id,label`) 직렬화 및 검증.
- systemd 타이머로 다음 제출 윈도우 자동화.

오프라인 추론·로컬 가중치·낮은 레이턴시라는 대회 제약을 코드 레벨에서 강제합니다.

## Status / 상태

| 구성 | 상태 | 비고 |
|---|---|---|
| 패키지 빌드 | ✅ | `pip install -e ".[vlm,dev]"` |
| 단위 테스트 | ✅ | `make test` (`pytest -q`) |
| 린트 | ✅ | `make lint` (`ruff`) |
| Qwen3-VL-8B 4bit 추론 | ✅ | 16GB GPU 검증 |
| InternVL3-8B 4bit 추론 | ✅ | 16GB GPU 검증 |
| >9B 풀 프리시전 | ⚠️ | 48GB A6000 환경 필요 |
| 앙상블 v1 | ✅ | 다중 변형 통합 |
| DACON 제출 직렬화 | ✅ | `submission/dacon.py` |
| 자동 제출 (systemd) | ✅ | `deploy/systemd/` |
| 외부 API 호출 | ❌ | 규칙 위반 — 의도적으로 차단 |

점수·레이턴시 상세: [`artifacts/final/SCORES.md`](artifacts/final/SCORES.md).

## Package contents / 패키지 구성

| 경로 | 역할 |
|---|---|
| `src/skku_vqa/cli.py` | `skku-vqa` 콘솔 엔트리 |
| `src/skku_vqa/config.py` | YAML + Pydantic 설정 로더 |
| `src/skku_vqa/constants.py` | 라벨·unknown 패턴 상수 |
| `src/skku_vqa/data/` | 데이터셋·스키마·검증 |
| `src/skku_vqa/models/` | VLM 어댑터 (InternVL3, LLaVA-OV, Qwen3-VL, Qwen-VL) |
| `src/skku_vqa/inference/` | `predictor.py`, `ensemble.py` |
| `src/skku_vqa/evaluation/` | 메트릭 |
| `src/skku_vqa/prompting/` | 프롬프트 빌더·파서·템플릿 |
| `src/skku_vqa/submission/dacon.py` | DACON 제출 직렬화 |
| `src/skku_vqa/ambiguity/` | 모호성 감지 (분석 전용) |
| `src/skku_vqa/utils/` | IO·시드 |
| `scripts/` | 운영 스크립트 (`prepare`, `infer`, `make_submission`, ...) |
| `scripts/remote/` | 원격 GPU 호스트 셋업·실행·동기화 |
| `deploy/systemd/` | 자동 제출 서비스·타이머 |
| `artifacts/final/` | 잠금된 최종 설정 + 결과물 |
| `artifacts/lora_adapters/` | (선택) LoRA 어댑터 |
| `configs/` | 실험 설정 (`default.yaml` + 실험별 YAML) |

## First files to read / 먼저 읽을 파일

| 순서 | 파일 | 이유 |
|---|---|---|
| 1 | [`AGENTS.md`](AGENTS.md) | 대회 규칙·라벨 규약·운영 제약 |
| 2 | [`Makefile`](Makefile) | 파이프라인 명령 |
| 3 | [`src/skku_vqa/cli.py`](src/skku_vqa/cli.py) | CLI 진입점 |
| 4 | [`src/skku_vqa/config.py`](src/skku_vqa/config.py) | 설정 스키마 |
| 5 | [`src/skku_vqa/prompting/parsers.py`](src/skku_vqa/prompting/parsers.py) | 라벨 파싱 (0-based) |
| 6 | [`src/skku_vqa/models/loader.py`](src/skku_vqa/models/loader.py) | 모델 디스패치·오프라인 강제 |
| 7 | [`artifacts/final/SCORES.md`](artifacts/final/SCORES.md) | 현재 점수표 |

## API or entry points / API 및 진입점

### 콘솔 스크립트

| 이름 | 호출 |
|---|---|
| `skku-vqa` | 설치 후 `skku-vqa ...` 또는 `python -m skku_vqa` |

### Python API

| 심볼 | 위치 | 설명 |
|---|---|---|
| `skku_vqa.cli.main` | [`cli.py`](src/skku_vqa/cli.py) | CLI 엔트리 |
| `skku_vqa.config.load_config` | [`config.py`](src/skku_vqa/config.py) | YAML + Pydantic 로더 |
| `skku_vqa.models.loader.load_model` | [`loader.py`](src/skku_vqa/models/loader.py) | VLM 디스패처 |
| `skku_vqa.inference.predictor.Predictor` | [`predictor.py`](src/skku_vqa/inference/predictor.py) | 단일 모델 추론기 |
| `skku_vqa.inference.ensemble.run_ensemble` | [`ensemble.py`](src/skku_vqa/inference/ensemble.py) | 다중 모델 결과 통합 |
| `skku_vqa.prompting.parsers.parse_label` | [`parsers.py`](src/skku_vqa/prompting/parsers.py) | 라벨 추출 (0-based) |
| `skku_vqa.submission.dacon.write_submission` | [`submission/dacon.py`](src/skku_vqa/submission/dacon.py) | 제출 CSV 직렬화 |
| `skku_vqa.evaluation.metrics.compute_metrics` | [`metrics.py`](src/skku_vqa/evaluation/metrics.py) | 메트릭 계산 |

## Quickstart / 빠른 시작

### 1. 설치

```bash
make setup
# 또는: pip install -e ".[vlm,dev]"
```

### 2. 설정 선택

`configs/experiment/`에서 사용할 실험 설정을 고릅니다.
상속이 필요하면 `defaults: configs/default.yaml`을 명시하세요.

### 3. 스모크 테스트

```bash
python scripts/run_inference.py \
    --config configs/experiment/<exp>.yaml \
    --limit 50
```

### 4. 앙상블과 제출

앙상블은 `skku_vqa.inference.ensemble` 모듈을 사용합니다.
자체 통합 스크립트나 Python REPL에서 호출하세요.

```python
from skku_vqa.inference.ensemble import run_ensemble
```

제출 파일 생성 및 검증:

```bash
python scripts/make_submission.py --config configs/experiment/<exp>.yaml
python scripts/validate_submission.py --submission outputs/submissions/submission.csv
```

### 5. 최종 재현

```bash
make reproduce
```

## Configuration / 구성

| 그룹 | 위치 | 설명 |
|---|---|---|
| `data.*` | `configs/default.yaml` | 입력 CSV/이미지 경로 |
| `model.*` | 실험 설정 | 모델 이름 prefix + 백엔드 |
| `prompting.*` | 실험 설정 | 템플릿 ID, 옵션 수, 변형 수 |
| `inference.*` | 실험 설정 | `num_prompt_variants`, 디코딩 파라미터 |
| `submission.*` | 실험 설정 | 출력 경로, 컬럼 매핑 |

상속: 하위 설정에 `defaults: configs/default.yaml`이 있으면
로더가 deep-merge 후 오버라이드합니다.

## Commands / 명령어

### Make 타깃

| 명령 | 동작 |
|---|---|
| `make setup` | editable 설치 (`.[vlm,dev]`) |
| `make prepare` | `scripts/prepare_data.py` |
| `make infer` | `scripts/run_inference.py` (예시 설정 — 실제 경로 확인) |
| `make ensemble` | ⚠️ 예시 설정 + 스크립트 미존재 — Python API 사용 |
| `make submit` | `scripts/make_submission.py` (예시 설정 — 실제 경로 확인) |
| `make validate` | `scripts/validate_submission.py` |
| `make reproduce` | `scripts/reproduce_submission.py` (잠금 설정) |
| `make test` | `pytest -q` |
| `make lint` | `ruff check src tests