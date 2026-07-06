# SKKU-VQA — 2026 SKKU 멀티모달 AI 챌린지 (DACON #236722)

[![Python](https://img.shields.io/badge/python-≥3.10-3776AB.svg)](#로컬-개발)
[![License](https://img.shields.io/badge/license-저장소_LICENSE-green.svg)](LICENSE)
[![Competition](https://img.shields.io/badge/DACON-236722-orange.svg)](https://dacon.io/competitions/official/236722/overview/description)
[![Status](https://img.shields.io/badge/status-competition_active-yellow.svg)](#status)

> 이미지 + 질문 + 선택지로 답하는 **멀티모달 편향 인식 VQA** 파이프라인. BBQ 스타일 벤치마크에서 **Balanced Accuracy**를 최적화하고, 근거가 부족한 샘플은 "알 수 없음" 선택지로 매핑해 ambiguous 그룹의 정확도까지 끌어올린다.

## 한눈에 보기

| 항목 | 값 |
| --- | --- |
| 대회 | DACON #236722 (2026 SKKU 멀티모달 AI 챌린지) |
| 평가지표 | Balanced Accuracy (ambiguous + disambiguated 그룹 평균) |
| 입력 | 이미지, 질문, 선택지 (`0..N-1`) |
| 출력 | `sample_id,label` CSV (`label`은 0-based 옵션 인덱스) |
| 백엔드 모델 | Qwen2.5-VL-7B, Qwen3-VL-8B, InternVL3-8B, LLaVA-OV |
| 추론 런타임 | vLLM (기본) 또는 transformers + 4bit (옵션) |
| 데이터 정책 | 평가 데이터만 제공, 학습 데이터는 참가자가 직접 구성 |
| 실행 모드 | 오프라인 추론 필수 (스코어링 시 인터넷 사용 불가) |
| 배포 | systemd `dacon-auto-submit.timer` 주기 제출 |
| 추론 환경 | 16GB GPU에서 4bit 기준 ~8–9B VLM 상한, 48GB 환경은 2단계 |

## 파이프라인 한 줄 요약

`make prepare` → (GPU 호스트에서 `scripts/run_inference.py`) → `scripts/make_submission.py` → `scripts/validate_submission.py` → `submit_dacon.py` 또는 `deploy/systemd/dacon-auto-submit.timer` 자동 제출. 라벨은 **LLM 텍스트 파싱으로만** 결정하며, direct / evidence-first / bias-guarded 다중 프롬프트 앙상블을 지원한다.

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [핵심 전략](#핵심-전략)
3. [저장소 구조](#저장소-구조)
4. [아키텍처](#아키텍처)
5. [빠른 시작](#빠른-시작)
6. [명령어 레퍼런스](#명령어-레퍼런스)
7. [설정 시스템](#설정-시스템)
8. [대회 규칙 (코드 전반에 인코딩됨)](#대회-규칙-코드-전반에-인코딩됨)
9. [라벨 의미 (주의)](#라벨-의미-주의)
10. [제출 및 배포](#제출-및-배포)
11. [로컬 개발](#로컬-개발)
12. [테스트](#테스트)
13. [유지보수 / 문의](#유지보수--문의)
14. [추가 문서](#추가-문서)
15. [라이선스](#라이선스)

## 프로젝트 개요

본 저장소는 **2026 SKKU 멀티모달 AI 챌린지 (DACON #236722)** 참가 솔루션이다. 평가 데이터만으로 구성된 BBQ 스타일 멀티모달 편향 벤치마크에서, 강력한 오픈웨이트 VLM을 zero-shot으로 운용하고 **편향(bias)** 과 **불확실성(ambiguity)** 을 동시에 처리하는 것을 목표로 한다.

- **대상 사용자**: 챌린지 참가 팀, 멀티모달 편향/안전 연구자, VLM 평가 파이프라인을 재현하려는 실무자.
- **왜 유용한가**: 공개 BBQ 벤치마크에 대한 재현 가능한 추론 파이프라인, 다중 프롬프트 앙상블, 불확실성 매핑, 오프라인 추론을 위한 가중치 캐싱 전략을 한 번에 제공한다.
- **할 수 있는 일**:
  - 평가 데이터에 대해 단일 모델 또는 앙상블 추론 실행.
  - 제출 형식 검증을 거친 CSV를 DACON 포맷으로 생성.
  - 잠긴(final) 설정을 그대로 재현 (`make reproduce`).
  - 로컬 분석용 모호성 점수 및 "알 수 없음" 분포 산출.

## Status

- **개발 상태**: 대회 진행 중 (active). 본 저장소는 리더보드 제출에 사용된 산출물(`artifacts/final/`)을 함께 보존한다.
- **프로덕션 준비도**: 챌린지 제출용 검증 완료. 일반 제품용이 아닌 **경진 대회 엔트리**이므로 모델/프롬프트는 해당 과제에 특화되어 있다.
- **유지보수**: `AGENTS.md`의 명령·규칙·라벨 의미 참조. 운영 중 변경이 잦은 영역은 설정/YAML과 `artifacts/final/` 잠금 파일이다.

## 핵심 전략

1. **Zero-shot VLM 우선** — Qwen2.5-VL, InternVL3, Qwen3-VL 등 강력한 오픈웨이트 VLM을 프롬프트만으로 운용.
2. **불확실성 탐지** — 근거 부족 시 "알 수 없음" 선택지로 강제 매핑해 ambiguous 그룹 정확도를 확보.
3. **프롬프트 앙상블** — direct / evidence-first / bias-guarded 프롬프트의 다수결.
4. **합성 데이터 + LoRA (선택)** — zero-shot이 정체될 때만 2차로 도입. 어댑터는 `artifacts/lora_adapters/`에 보관.
5. **오프라인 + 재현 가능** — 모든 가중치 사전 캐시, 최종 제출은 잠긴 설정으로 재현.

## 저장소 구조

| 경로 | 역할 |
| --- | --- |
| `src/skku_vqa/` | 핵심 라이브러리 (CLI, 모델, 추론, 평가, 데이터, 프롬프트, 제출, 모호성) |
| `src/skku_vqa/models/` | VLM 어댑터 (`internvl3`, `llava_ov`, `llava_ov_transformers`, `qwen3_vl`, `qwen_vl`) |
| `src/skku_vqa/inference/` | 단일 모델 `predictor` + 다중 모델 `ensemble` |
| `src/skku_vqa/evaluation/` | Balanced Accuracy 등 메트릭 |
| `src/skku_vqa/prompting/` | 프롬프트 빌더/템플릿, 0-based 정답 파서 |
| `src/skku_vqa/data/` | 데이터셋, 스키마, 검증 |
| `src/skku_vqa/ambiguity/` | 모호성 탐지 + "알 수 없음" 매퍼 |
| `src/skku_vqa/submission/` | DACON 제출 어댑터 |
| `scripts/` | 실행 스크립트 (prepare/infer/ensemble/submit/validate/reproduce) |
| `scripts/remote/` | 원격 GPU 호스트 셋업 / 추론 / 동기화 |
| `configs/` | Pydantic-검증 YAML (`default` + `experiment/*`) |
| `artifacts/final/` | 리더보드 제출에 사용된 잠금 산출물 (config, csv, submission.json, timing.json) |
| `artifacts/lora_adapters/` | LoRA 어댑터 저장 위치 (현재 비어 있음, 필요 시 채움) |
| `deploy/systemd/` | `dacon-auto-submit.service` / `.timer` |
| `pyproject.toml`, `Makefile` | 의존성, 작업 정의 |
| `LICENSE`, `CONTRIBUTING.md`, `AGENTS.md` | 라이선스, 기여 가이드, 에이전트 운영 노트 |

## 아키텍처

### 모듈 의존성

| 계층 | 모듈 | 책임 |
| --- | --- | --- |
| 진입점 | `cli.py` | `skku-vqa` 콘솔 커맨드 (`project.scripts`) |
| 설정 | `config.py`, `constants.py` | YAML 로딩 + Pydantic 검증, 상수 |
| 데이터 | `data/dataset.py`, `data/schema.py`, `data/validation.py` | 로딩, 0-based 스키마, 형식 검증 |
| 프롬프트 | `prompting/builders.py`, `prompting/templates.py`, `prompting/parsers.py` | 프롬프트 생성, 템플릿, **라벨 파싱** |
| 모델 | `models/loader.py`, `models/base.py`, 각 모델 어댑터 | 가중치 로딩 + prefix 디스패치 |
| 추론 | `inference/predictor.py`, `inference/ensemble.py` | 단일/앙상블 추론, 타이밍 기록 |
| 평가 | `evaluation/metrics.py` | Balanced Accuracy 등 |
| 모호성 | `ambiguity/detector.py`, `ambiguity/unknown_mapper.py` | 모호성 신호 + 알 수 없음 매핑 |
| 제출 | `submission/dacon.py` | DACON 포맷 변환 |
| 유틸 | `utils/io.py`, `utils/seed.py` | I/O, 시드 고정 |

### 추론 흐름

1. `scripts/prepare_data.py`가 DACON 원본을 로딩하고 스키마를 검증.
2. `scripts/run_inference.py`가 설정의 모델 prefix로 `models/loader.py`를 디스패치 → vLLM 또는 transformers + 4bit 백엔드 선택.
3. `prompting/builders.py`가 옵션을 `0..N-1`로 번호 매긴 프롬프트를 생성하고 모델이 자유 텍스트로 답변.
4. `prompting/parsers.py`가 `ANSWER: N` 패턴을 우선 파싱, 실패 시 마지막 독립 정수로 폴백, 최종 폴백은 빈 행 방지를 위한 label `0`.
5. `inference/ensemble.py`가 다중 프롬프트 / 다중 모델 결과를 집계하되 **최종 라벨은 LLM 텍스트에서 파싱**.
6. `ambiguity/` 모듈이 근거 부족 신호에 대해 "알 수 없음" 선택지로 강제 매핑 (프롬프트 전략 기반).
7. `submission/dacon.py`가 `sample_id,label` CSV를 생성하고 `scripts/validate_submission.py`가 형식과 라벨 범위를 검증.
8. `submit_dacon.py` 또는 `deploy/systemd/dacon-auto-submit.timer`가 자동 업로드.

## 빠른 시작

```bash
# 1) 설치 (개발 + VLM 옵션 + lint/test 포함)
make setup

# 2) 데이터 전처리 (DACON 원본을 data/raw/ 아래에 둔 뒤)
python scripts/prepare_data.py --config configs/default.yaml

# 3) 추론 (GPU 호스트에서 실행)
python scripts/run_inference.py --config configs/experiment/qwen3vl_8b.yaml --limit 8   # 스모크
python scripts/run_inference.py --config configs/experiment/qwen3vl_8b.yaml            # 전체

# 4) 제출 파일 생성 + 검증
python scripts/make_submission.py --config configs/experiment/ensemble_v1.yaml
python scripts/validate_submission.py --submission outputs/submissions/submission.csv

# 5) 최종 잠금 설정 재현
python scripts/reproduce_submission.py --config artifacts/final/config.yaml
```

> ⚠️ **Makefile 경로 주의**: `make infer` / `make ensemble` / `make submit` 가 가리키는 YAML 경로(`configs/experiment/zero_shot_qwen.yaml`, `configs/experiment/ensemble_v1.yaml`)가 현재 저장소에 존재하지 않는다. 위처럼 스크립트를 직접 + 실제 존재하는 `configs/experiment/*.yaml`을 지정해서 실행한다.

## 명령어 레퍼런스

| 명령 | 설명 |
| --- | --- |
| `make setup` | `pip install -e ".[vlm,dev]"` (transformers, vLLM 옵션, dev 도구 일괄 설치) |
| `make prepare` | `scripts/prepare_data.py`로 데이터 준비/검증 |
| `make infer` | ⚠️ 경로 부재. `python scripts/run_inference.py --config configs/experiment/<name>.yaml` 직접 호출 |
| `make ensemble` | ⚠️ 경로 부재. 앙상블 설정으로 `run_inference.py` 직접 호출 |
| `make submit` | ⚠️ 경로 부재. `python scripts/make_submission.py --config ...` 직접 호출 |
| `make validate` | `scripts/validate_submission.py`로 제출 CSV 형식/라벨 검증 |
| `make reproduce` | `artifacts/final/config.yaml`로 최종 잠금 재현 |
| `make test` | `pytest -q` (`tests/`, `pythonpath=["src"]`) |
| `make lint` | `ruff check src tests scripts` (line-length 100, py310, rules `E,F,I,W,UP,B`) |
| `mypy src` | 타입체크 (Make 타깃 없음, dev 의존성으로만 설치됨) |
| `skku-vqa` | 콘솔 진입점 (`skku_vqa.cli:main`) |

## 설정 시스템

- 모든 실험은 `configs/*.yaml`로 표현되며 `src/skku_vqa/config.py`의 Pydantic 모델로 로드 + 검증된다.
- **상속**: 설정에 `defaults: configs/default.yaml`을 두면 베이스를 deep-merge 후 오버라이드. 새 키를 추가할 때는 YAML만이 아니라 Pydantic 모델에도 함께 반영한다.
- `configs/default.yaml`은 경로 기본값, 모델/프롬프트/추론/제출 섹션을 보유.
- **모델 디스패치** (`models/loader.py`): `llava*`, `internvl*`, `qwen3_vl*`, `qwen*` 접두어로 분기. `backend=transformers`만 `load_in_4bit` 지원, vLLM 경로는 미지원.
- **추론 옵션**: `inference.num_prompt_variants` (앙상블용 프롬프트 수) — 샘플당 0.5초 예산과 직결되므로 낮게 유지하고 `outputs/predictions/<exp>/timing.json`으로 확인한다.

## 대회 규칙 (코드 전반에 인코딩됨)

다음은 코드 전반에 "규칙 #N"으로 인코딩된 **하드 제약**이다. 위반 시 제출이 실격 처리된다.

| # | 규칙 | 코드 위치 |
| --- | --- | --- |
| 1 | 최종 라벨은 **LLM 생성 텍스트에서 파싱**. 규칙/다수결/조건부 오버라이드 금지 (앙상블도 LLM 결과를 집계) | `inference/predictor.py`, `prompting/parsers.py` |
| 2 | **외부 추론 API 금지** (OpenAI / Gemini / HF Inference 등). 로컬 가중치만 사용 | `models/loader.py` (enforce). `.env`의 `OPENAI_API_KEY` / `HF_TOKEN`은 dev 전용 |
| 3 | **완전 오프라인 실행** (스코어링 시 인터넷 없음). 가중치 사전 다운로드/캐시 필수 | README 운영 노트 |
| 4 | **지연 예산 ≈ 샘플당 0.5초 평균**. `num_prompt_variants` 낮게 유지, `outputs/predictions/<exp>/timing.json` 확인 | `inference/` |
| 5 | 사전훈련 가중치는 **2026-05-31 이전 공개** 되어야 함. 모델 ID + 라이선스 기록 | `models/` 어댑터 |
| 6 | **데이터 누수 금지** — 평가 세트 분석으로 유사 학습 데이터/프롬프트 제작 금지 | 파이프라인 전반 |

## 라벨 의미 (주의)

- 제출 CSV 컬럼: `sample_id,label`. **`label`은 0-based 옵션 인덱스** (`prompting/parsers.py`, `data/schema.py`).
- 프롬프트는 옵션을 `0..N-1`로 번호 매기고, 파서는 `ANSWER: N` 우선 → 실패 시 마지막 독립 정수 → 최종 폴백은 빈 행 방지를 위한 label `0`.
- "알 수 없음 / 판단 불가" 처리는 **규칙 매핑이 아니라 `bias_guarded` 프롬프트 전략**으로 수행. `constants.py`의 `UNKNOWN_OPTION_PATTERNS`는 로컬 분석 전용.
- README/문서의 "옵션 1" 같은 표현은 예시일 뿐이며, 실제 코드는 0-based다. 스키마/제출 검증도 0-based 범위만 통과시킨다.

## 제출 및 배포

- **수동 제출**: `scripts/submit_dacon.py` (DACON 자격 증명 필요).
- **자동 제출 (systemd)**: `deploy/systemd/dacon-auto-submit.service` + `.timer`. 자세한 운영은 `deploy/systemd/README.md` 참조.
- **자동 제출 (셸 래퍼)**: `scripts/auto_submit_next_window.sh`가 다음 제출 윈도우를 계산해 자동 업로드.
- **원격 추론**: `scripts/remote/setup.sh` (호스트 초기화), `infer.sh` (원격 실행), `sync.sh` (결과 동기화). 추론 호스트는 별도 GPU 머신(`<inference-host>`)이며 사설 IP/호스트명은 환경에 맞게 설정한다.
- **잠금 산출물**: `artifacts/final/<exp>.{config.resolved.json,csv,submission.json,timing.json}`가 같은 실험의 재현 단위다.

## 로컬 개발

- **Python**: 3.10 이상 (3.10/3.11 권장). GPU 추론은 별도 CUDA 호스트에서 수행.
- **테스트는 dev 박스에서 가능**, 추론은 GPU 호스트에서 실행. GPU/가중치가 없는 라이트 박스에서는 `run_inference.py`를 실제 모델 대상으로 돌리지 않는다.
- **에디팅 흐름**: 코드 수정 → `make lint` → `make test` → GPU 호스트에 `scripts/remote/sync.sh`로 반영 → 원격 `infer.sh`.
- **시드**: `utils/seed.py`로 재현성 확보.
- **GPU 한도**: 16GB 박스 + 4bit 양자화 기준 **8–9B VLM이 상한** (Qwen3-VL-8B, InternVL3-8B). 그 이상은 48GB(A6000) 2단계 환경이 필요하다.

## 테스트

```bash
make test                                       # 전체
pytest -q tests/test_parsers.py                 # 단일 파일
pytest -q tests/test_parsers.py::test_extract_label   # 단일 케이스
```

`pytest`는 `pythonpath=["src"]`로 동작하므로 `pip install -e .` 없이도 `import skku_vqa`가 가능하다. 런타임 스크립트는 자체적으로 `sys.path`에 `src/`를 주입한다.

## 유지보수 / 문의

- **저장소 운영 노트**: [`AGENTS.md`](AGENTS.md) — 명령, 규칙, 라벨 의미, 모델 디스패치 등 빠른 참조.
- **기여 가이드**: [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **챌린지 원문**: [DACON #236722](https://dacon.io/competitions/official/236722/overview/description).
- **이슈/PR**: 저장소 이슈 트래커 사용.

## 추가 문서

- [`artifacts/final/SCORES.md`](artifacts/final/SCORES.md) — 리더보드 점수, 사용된 잠금 설정, 비교 표.
- `artifacts/final/<exp>.config.resolved.json` — 실행 시점에 해소된 설정 스냅샷.
- `artifacts/final/<exp>.timing.json` — 샘플당 지연 (0.5초 예산 검증용).
- `artifacts/final/<exp>.submission.json` — DACON 제출 직전 내부 표현.
- [`deploy/systemd/README.md`](deploy/systemd/README.md) — systemd 기반 자동 제출 운영.
- `scripts/` 각 스크립트 docstring — 단계별 옵션 설명.
- (해당 시) `docs/` 디렉터리의 대회 노트 / 재현성 문서.

## 라이선스

본 저장소는 [`LICENSE`](LICENSE) 파일을 따른다. 사용된 외부 VLM 가중치는 각 모델의 원 라이선스를 따르며, 챌린지 규칙에 따라 **2026-05-31 이전 공개 모델만 허용**된다.