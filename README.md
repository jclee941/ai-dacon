# skku-vqa — 2026 SKKU 멀티모달 AI Challenge

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License](https://img.shields.io/badge/license-see%20LICENSE-blue)](LICENSE)
[![DACON #236722](https://img.shields.io/badge/DACON-%23236722-orange)](https://dacon.io/competitions/official/236722/overview/description)

## 한국어 요약

`skku-vqa`는 DACON #236722(2026 SKKU 멀티모달 AI Challenge) 제출용 Python 파이프라인입니다. BBQ 스타일 멀티모달 편향 벤치마크에서 이미지·질문·선택지를 받아 **0-based 라벨**을 예측하며, 평가는 **Balanced Accuracy**(ambiguous + disambiguated 그룹 평균)입니다. 핵심 전략은 ① 강력한 오픈웨이트 VLM zero-shot 추론 ② "unknown" 불확실성 프롬프트 매핑 ③ direct / evidence-first / bias-guarded 프롬프트 앙상블 ④ zero-shot 정체 시 합성 데이터 + LoRA입니다. 추론은 GPU 워커에서만 수행하며, 본 dev 호스트에서는 편집·테스트·검증만 가능합니다.

> 참고: 본 README의 "선택지 1" 표기는 코드 기준 인덱스 0을 의미합니다. 라벨은 항상 0-based입니다.

### English summary

`skku-vqa` is the Python pipeline behind the 2026 SKKU Multimodal AI Challenge (DACON #236722) submission. Given an image, a question, and multiple-choice options, the system predicts a **0-based** answer label for a BBQ-style multimodal bias benchmark scored by **Balanced Accuracy**. It runs local open-weight VLMs (Qwen3-VL, Qwen2.5-VL, InternVL3, LLaVA-OV) with prompt strategies (direct / evidence-first / bias-guarded) and an LLM-mediated ensemble. External inference APIs, rule-based overrides, and online access at scoring time are forbidden. Inference requires a GPU; CPU-only hosts are limited to editing, testing, and submission validation.

---

## 한눈에 보기 (Status snapshot)

| 항목 | 상태 |
|------|------|
| 대회 / 평가지표 | DACON #236722, **Balanced Accuracy** (ambiguous + disambiguated 평균) |
| 라벨 인덱싱 | **0-based** 정수 (`0..N-1`), CSV 컬럼 `sample_id,label` |
| 추론 백엔드 | vLLM (기본), Transformers + bitsandbytes 4bit 옵션 |
| 지원 VLM | Qwen3-VL-8B, Qwen2.5-VL-7B, InternVL3-8B, LLaVA-OV |
| 학습 데이터 | 평가 데이터만 공개, 합성 + LoRA는 zero-shot 정체 시에만 |
| 외부 추론 API | **금지** (OpenAI / Gemini / HF Inference 등) |
| 인터넷 연결 | 추론 시 **완전 오프라인 필수** |
| 레이턴시 예산 | 평균 ≈ 0.5초/샘플 |
| Makefile 일부 타깃 | **stale**: `infer` / `ensemble` / `submit`이 존재하지 않는 config 참조 |
| GPU 미보유 dev 호스트 | 편집·테스트·검증만 가능, 추론은 별도 GPU 워커에서 |

## 실행 흐름 (Operator flow)

1. `make setup` — 의존성 설치 (`pip install -e ".[vlm,dev]"`).
2. `make prepare` — `scripts/prepare_data.py`로 raw / 외부 데이터 정합성 검증.
3. `python scripts/run_inference.py --config configs/experiment/<exp>.yaml [--limit N]` — 모델별 prediction CSV 생성 (`outputs/predictions/<exp>/`).
4. (선택) `python scripts/run_ensemble.py --config configs/experiment/<ensemble>.yaml` — LLM-매개 앙상블.
5. `python scripts/make_submission.py --config configs/experiment/<exp>.yaml` — DACON 제출 CSV 생성.
6. `python scripts/validate_submission.py --submission <csv>` — 라벨 / 스키마 검증.
7. `make reproduce` — 잠금된 `artifacts/final/config.yaml`로 최종 제출 재현.
8. (선택) `scripts/submit_dacon.py` + `deploy/systemd/*.timer` — 자동 재제출.

---

## 목차

- [패키지 구성 (Package contents)](#패키지-구성-package-contents)
- [상세 상태 및 주의사항 (Status & caveats)](#상세-상태-및-주의사항-status--caveats)
- [먼저 읽을 파일 (First files to read)](#먼저-읽을-파일-first-files-to-read)
- [아키텍처 (Architecture)](#아키텍처-architecture)
- [빠른 시작 (Quickstart)](#빠른-시작-quickstart)
- [설정 시스템 (Configuration)](#설정-시스템-configuration)
- [명령어 레퍼런스 (Commands reference)](#명령어-레퍼런스-commands-reference)
- [배포·자동제출 (Deployment & auto-submit)](#배포자동제출-deployment--auto-submit)
- [로컬 개발·테스트 (Local development & testing)](#로컬-개발테스트-local-development--testing)
- [운영·검증 (Operations & validation)](#운영검증-operations--validation)
- [기여 (Contributing)](#기여-contributing)
- [라이선스·연락처 (License & contact)](#라이선스연락처-license--contact)
- [참고 문서 (Further documentation)](#참고-문서-further-documentation)

---

## 패키지 구성 (Package contents)

| 경로 | 역할 |
|------|------|
| `src/skku_vqa/cli.py` | 콘솔 진입점 (`skku-vqa` 명령) |
| `src/skku_vqa/config.py` | Pydantic 기반 YAML 로더, `defaults:` deep-merge |
| `src/skku_vqa/constants.py` | 라벨 패턴, "unknown" 매핑(로컬 분석 전용) |
| `src/skku_vqa/models/` | VLM 어댑터 (`base`, `internvl3`, `qwen3_vl`, `qwen_vl`, `llava_ov`, `llava_ov_transformers`, `loader`) |
| `src/skku_vqa/inference/` | `predictor.py`, `ensemble.py` |
| `src/skku_vqa/evaluation/metrics.py` | Balanced Accuracy 등 지표 |
| `src/skku_vqa/utils/` | `io.py`, `seed.py` |
| `src/skku_vqa/data/` | `dataset.py`, `schema.py`, `validation.py` |
| `src/skku_vqa/prompting/` | `builders.py`, `parsers.py`, `templates.py` |
| `src/skku_vqa/submission/dacon.py` | DACON 제출 포맷 직렬화 |
| `src/skku_vqa/ambiguity/` | `detector.py`, `unknown_mapper.py` (로컬 분석 전용) |
| `scripts/` | 실행 스크립트 (prepare / infer / ensemble / submit / validate / reproduce) |
| `scripts/remote/` | 원격 GPU 호스트 셋업·실행 (`setup.sh`, `infer.sh`, `sync.sh`) |
| `configs/` | 기본 / 실험 설정 (YAML) |
| `artifacts/final/` | 리더보드 제출에 사용된 최종 산출물 |
| `artifacts/final/SCORES.md` | 리더보드 점수 및 잠금된 설정 요약 |
| `artifacts/lora_adapters/` | LoRA 어댑터 산출물 |
| `deploy/systemd/` | 자동 재제출 서비스 / 타이머 유닛 |
| `docs/` | 대회 노트, 재현성 문서 |
| `data/raw/` | DACON 원본 데이터 (read-only 가정) |
| `data/external/` | 직접 구성한 합성 / 캘리브레이션 데이터 |
| `outputs/` | 예측 / 제출 / 리포트 (gitignored) |

## 상세 상태 및 주의사항 (Status & caveats)

### 대회 하드 규칙 (코드 전반에 "규칙 #N"으로 인코딩, 위반 시 실격)

| # | 규칙 | 코드 위치 |
|---|------|----------|
| 1 | 최종 라벨은 LLM 생성 텍스트에서 파싱. 규칙 / 다수결 / 조건부 오버라이드 금지. 앙상블도 LLM이 답해야 함. | `predictor.py`, `prompting/parsers.py`, `inference/ensemble.py` |
| 2 | 외부 추론 API 금지 (OpenAI / Gemini / HF Inference 등). 로컬 가중치만 허용. | `models/loader.py` |
| 3 | 추론 시 완전 오프라인. 가중치 사전 다운로드 / 캐시. | — |
| 4 | 평균 레이턴시 ≈ 0.5초 / 샘플. `inference.num_prompt_variants` 낮게 유지. | `outputs/predictions/<exp>/timing.json` |
| 5 | 사전학습 가중치는 2026-05-31 이전 공개, model id + 라이선스 기록. | `configs/` 주석 |
| 6 | 데이터 누수 금지. 평가 데이터 분석으로 학습 / 프롬프트 튜닝 금지. | — |

### 라벨 의미 (Label semantics)

| 항목 | 값 |
|------|---|
| 제출 CSV 컬럼 | `sample_id,label` |
| 라벨 도메인 | 0-based 정수 (`0..N-1`) |
| 프롬프트 표기 | `ANSWER: N` 선호, 없으면 단독 숫자 마지막 매치 |
| 파싱 실패 fallback | 라벨 `0` (빈 행 방지 목적) |
| "unknown" 처리 | 프롬프트 전략(`bias_guarded`)으로 강제. 매핑 규칙 아님. |
| `UNKNOWN_OPTION_PATTERNS` | 로컬 분석 전용, 제출 결정에 사용 금지 |

### Stale Makefile 경고

다음 타깃은 존재하지 않는 config을 가리키므로 스크립트를 직접 호출하세요.

| 타깃 | 현재 참조 | 권장 명령 |
|------|----------|----------|
| `make infer` | `configs/experiment/zero_shot_qwen.yaml` (없음) | `python scripts/run_inference.py --config configs/experiment/<exp>.yaml [--limit N]` |
| `make ensemble` | `configs/experiment/ensemble_v1.yaml` (없음) | `python scripts/run_ensemble.py --config configs/experiment/<exp>.yaml` |
| `make submit` | `configs/experiment/ensemble_v1.yaml` (없음) | `python scripts/make_submission.py --config configs/experiment/<exp>.yaml` |

`make setup` / `make prepare` / `make validate` / `make reproduce` / `make test` / `make lint`은 정상 동작합니다.

### GPU 환경 제약

- 추론은 **GPU 필수**. CPU-only dev 호스트에서는 코드 편집 / 테스트 / 제출 검증만 가능.
- 검증된 상한 (16GB 추론 박스): Qwen3-VL-8B / InternVL3-8B 4bit.
- 그 이상 가중치는 48GB(A6000 등) 별도 워커 필요.

## 먼저 읽을 파일 (First files to read)

1. [`AGENTS.md`](AGENTS.md) — 대회 규칙, 명령어, 컨벤션.
2. [`pyproject.toml`](pyproject.toml) — 의존성, `skku-vqa` 콘솔 스크립트, ruff / pytest 설정.
3. [`Makefile`](Makefile) — 권장 진입 명령어와 stale 타깃 목록.
4. `src/skku_vqa/config.py` — YAML 스키마와 `defaults:` 머지 규칙.
5. `src/skku_vqa/models/loader.py` — 모델 디스패치와 외부 API 차단.
6. `src/skku_vqa/prompting/parsers.py` — 0-based 라벨 파싱 규칙.
7. `src/skku_vqa/inference/predictor.py` — 단일 모델 추론 루프.
8. `src/skku_vqa/inference/ensemble.py` — LLM-매개 앙상블.
9. [`artifacts/final/SCORES.md`](artifacts/final/SCORES.md) — 리더보드 점수 및 잠금된 설정.

## 아키텍처 (Architecture)

### 모듈 의존성 (요약)

| 계층 | 모듈 | 책임 |
|------|------|------|
| Entry | `cli.py`, `scripts/*.py` | 인자 파싱, 파이프라인 오케스트레이션 |
| Config | `config.py`, `constants.py` | YAML 로드·검증, 상수 |
| Data | `data/dataset.py`, `data/schema.py`, `data/validation.py` | 샘플 로드, 라벨 스키마, 데이터셋 정합성 |
| Prompt | `prompting/builders.py`, `prompting/templates.py`, `prompting/parsers.py` | 프롬프트 합성, 템플릿, 0-based 파싱 |
| Model | `models/base.py`, `models/loader.py`, `models/{qwen3_vl,qwen_vl,internvl3,llava_ov}*.py` | VLM 어댑터, 디스패치 |
| Inference | `inference/predictor.py`, `inference/ensemble.py` | 단일 / 앙상블 추론 루프 |
| Ambiguity | `ambiguity/detector.py`, `ambiguity/unknown_mapper.py` | 로컬 분석 (제출 결정 미사용) |
| Eval / Submission | `evaluation/metrics.py`, `submission/dacon.py` | Balanced Accuracy, DACON 직렬화 |
| Util | `utils/io.py`, `utils/seed.py` | I/O, 시드 고정 |

### 단일 모델 추론 흐름

1. `scripts/run_inference.py`가 `--config`로 YAML 로드 → `config.py`가 Pydantic 검증.
2. `data/dataset.py`가 raw / 외부 데이터를 `schema.Sample` 리스트로 정규화.
3. `prompting/builders.py`가 `num_prompt_variants`만큼 프롬프트 합성 (direct / evidence-first / bias-guarded).
4. `models/loader.py`가 `model.name` prefix로 VLM 어댑터 선택, 로컬 가중치만 허용.
5. `inference/predictor.py`가 배치 생성 → `prompting/parsers.py`로 0-based 라벨 추출 → 파싱 실패 시 fallback `0`.
6. `outputs/predictions/<exp>/`에 CSV + `timing.json` + `config.resolved.json` 기록.

### 앙상블 흐름 (규칙 #1 준수)

1. 각 베이스 모델 prediction을 입력으로 `inference/ensemble.py` 호출.
2. 앙상블도 LLM이 최종 라벨을 생성. 다수결 / 규칙 오버라이드 금지.
3. LLM이 근거를 보고 `ANSWER: N`으로 답하도록 프롬프트 구성.
4. DACON 제출 직렬화 → `validate_submission.py`로 1차 검증.

### 산출물 디렉터리 (실행 1회당)

| 파일 | 내용 |
|------|------|
| `<exp>.csv` | 모델 예측 (`sample_id,label`) |
| `<exp>.submission.json` | 제출 직전 검증용 JSON |
| `<exp>.timing.json` | 샘플 / 배치당 레이턴시 통계 (규칙 #4 점검) |
| `<exp>.config.resolved.json` | 머지 후 최종 Pydantic 직렬화 |
| `<exp>.raw_outputs.jsonl.gz` | LLM 원시 응답 (재현 디버깅용) |

## 빠른 시작 (Quickstart)

### 1) 설치

```bash
git clone <repo-url> skku-vqa
cd skku-vqa
make setup
```

`make setup`은 `pip install -e ".[vlm,dev]"`와 동등합니다 (vlm + dev 옵션 동시 설치).

### 2) 데이터 준비

평가 데이터는 DACON에서 다운로드하여 `data/raw/`에 read-only로 배치합니다. 합성 / 캘리브레이션 데이터는 `data/external/`에 둡니다.

```bash
make prepare
```

### 3) 추론 (GPU 워커)

dev 호스트에 GPU가 없을 수 있으므로 원격 GPU 워커에서 실행합니다.

```bash
# 원격 셋업 (1회)
bash scripts/remote/setup.sh

# 추론 예시 (Qwen3-VL-8B)
python scripts/run_inference.py \
  --config configs/experiment/qwen3vl_8b.yaml \
  --limit 32      # 스모크 서브셋, 전체 실행 시 생략
```

결과는 `outputs/predictions/qwen3vl_8b/`에 저장됩니다.

### 4) 앙상블 → 제출

```bash
python scripts/run_ensemble.py     --config configs/experiment/<ensemble>.yaml
python scripts/make_submission.py  --config configs/experiment/<ensemble>.yaml
python scripts/validate_submission.py --submission outputs/submissions/<exp>.csv
```

### 5) 최종 제출 재현

```bash
make reproduce
```

잠금된 [`artifacts/final/config.yaml`](artifacts/final/config.yaml)을 사용해 최종 제출을 재생성합니다.

## 설정 시스템 (Configuration)

- 모든 실험은 `configs/*.yaml`. Pydantic 모델(`src/skku_vqa/config.py`)이 로드·검증.
- 상속: config에 `defaults: configs/default.yaml`을 지정하면 base → override 순으로 deep-merge.
- 새 옵션을 추가할 때 **반드시** Pydantic 모델에도 추가. YAML만 수정하면 검증 단계에서 누락됩니다.
- `configs/default.yaml`은 경로 기본값(`data/raw/...`)과 모델 / 프롬프트 / 추론 / 제출 섹션을 보유.

| 섹션 | 핵심 키 |
|------|--------|
| `model` | `name`, `backend` (`vllm` \| `transformers`), `load_in_4bit` (transformers 전용), 가중치 경로 |
| `inference` | `num_prompt_variants`, `max_new_tokens`, 배치 크기, 시드 |
| `prompting` | 변종 선택 (`direct` / `evidence-first` / `bias-guarded`) |
| `submission` | 출력 경로, 컬럼 (`sample_id,label`) |
| `paths` | raw / external / outputs 디렉터리 |

## 명령어 레퍼런스 (Commands reference)

| 명령 | 설명 |
|------|------|
| `make setup` | `pip install -e ".[vlm,dev]"` |
| `make prepare` | 데이터 전처리 / 검증 (`scripts/prepare_data.py`) |
| `python scripts/run_inference.py --config <yaml> [--limit N]` | 단일 모델 추론. `--limit`은 스모크 서브셋. |
| `python scripts/run_ensemble.py --config <yaml>` | LLM-매개 앙상블 |
| `python scripts/make_submission.py --config <yaml>` | DACON 제출 CSV 생성 |
| `python scripts/validate_submission.py --submission <csv>` | 라벨 / 스키마 검증 |
| `python scripts/validate_local.py` | 로컬 데이터 정합성 추가 검증 |
| `make validate` | `validate_submission.py` 별칭 |
| `make reproduce` | `artifacts/final/config.yaml` 기반 최종 제출 재현 |
| `make test` | `pytest -q` |
| `make lint` | `ruff check src tests scripts` (line-length 100, py310, 규칙 `E,F,I,W,UP,B`) |
| `bash scripts/remote/setup.sh` | 원격 GPU 호스트 초기 셋업 |
| `bash scripts/remote/sync.sh` | 코드 / 데이터 동기화 |
| `bash scripts/remote/infer.sh` | 원격 추론 실행 |
| `python scripts/submit_dacon.py` | DACON 포털 업로드 (자격증명 필요) |
| `bash scripts/auto_submit_next_window.sh` | 다음 제출 윈도우 자동 트리거 |

> **주의**: `make infer` / `make ensemble` / `make submit`은 stale 타깃입니다. 위 표의 스크립트 직접 호출을 권장합니다.

## 배포·자동제출 (Deployment & auto-submit)

`deploy/systemd/`는 제출 윈도우 만료 전 자동 재제출을 위한 유닛입니다.

| 파일 | 역할 |
|------|------|
| `deploy/systemd/dacon-auto-submit.service` | 1회성 작업 유닛 |
| `deploy/systemd/dacon-auto-submit.timer` | 스케줄 트리거 |
| `deploy/systemd/README.md` | 설치·활성화 절차 |

활성화 예시 (`sudo` 필요):

```bash
sudo cp deploy/systemd/dacon-auto-submit.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dacon-auto-submit.timer
```

자격증명, 로그 위치, 재시도 정책 등 세부 절차는 [`deploy/systemd/README.md`](deploy/systemd/README.md) 참조.

## 로컬 개발·테스트 (Local development & testing)

- pytest 경로: `tests/`, `pythonpath=["src"]` 설정으로 `skku_vqa` 패키지를 설치 없이 import 가능.
- 단일 테스트: `pytest tests/test_parsers.py -q` 또는 `pytest tests/test_parsers.py::test_name -q`.
- ruff만 lint 타깃 (포맷터 없음). 줄 길이 100, Python 3.10, 규칙 `E,F,I,W,UP,B`.
- 타입 체크: `mypy`는 dev 의존성이지만 Make 타깃 없음. 필요 시 `mypy src` 수동 실행.
- 모든 코드 경로에 인코딩된 대회 하드 규칙을 깨지 않도록 PR 전 재확인.

## 운영·검증 (Operations & validation)

| 점검 항목 | 도구 / 출력 |
|----------|------------|
| 제출 포맷 | `scripts/validate_submission.py`, `submission/dacon.py` |
| Balanced Accuracy | `evaluation/metrics.py` |
| 레이턴시 예산 (규칙 #4) | `outputs/predictions/<exp>/timing.json` |
| 사용된 설정 재현 | `outputs/predictions/<exp>/config.resolved.json` |
| 원시 LLM 출력 | `outputs/predictions/<exp>/raw_outputs.jsonl.gz` |
| 리더보드 결과 | [`artifacts/final/SCORES.md`](artifacts/final/SCORES.md) |
| 최종 잠금 설정 | [`artifacts/final/config.yaml`](artifacts/final/config.yaml) |

## 기여 (Contributing)

- 절차: [`CONTRIBUTING.md`](CONTRIBUTING.md) 참조.
- 작업 전 [`AGENTS.md`](AGENTS.md)의 "Hard competition rules"를 재확인. 규칙 회피 경로 추가 금지.
- 새 모델 어댑터는 `models/base.py`를 상속하고 `models/loader.py` 디스패치 prefix에 등록.
- 새 프롬프트 변종은 `prompting/templates.py`에 정의하고 `parsers.py` 파싱 패턴과 일치 여부 검증 테스트 추가.
- 새 옵션은 YAML만이 아니라 Pydantic 모델에도 함께 추가.
- PR 전 `make test`, `make lint` 통과 필수.

## 라이선스·연락처 (License & contact)

- 라이선스: [`LICENSE`](LICENSE) 참조.
- 대회: [DACON #236722](https://dacon.io/competitions/official/236722/overview/description).
- 유지보수: 본 저장소 [`AGENTS.md`](AGENTS.md) 상단 명시 팀 / 개인.

## 참고 문서 (Further documentation)

- [`AGENTS.md`](AGENTS.md) — 운영 규칙, 명령어, 컨벤션.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — 기여 절차.
- [`artifacts/final/SCORES.md`](artifacts/final/SCORES.md) — 최종 리더보드 점수 / 설정 요약.
- [`deploy/systemd/README.md`](deploy/systemd/README.md) — 자동 재제출 유닛 설치.
- `docs/` — 대회 노트, 프롬프트 디자인, 재현 절차.