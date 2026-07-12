# SKKU VQA

2026 SKKU Multimodal AI Challenge (DACON #236722) 제출용 멀티모달 VQA 파이프라인.
LLM이 생성한 텍스트에서 최종 라벨을 직접 파싱하고, ambiguous 케이스는 매핑 규칙이 아닌
가드 프롬프트로만 처리하며, 로컬 가중치만으로 오프라인 추론을 수행합니다.

## English Summary

Multimodal bias-aware VQA pipeline for the 2026 SKKU Multimodal AI Challenge
(DACON #236722). Parses the final answer directly from LLM-generated text, handles
ambiguous cases via prompt-only strategies, and runs fully offline with local weights.

## Status

| 항목 | 상태 |
| --- | --- |
| 메인 라인 | `main` |
| 추론 백엔드 | vLLM (기본), `transformers` (옵션, 4bit 지원) |
| 지원 모델 | LLaVA-OneVision, InternVL3, Qwen3-VL, Qwen-VL |
| 추론 실행 환경 | GPU 호스트 필요 (16GB ≈ ~8–9B 4bit, 48GB ≈ 더 큰 모델) |
| 네트워크 | 완전 오프라인 (외부 추론 API 금지) |
| 데이터 누출 | 평가 세트 분석 금지 |
| 지연 예산 | 평균 ~0.5s/sample (`timing.json`으로 점검) |
| 운영 준비도 | 챌린지 제출용 — 프로덕션 일반화 목적 아님 |

## 파이프라인 흐름

```
prepare → infer → (ensemble) → submit → validate → (auto-submit)
```

| 단계 | 명령 | 책임 |
| --- | --- | --- |
| 데이터 준비 | `make prepare` | 원본을 학습/추론용 포맷으로 변환 |
| 단일 모델 추론 | `python scripts/run_inference.py --config <yaml>` | 모델별 `submission.json` |
| 다중 모델 추론 | 동일 스크립트, ensemble 설정 사용 | 모델 N개 답변 통합 |
| 제출 파일 | `make submit` | `submission.csv` 생성 |
| 검증 | `make validate` | 스키마/제약 검사 |
| 재현 | `make reproduce` | `artifacts/final/config.yaml` 잠긴 설정 실행 |

## 목차

1. 먼저 읽을 파일
2. 패키지 구성
3. 아키텍처
4. 진입점
5. 빠른 시작
6. 디렉터리 구조
7. 모델
8. 설정 시스템
9. 라벨 의미
10. 명령어 레퍼런스
11. 로컬 개발 & 테스트
12. 제출 & 자동화
13. 기여 가이드
14. 유지보수 & 연락처
15. 추가 문서
16. 라이선스

## 먼저 읽을 파일

| 파일 | 읽는 이유 |
| --- | --- |
| [`AGENTS.md`](AGENTS.md) | 작업 규칙, 컴페티션 규칙, 흔한 함정 요약 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | PR / 리뷰 절차 |
| [`pyproject.toml`](pyproject.toml) | 의존성, 콘솔 스크립트, 린트/테스트 설정 |
| `src/skku_vqa/config.py` | 설정 스키마 (Pydantic) |
| `src/skku_vqa/prompting/parsers.py` | 라벨 파싱 규칙 |
| [`artifacts/final/SCORES.md`](artifacts/final/SCORES.md) | 잠긴 최종 점수/리포트 |

## 패키지 구성

소스는 `src/skku_vqa/` 아래 모듈로 분리되어 있습니다.

| 모듈 | 역할 |
| --- | --- |
| `cli` | 콘솔 진입점 (`skku-vqa`) |
| `config` | Pydantic 기반 YAML 로더/검증 |
| `constants` | 라벨 매핑, unknown 패턴 등 상수 |
| `models/` | VLM 래퍼 (`loader`, `base`, `internvl3`, `llava_ov*`, `qwen*`) |
| `inference/` | `predictor`, 다중 모델 `ensemble` |
| `prompting/` | `builders`, `parsers`, `templates` |
| `data/` | `dataset`, `schema`, `validation` |
| `evaluation/metrics` | 로컬 평가 지표 |
| `ambiguity/` | ambiguous 탐지/매핑 (로컬 분석 전용) |
| `submission/dacon` | DACON 형식 변환 |
| `utils/` | I/O, seed 등 |

## 아키텍처

| 계층 | 책임 | 핵심 파일 |
| --- | --- | --- |
| 데이터 | 로드/검증/스키마 | `data/dataset.py`, `data/schema.py`, `data/validation.py` |
| 프롬프트 | 템플릿 + 빌더 + 파서 | `prompting/templates.py`, `builders.py`, `parsers.py` |
| 모델 | 디스패치 + 래퍼 | `models/loader.py`, `models/base.py`, `models/<family>.py` |
| 추론 | 예측 + 앙상블 | `inference/predictor.py`, `inference/ensemble.py` |
| 평가 | 메트릭 (로컬) | `evaluation/metrics.py` |
| 제출 | DACON 직렬화 | `submission/dacon.py` |
| CLI/스크립트 | 오퍼레이터 인터페이스 | `cli.py`, `scripts/*.py` |

요청 흐름:

1. `scripts/run_inference.py`가 설정/데이터 로드
2. `prompting.builders`가 모델별 프롬프트 생성
3. `models.loader`가 가중치 로드 (오프라인 검증)
4. `inference.predictor`가 LLM 생성 → `parsers`가 라벨 추출
5. 결과가 `outputs/predictions/<exp>/`에 저장, `submission/dacon`이 CSV로 직렬화

## 진입점

| 종류 | 이름 | 위치 |
| --- | --- | --- |
| 콘솔 스크립트 | `skku-vqa` | `src/skku_vqa/cli.py:main` |
| 파이썬 패키지 | `skku_vqa` | `src/skku_vqa/__init__.py` |
| 데이터 준비 | `prepare_data.py` | `scripts/` |
| 추론 | `run_inference.py` | `scripts/` |
| 제출 생성 | `make_submission.py` | `scripts/` |
| DACON 업로드 | `submit_dacon.py` | `scripts/` |
| 자동 제출 셸 | `auto_submit_next_window.sh` | `scripts/` |
| 재현 스크립트 | `reproduce_submission.py` | `scripts/` |
| 검증 스크립트 | `validate_submission.py`, `validate_local.py` | `scripts/` |
| 원격 셋업 | `scripts/remote/{setup,sync,infer}.sh` | `scripts/remote/` |

## 빠른 시작

```bash
git clone <repo-url> skku-vqa
cd skku-vqa
make setup
make prepare
python scripts/run_inference.py --config configs/experiment/qwen3vl_8b.yaml --limit 4
python scripts/validate_submission.py --submission outputs/submissions/<exp>/submission.csv
make reproduce
```

주의:

- 추론은 GPU 호스트에서 실행합니다. 데브 박스에서는 편집/검증만 가능합니다.
- `Makefile`의 `infer`/`ensemble`/`submit` 타깃은 아직 존재하지 않는 config를 가리키므로
  실제 `configs/experiment/*.yaml`을 골라 스크립트에 직접 넘기세요.
- `--limit N`은 스모크 서브셋 실행용입니다.

## 디렉터리 구조

| 경로 | 내용 |
| --- | --- |
| `src/skku_vqa/` | 파이썬 패키지 본체 |
| `scripts/` | 파이프라인 스크립트 |
| `scripts/remote/` | 원격 호스트 셋업/동기화 헬퍼 |
| `artifacts/final/` | 잠긴 최종 설정, 모델별 제출물, 타이밍, 점수 |
| `artifacts/lora_adapters/` | LoRA 어댑터 |
| `deploy/systemd/` | 자동 제출용 systemd 유닛 + README |
| `AGENTS.md` | 작업자/에이전트 가이드 |
| `CONTRIBUTING.md` | 기여 절차 |
| `Makefile` | 자주 쓰는 명령 |
| `pyproject.toml` | 의존성, 콘솔 스크립트, 린트/테스트 설정 |
| `LICENSE` | 라이선스 |

## 모델

`models/loader.py`가 `model.name` 접두사로 디스패치합니다.

| 접두사 | 비고 |
| --- | --- |
| `llava*` | 기본 `vllm`, `transformers` 옵션 (4bit 지원) |
| `internvl*` | InternVL3 (8B 검증) |
| `qwen3_vl*` | Qwen3-VL (8B, lowres/hires/bgv2 변형) |
| `qwen*` | Qwen2.5-VL 등 |

각 추론 실행은 `outputs/predictions/<exp>/timing.json`에 타이밍을 기록합니다.
지연 예산 점검에 사용하세요.

## 설정 시스템

- 모든 실험은 YAML, `src/skku_vqa/config.py`에서 Pydantic으로 로드/검증.
- `defaults: configs/default.yaml`로 베이스 상속, 딥 머지.
- 새 옵션은 YAML만 건드리지 말고 Pydantic 모델에도 추가하세요.
- `configs/default.yaml`은 경로 기본값과 모델/프롬프트/추론/제출 섹션을 보유.

## 라벨 의미 (자주 혼동)

- `submission.csv` 컬럼: `sample_id,label`. `label`은 **0-based** 옵션 인덱스.
- 프롬프트는 옵션을 `0..N-1`로 표기, 파서는 정수 추출 (`ANSWER: N` 우선, 없으면 마지막 독립 정수).
- 파싱 실패 시 `predictor.py`는 빈 행 방지를 위해 `0`으로 폴백.
- "Unknown / cannot determine"은 매핑 규칙이 아닌 `bias_guarded` 프롬프트 전략으로 처리.
- `constants.UNKNOWN_OPTION_PATTERNS`는 로컬 분석 전용이며 제출 라벨에 영향을 주지 않습니다.

## 명령어 레퍼런스

| 명령 | 동작 |
| --- | --- |
| `make setup` | `pip install -e ".[vlm,dev]"` |
| `make prepare` | `python scripts/prepare_data.py --config configs/default.yaml` |
| `make infer` | `run_inference.py` 실행 (stale config 주의) |
| `make ensemble` | `run_ensemble.py` 실행 (stale config 주의) |
| `make submit` | `make_submission.py` 실행 (stale config 주의) |
| `make validate` | `validate_submission.py` 실행 |
| `make reproduce` | `reproduce_submission.py` (잠긴 설정) |
| `make test` | `pytest -q` |
| `make lint` | `ruff check src tests scripts` |

`mypy`는 dev 의존성으로 설치되지만 별도 타깃은 없습니다. 필요 시 `mypy src`.

## 로컬 개발 & 테스트

- Python 3.10+, GPU는 추론 시에만 필요.
- `pytest`는 `pythonpath=["src"]`로 동작하므로 패키지 설치 없이 import 가능.
- 단일 테스트: `pytest tests/test_parsers.py -q` 또는
  `pytest tests/test_parsers.py::test_name`.
- 린트 규칙: `E,F,I,W,UP,B`, line-length 100, py310.

## 제출 & 자동화

- `scripts/make_submission.py`가 `submission.csv` 생성 (`sample_id,label`, 0-based).
- `scripts/submit_dacon.py`로 DACON 업로드.
- `scripts/auto_submit_next_window.sh`로 다음 제출 윈도우 자동화.
- [`deploy/systemd/README.md`](deploy/systemd/README.md)의
  `dacon-auto-submit.service` / `.timer`로 systemd 기반 자동 제출 가능.

## 기여 가이드

- 작업 전 [`AGENTS.md`](AGENTS.md)의 컴페티션 규칙을 다시 확인하세요.
- 새 옵션은 YAML과 Pydantic 모델 양쪽에 동기화.
- 라벨 파싱/폴백 로직은 컴페티션 규칙에 직결되므로 신중하게 변경.
- PR 전 `make lint`와 `make test` 통과.

## 유지보수 & 연락처

- 챌린지: 2026 SKKU Multimodal AI Challenge (DACON #236722)
- 작업자 가이드: [`AGENTS.md`](AGENTS.md)
- 기여 절차: [`CONTRIBUTING.md`](CONTRIBUTING.md)

## 추가 문서

| 문서 | 내용 |
| --- | --- |
| [`AGENTS.md`](AGENTS.md) | 컴페티션 규칙, 함정, 워크플로 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 기여 절차 |
| [`deploy/systemd/README.md`](deploy/systemd/README.md) | systemd 자동 제출 설정 |
| [`artifacts/final/SCORES.md`](artifacts/final/SCORES.md) | 잠긴 최종 점수/리포트 |
| [`artifacts/final/config.yaml`](artifacts/final/config.yaml) | 재현용 최종 설정 |

## 라이선스

[`LICENSE`](LICENSE) 파일을 참조하세요.