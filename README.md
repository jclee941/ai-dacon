# skku-vqa

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![DACON](https://img.shields.io/badge/DACON-236722-orange.svg)
![Inference](https://img.shields.io/badge/inference-offline--only-critical.svg)
![Status](https://img.shields.io/badge/status-active--competition-green.svg)
![License](https://img.shields.io/badge/license-see%20LICENSE-lightgrey.svg)

## 한 줄 요약 (Summary)

`skku-vqa` 는 2026 SKKU 멀티모달 AI 챌린지(DACON #236722) 제출용 Python 패키지입니다. LLaVA-OV · InternVL3 · Qwen3-VL · Qwen-VL 등 로컬 VLM 가중치를 적재해 편향-가드(bias-guarded) 프롬프트로 추론하고, LLM 출력 텍스트에서 **0-based 옵션 인덱스**를 파싱해 DACON 제출 CSV(`sample_id,label`)를 만듭니다. 외부 추론 API 없이 **완전 오프라인**으로 동작하며, 운영 자동화는 `deploy/systemd/` 의 systemd 타이머가 담당합니다.

A Python pipeline for the 2026 SKKU Multimodal AI Challenge (DACON #236722): local VLM weights → bias-guarded prompting → 0-based label parsing → DACON submission CSV. Fully offline, no external inference APIs.

## 빠른 상태 (Status at a Glance)

| 항목 | 값 |
| --- | --- |
| 패키지 / 임포트 | `skku-vqa` / `skku_vqa` |
| Python | `>=3.10` |
| CLI 진입점 | 콘솔 스크립트 `skku-vqa` → `skku_vqa.cli:main` |
| 지원 모델 | LLaVA-OV, InternVL3, Qwen3-VL, Qwen-VL (`models/loader.py` prefix 디스패치) |
| 추론 백엔드 | `vllm` (기본), `transformers` (`load_in_4bit` 지원) |
| 제출 컬럼 | `sample_id, label` (label = 0-based 옵션 인덱스) |
| 지연 예산 | 표본 평균 ≈ 0.5초 (`outputs/predictions/<exp>/timing.json` 로 검증) |
| 운영 모드 | 오프라인 전용, 외부 추론 API 금지 |
| 추론 호스트 | GPU 워커 필요 (16GB VRAM → ~8–9B 4bit 한도) |
| 자동 제출 | `deploy/systemd/dacon-auto-submit.timer` |
| 대회 상태 | 활성 경연 (DACON #236722) |