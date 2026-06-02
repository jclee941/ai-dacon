# 대회 노트 (DACON #236722)

## 과제
- 멀티모달(VLM) 이미지-텍스트 QA. 입력: **image, context, question, options** → 출력: 정답 **선택지 번호(label)**.
- BBQ 스타일 편향 벤치마크: 근거 있으면 정답, 근거 부족(ambiguous)이면 **unknown/판단불가** 선택.

## 평가
- 지표: **Balanced Accuracy** = (ambiguous 정확도 + disambiguated 정확도) / 2.
- ambiguous/disambiguated 라벨은 테스트셋에 **비공개**.
- Public 60% / Private 40%.

## 제출 흐름 (중요)
1. **리더보드**: 참가자가 직접 추론 → CSV(`sample_id,label`) 업로드 → 서버는 점수만 계산. **서버가 코드를 실행하지 않음.**
2. **2차 평가**: Private 상위 15팀만 추론 코드+가중치 제출 → 운영진이 기준 환경에서 재실행 검증.

## 규칙 핵심
- 언어: Python only.
- **외부 API 추론 금지** (OpenAI/Gemini/HF Inference/Together/OpenRouter 등). 로컬 가중치 로드만.
- 사전학습 모델: **2026.05.31 이전 공개 가중치만**. 출처/라이선스 기록 필수.
- **최종 답은 LLM이 생성** — 단순 룰/다수결/조건문/사전정의 목록 선택 **금지**. 앙상블해도 LLM이 종합 생성해야 함.
- 추론 시간: 샘플당 평균 **0.5초** 권장 (Test 8,500 ≈ 70분, Hidden 1,500 ≈ 13분).
- 1일 제출 5회, CSV UTF-8.
- Data Leakage 금지: 평가셋 분석해 유사 학습데이터/프롬프트 생성 금지.

## 기준 평가 환경 (2차 검증)
- RTX A6000 48GB, Python 3.10, CUDA 12.4, PyTorch 2.6.0, Ubuntu 20.04, **오프라인**.

## 우리 인프라
- 개발/추론: `youtube`(192.168.50.220) — Ubuntu 24.04, RTX 5070 Ti **16GB**, Py3.12.
- 16GB 한계: LLaVA-OV-0.5B/7B는 vLLM으로 가능. 큰 모델은 4bit(transformers) 또는 회피.
