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

## 제출 운영 runbook (자정 자동 제출 + 수동 백업)
다음 제출창(매일 00:05 KST 한도 리셋)에는 systemd user 타이머 `dacon-auto-submit.timer`가
`scripts/auto_submit_next_window.sh qws941`을 자동 실행하여 top-5 후보를 제출한다.
각 단계는 멱등(.dacon_auto_state/<name>.done), cap-safe(한도 시 10분 재시도), preflight 검증,
전 exit Telegram 알림을 포함한다.

**타이머가 어떤 이유로든 발화하지 않으면(수동 백업):**
```bash
cd /home/jclee/dev/ai-dacon
# 전체 5개 자동(권장; 멱등하므로 중복 제출 안 함):
./scripts/auto_submit_next_window.sh qws941
# 또는 후보 1개씩 직접:
.dacon_auto_venv/bin/python scripts/submit_dacon.py \
  --submission artifacts/final/qwen3vl_8b.csv --team qws941
```
- 사전 점검(제출 없이): `... scripts/submit_dacon.py --submission <csv> --team qws941 --dry-run`
- 인자 이름은 `--submission`(파일 경로), `--team`(qws941). 토큰/대회ID는 `.env`에서 자동 로드.
- 상태 확인: `systemctl --user list-timers dacon-auto-submit.timer`, 로그 `/tmp/dacon_auto_submit.log`.
- 후보 5개: qwen3vl_8b(best 0.95867), internvl3_8b, qwen3vl_8b_hires, qwen3vl_8b_lowres, qwen3vl_8b_bgv2 (artifacts/final/).
