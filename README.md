# 2026 SKKU Multimodal AI Challenge (DACON #236722)

멀티모달 AI(VLM) 기반 이미지-텍스트 질의응답 — 편향 인식 및 불확실성 처리.

- 대회: https://dacon.io/competitions/official/236722/overview/description
- 과제: 이미지 + 질문 + 선택지 → 최적 답변 예측 (BBQ 스타일 멀티모달 편향 벤치마크)
- 평가지표: **Balanced Accuracy** (ambiguous / disambiguated 그룹 정확도 평균)
- 데이터: 평가 데이터만 제공, 학습 데이터는 예시 1개. 참가자가 직접 학습 데이터 구성.

## 핵심 전략

1. **Zero-shot VLM 우선** — 강력한 오픈웨이트 VLM(Qwen2.5-VL, InternVL)으로 프롬프트 기반 추론.
2. **불확실성 탐지** — 근거가 부족하면 "알 수 없음" 선택지로 강제 매핑 (ambiguous 정확도 핵심).
3. **프롬프트 앙상블** — direct / evidence-first / bias-guarded 프롬프트 투표.
4. **합성 데이터 + LoRA** — zero-shot이 정체될 때만 도입.

## 빠른 시작

```bash
make setup        # 의존성 설치
make prepare      # 데이터 전처리/검증
make infer        # 추론 실행
make submit       # 제출 파일 생성
make validate     # 제출 형식 검증
make reproduce    # 최종 제출 재현
```

## 디렉토리

| 경로 | 역할 |
|------|------|
| `configs/` | 모든 실험 설정 (모델/프롬프트/실험) |
| `data/raw/` | DACON 원본 데이터 (read-only) |
| `data/external/` | 직접 구성한 합성/캘리브레이션 데이터 |
| `src/skku_vqa/` | 핵심 소스 코드 |
| `scripts/` | 실행 스크립트 |
| `outputs/` | 예측/제출/리포트 (gitignored) |
| `artifacts/final/` | 리더보드 제출에 사용된 최종 산출물 |
| `docs/` | 대회 노트, 재현성 문서 |

자세한 내용은 `docs/` 참고.
