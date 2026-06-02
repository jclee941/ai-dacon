# 재현성 (2차 평가 대비)

## 한 줄 재현
```bash
make reproduce   # artifacts/final/config.yaml 기반 추론 → 제출 CSV 생성
```

## 최종 제출 시 고정해야 할 것
- `artifacts/final/config.yaml` — 최종 실험 설정 스냅샷
- 모델 ID + revision/commit, 라이선스, 다운로드 경로
- 프롬프트 템플릿 (`src/skku_vqa/prompting/`)
- 시드(42), 디코딩 파라미터(temperature=0)
- 의존성 버전 (torch, transformers, vllm 등) — lock 파일

## 기준 환경 체크리스트
- [ ] RTX A6000 48GB에서 OOM 없이 실행
- [ ] 샘플당 평균 ≤ 0.5초 (outputs/predictions/<exp>/timing.json 확인)
- [ ] 오프라인(인터넷 차단) 실행 — 모델 가중치 사전 다운로드/캐싱
- [ ] Python 3.10 / CUDA 12.4 / PyTorch 2.6.0 호환
- [ ] 외부 API 호출 코드 없음
- [ ] 최종 label이 LLM 생성 텍스트에서 파싱됨 (룰 override 없음)

## 우리 개발 환경 (참고)
| 항목 | youtube (개발) | 기준 (2차) |
|------|----------------|------------|
| GPU | RTX 5070 Ti 16GB | RTX A6000 48GB |
| Python | 3.12 | 3.10 |
| 비고 | 빠른 반복/측정 | 최종 재현 검증 |
