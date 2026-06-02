# 프롬프트 전략 (규칙 #5 부합)

최종 답은 항상 LLM이 생성한다. 룰 override 없음. 템플릿은 `src/skku_vqa/prompting/templates.py`.

| 템플릿 | 용도 |
|--------|------|
| `direct` | 빠른 베이스라인. 근거 부족 시 unknown 선택 유도 |
| `bias_guarded` | 사회적 속성 추정 금지 강조. 기본값 |
| `evidence_then_answer` | 근거 단계적 검토 후 `ANSWER: N` 생성. 정확도↑, 시간↑ |

## Balanced Accuracy 튜닝 포인트
- ambiguous에서 unknown을 과도하게 고르면 disambiguated 정확도가 떨어진다.
- 로컬 캘리브레이션셋(`data/external/`)으로 두 그룹 균형을 점검 (`evaluation/metrics.py`).
- 0.5s/sample 제약 때문에 다중 프롬프트 앙상블은 신중히. 앙상블 시에도 최종은 LLM 종합 생성.
