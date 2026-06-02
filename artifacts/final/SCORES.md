# DACON 236722 (team qws941) — actual Balanced Accuracy scores

| submission | model / config | score |
|---|---|---|
| qwen3vl_8b.csv | Qwen3-VL-8B 4bit, bias_guarded, max_pixels 200704 | **0.95867** (best) |
| qwen3vl_8b_hires.csv | Qwen3-VL-8B 4bit, bias_guarded, max_pixels 786432 | 0.95767 |
| qwen3vl_8b_evidence_only.csv | Qwen3-VL-8B 4bit, evidence_only, max_pixels 786432 | 0.9295 |
| qwen3vl_8b_lowres.csv | Qwen3-VL-8B 4bit, bias_guarded, max_pixels 100352 | pending |
| qwen25vl_7b.csv | Qwen2.5-VL-7B 4bit, bias_guarded | 0.90658 |
| qwen25vl_7b_mp1m.csv | Qwen2.5-VL-7B, higher-res | 0.90458 |
| baseline_llava_ov_transformers.csv | LLaVA-OV-0.5B | 0.469 |
| sample_submission.csv | all-zero | 0.34083 |

## Learnings
- Model is the big lever: Qwen2.5-VL-7B 0.9066 -> Qwen3-VL-8B 0.9587.
- bias_guarded >> evidence_only (0.958 vs 0.9295) for this BBQ task.
- Higher resolution does NOT help: 200704 (0.95867) >= 786432 (0.95767); lowres 100352 ~ same.

## Today's submissions (KST 2026-06-03, all 5 daily slots used)
1. qwen3vl_8b.csv (200704, best 0.95867)
2. qwen3vl_8b_hires.csv (786432, 0.95767)
3. qwen3vl_8b_lowres.csv (100352, pending)
4. qwen3vl_8b_evidence_only.csv (786432, 0.9295)
5. qwen25vl_7b.csv (0.90658)
All returned dacon_api_result isSubmitted=true detail=Success.
