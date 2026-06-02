# DACON 236722 (team qws941) — actual Balanced Accuracy scores

| submission | model / config | score |
|---|---|---|
| qwen3vl_8b.csv | Qwen3-VL-8B 4bit, bias_guarded, max_pixels 200704 | **0.95867** (best) |
| qwen3vl_8b_hires.csv | Qwen3-VL-8B 4bit, bias_guarded, max_pixels 786432 | 0.95767 |
| qwen3vl_8b_evidence_only.csv | Qwen3-VL-8B 4bit, evidence_only, max_pixels 786432 | 0.9295 |
| qwen3vl_8b_lowres.csv | Qwen3-VL-8B 4bit, bias_guarded, max_pixels 100352 | pending |
| internvl3_8b.csv | InternVL3-8B 4bit, bias_guarded, 448x448 tiling | not yet submitted |
| qwen25vl_7b.csv | Qwen2.5-VL-7B 4bit, bias_guarded | 0.90658 |
| qwen25vl_7b_mp1m.csv | Qwen2.5-VL-7B, higher-res | 0.90458 |
| baseline_llava_ov_transformers.csv | LLaVA-OV-0.5B | 0.469 |
| sample_submission.csv | all-zero | 0.34083 |

## Learnings
- Model is the big lever: Qwen2.5-VL-7B 0.9066 -> Qwen3-VL-8B 0.9587.
- bias_guarded >> evidence_only (0.958 vs 0.9295) for this BBQ task.
- Higher resolution does NOT help: 200704 (0.95867) >= 786432 (0.95767); lowres 100352 ~ same.

## Already submitted (KST 2026-06-03 window, all 5 daily slots used)
1. qwen3vl_8b.csv (200704, best 0.95867)
2. qwen3vl_8b_hires.csv (786432, 0.95767)
3. qwen3vl_8b_lowres.csv (100352, pending)
4. qwen3vl_8b_evidence_only.csv (786432, 0.9295)
5. qwen25vl_7b.csv (0.90658)
All returned dacon_api_result isSubmitted=true detail=Success.

## Selected top-5 for NEXT submission window (prepared, not yet submitted)
Strongest distinct candidates to submit after the next daily reset:
1. qwen3vl_8b.csv (confirmed best 0.95867)
2. internvl3_8b.csv (NEW InternVL3-8B; 1031/8500 labels differ from best; pending)
3. qwen3vl_8b_hires.csv (confirmed 0.95767)
4. qwen3vl_8b_lowres.csv (pending; 249 diff from best)
5. qwen25vl_7b.csv (floor 0.90658)
Note: qwen3vl_8b_evidence_only.csv (0.9295) is intentionally EXCLUDED from the next-window top-5 as the weakest candidate.
Submit next window e.g.: python scripts/submit_dacon.py --submission artifacts/final/internvl3_8b.csv --team qws941

## Infra limits (learnings for future candidates)
- InternVL3-14B attempted as a stronger 6th candidate but REJECTED: 14B 4bit (~8.5GB) is too tight for ~7GB free VRAM (TTS holds ~8.4GB), and the 290GB host disk was 100% full (model is ~28GB download). Not viable on current infra without freeing disk + stopping TTS.
- Confirmed viable ceiling on this 16GB box (shared with TTS): ~8-9B VLMs in 4bit (Qwen3-VL-8B, InternVL3-8B). Larger models need the 2nd-stage A6000 48GB env.
- Next-window top-5 remains the strongest feasible set: qwen3vl_8b, internvl3_8b, qwen3vl_8b_hires, qwen3vl_8b_lowres, qwen25vl_7b.
