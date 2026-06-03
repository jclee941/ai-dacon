# DACON 236722 (team qws941) — actual Balanced Accuracy scores

| submission | model / config | score |
|---|---|---|
| qwen3vl_8b.csv | Qwen3-VL-8B 4bit, bias_guarded, max_pixels 200704 | **0.95867** (best) |
| qwen3vl_8b_hires.csv | Qwen3-VL-8B 4bit, bias_guarded, max_pixels 786432 | 0.95767 |
| qwen3vl_8b_evidence_only.csv | Qwen3-VL-8B 4bit, evidence_only, max_pixels 786432 | 0.9295 |
| qwen3vl_8b_lowres.csv | Qwen3-VL-8B 4bit, bias_guarded, max_pixels 100352 | 0.95683 |
| internvl3_8b.csv | InternVL3-8B 4bit, bias_guarded, 448x448 tiling | not yet submitted |
| qwen3vl_8b_bgv2.csv | Qwen3-VL-8B 4bit, bias_guarded_v2, max_pixels 200704 | not yet submitted |
| qwen25vl_7b.csv | Qwen2.5-VL-7B 4bit, bias_guarded | 0.90658 |
| qwen25vl_7b_mp1m.csv | Qwen2.5-VL-7B, higher-res | 0.90458 |
| baseline_llava_ov_transformers.csv | LLaVA-OV-0.5B | 0.469 |
| sample_submission.csv | all-zero | 0.34083 |

## Learnings
- Model is the big lever: Qwen2.5-VL-7B 0.9066 -> Qwen3-VL-8B 0.9587.
- bias_guarded >> evidence_only (0.958 vs 0.9295) for this BBQ task.
- Resolution barely matters: max_pixels 200704=0.95867 > 786432=0.95767 > 100352=0.95683 (spread only 0.0018). 200704 is the sweet spot.

## Already submitted (KST 2026-06-03 window, all 5 daily slots used)
1. qwen3vl_8b.csv (200704, best 0.95867)
2. qwen3vl_8b_hires.csv (786432, 0.95767)
3. qwen3vl_8b_lowres.csv (100352, 0.95683)
4. qwen3vl_8b_evidence_only.csv (786432, 0.9295)
5. qwen25vl_7b.csv (0.90658)
All returned dacon_api_result isSubmitted=true detail=Success.

## [SUPERSEDED by "Updated next-window top-5 (loop 14)" below] Selected top-5 (earlier)
Strongest distinct candidates to submit after the next daily reset:
1. qwen3vl_8b.csv (confirmed best 0.95867)
2. internvl3_8b.csv (NEW InternVL3-8B; 1031/8500 labels differ from best; pending)
3. qwen3vl_8b_hires.csv (confirmed 0.95767)
4. qwen3vl_8b_lowres.csv (0.95683; 249 diff from best)
5. qwen25vl_7b.csv (floor 0.90658)  <-- SUPERSEDED: replaced by qwen3vl_8b_bgv2.csv in loop 14
Note: qwen3vl_8b_evidence_only.csv (0.9295) is intentionally EXCLUDED from the next-window top-5 as the weakest candidate.
Submit next window e.g.: python scripts/submit_dacon.py --submission artifacts/final/internvl3_8b.csv --team qws941

## Infra limits (learnings for future candidates)
- InternVL3-14B attempted as a stronger 6th candidate but REJECTED: 14B 4bit (~8.5GB) is too tight for ~7GB free VRAM (TTS holds ~8.4GB), and the 290GB host disk was 100% full (model is ~28GB download). Not viable on current infra without freeing disk + stopping TTS.
- Confirmed viable ceiling on this 16GB box (shared with TTS): ~8-9B VLMs in 4bit (Qwen3-VL-8B, InternVL3-8B). Larger models need the 2nd-stage A6000 48GB env.
- Next-window top-5 (canonical, see loop-14 section): qwen3vl_8b, internvl3_8b, qwen3vl_8b_hires, qwen3vl_8b_lowres, qwen3vl_8b_bgv2 (qwen25vl_7b dropped as weakest).

## Infra status (2026-06-03, loop 12)
- Host disk 13G free of 290G (96% used). Cannot download additional 8-9B VLMs (~16GB each) for new candidates. The next-window top-5 is the final feasible set on this box.
- Auto-submit timer dacon-auto-submit.timer active (persistent, linger), next fire Thu 2026-06-04 00:05 KST; will submit the 5 candidates after the daily cap resets.
- Further score gains require the 2nd-stage A6000 48GB env (larger models) — out of scope for this 16GB box.

## Updated next-window top-5 (loop 14)
Replaced weakest qwen25vl_7b (0.90658) with qwen3vl_8b_bgv2 (bias_guarded_v2 prompt, 279-label diff from best):
1. qwen3vl_8b.csv (0.95867 best)
2. internvl3_8b.csv (NEW model, 1031 diff)
3. qwen3vl_8b_hires.csv (0.95767)
4. qwen3vl_8b_lowres.csv (0.95683)
5. qwen3vl_8b_bgv2.csv (NEW prompt, 279 diff; cached-model reuse, no download)
Auto-submit script (scripts/auto_submit_next_window.sh) updated to this set.

## Finding (loop 16): InternVL3 is prompt-insensitive; pool unchanged
- InternVL3-8B + bias_guarded_v2 vs InternVL3-8B + bias_guarded: only 2/128 labels differ (near-identical). The SAME prompt swap on Qwen3-VL gave 279/8500 diff.
- Conclusion: bias_guarded_v2 meaningfully changes Qwen3-VL but is a near no-op on InternVL3. The internvl3_8b_bgv2 candidate was rejected as a non-distinct duplicate and NOT added to the pool.
- Canonical next-window top-5 unchanged: qwen3vl_8b, internvl3_8b, qwen3vl_8b_hires, qwen3vl_8b_lowres, qwen3vl_8b_bgv2.

## Hardening (loop 20): auto-submit runtime risk removed
- Pre-built the repo-local isolated venv `.dacon_auto_venv` (gitignored) with `dacon_submit_api-0.1.2` BEFORE the midnight fire, so the timer no longer depends on curl/network/wheel availability at 00:05.
- Verified import chain under that venv: `submit_dacon.py` -> `skku_vqa.submission.dacon` (src) + `dacon_submit_api.post_submission_file`. Bootstrap now skips (venv exists); script `VENV` path matches the pre-built one.
- Timer next fire: Thu 2026-06-04 00:05 KST (enabled, active waiting).

## Hardening (loop 21): final root = exactly the 5 next-window CSVs
- Moved already-submitted, non-next-window candidates `qwen25vl_7b.csv` (0.90658) and `qwen3vl_8b_evidence_only.csv` (0.9295) to `artifacts/final/superseded/` so the final root holds EXACTLY the 5 canonical next-window CSVs (qwen3vl_8b, internvl3_8b, qwen3vl_8b_hires, qwen3vl_8b_lowres, qwen3vl_8b_bgv2). Receipts preserved, not deleted.
- auto_submit_next_window.sh: idempotency markers (.dacon_auto_state/<name>.done, gitignored) skip already-succeeded candidates on re-fire (cap-safe); marker write guarded; per-candidate failures are now counted and propagated to a non-zero script exit so systemd reflects partial-submission failure instead of false success.

## Hardening (loop 22): pre-midnight preflight (--dry-run)
- Added `--dry-run` to submit_dacon.py: validates .env token + competition_id + submission file existence + dacon_submit_api import, WITHOUT posting to DACON. Lets us confirm BEFORE midnight that the timer won't fail on all 5 candidates due to a bad token / missing file / broken venv.
- Ran the preflight NOW on all 5 candidates via the pre-built venv: every one returned `DRY-RUN OK` (token+comp+file+api verified).
- auto_submit_next_window.sh: runs the preflight for every candidate up front; if any fails it logs PREFLIGHT FAILED and exits 1 before posting anything (does not burn the daily-cap window on doomed submissions).
- Tests: tests/test_submit_dry_run.py covers dry-run OK / missing-token-fails / missing-file-fails (uses the venv python for the import check). Suite 56 passed.

## Hardening (loop 23): observability via Telegram notification
- auto_submit_next_window.sh notifies Telegram on EVERY exit: success ("all 5 candidates submitted"), aggregate per-candidate failure, AND every early fatal exit (wheel download / venv install / preflight failure) via a `fail()` helper. No exit path is silent — a midnight miss is always observable.
- Opt-in design: reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from .env (gitignored, no hardcoded secrets). If absent, notify() returns 0 silently — zero regression to the submit flow.
- Verified end-to-end NOW: real success-style and fail()-style test messages delivered (telegram ok=True); the no-creds fallback returns 0 (silent skip) so notify() can never break or change the submit exit code.

## Hardening (loop 25): systemd timeout reproducibility
- dacon-auto-submit.service now declares `TimeoutStartSec=infinity` (deploy copy + installed unit, daemon-reloaded). The auto-submit script can loop for hours waiting for the daily-cap reset (10-min retry); without this, a host whose DefaultTimeoutStartSec is finite would SIGTERM the oneshot mid-wait and silently miss the window. Effective value confirmed `TimeoutStartUSec=infinity`; timer still enabled, next fire Thu 2026-06-04 00:05 KST.

## Leaderboard update (loop 25, real scores in)
- qwen3vl_8b_lowres.csv scored 0.95683 (was pending). Full confirmed ranking: qwen3vl_8b 0.95867 (best) > hires 0.95767 > lowres 0.95683 > evidence_only 0.9295 > qwen25vl_7b 0.90658 > mp1m 0.90458 > llava 0.469 > all-zero 0.34083.
- Of the next-window top-5, THREE are already scored (qwen3vl_8b 0.95867, hires 0.95767, lowres 0.95683). Re-submitting them is harmless (idempotent) but yields no new info.
- The only UNSCORED candidates are internvl3_8b.csv and qwen3vl_8b_bgv2.csv. The next window's real value = learning whether either beats 0.95867. internvl3_8b differs from best on 1031/8500 labels (genuinely different model); bgv2 differs on 279/8500 (prompt variant). Neither is guaranteed to beat best; qwen3vl_8b remains the safe top pick already locked in at 0.95867.

## Model survey (loop 25, librarian bg_b7c9b956) — validates current pool
- Searched for transformers-4.57-compatible VLMs that could BEAT Qwen3-VL-8B (0.95867) in 4bit on ~7GB free VRAM. Decisive result:
  - Qwen3-VL-30B-A3B (MoE): REJECTED — 31B total params ≈ 15.5GB at 4bit (all experts must be VRAM-resident even with 3B active); >7GB budget. class Qwen3VLMoeForConditionalGeneration exists in 4.57 but infeasible here.
  - Qwen3-VL-8B-Thinking: REJECTED — fits (~5GB) but CoT generates hundreds of extra tokens/sample, blows the 0.5s/sample budget; reasoning gains are STEM-oriented, not BBQ-relevant.
  - InternVL3-8B-hf: #1 PICK — ~4.9GB 4bit, native transformers 4.57, strongest MMMU/MMStar position. ALREADY in our pool as internvl3_8b.csv (unscored, queued for next window). Survey confirms it was the right new candidate.
  - InternVL2.5-8B: #2 fallback — ~4-5GB 4bit but needs trust_remote_code and trails InternVL3 by ~3-5% MMMU. Only worth trying on the inference host (youtube 192.168.50.220), NOT here.
- This box (jclee-dev) has NO GPU and NO Qwen3-VL/InternVL weights — new candidate CSVs cannot be generated here; inference runs on youtube 192.168.50.220. So the survey changes nothing actionable now: the next-window pool already contains the survey's #1 recommendation. Larger/stronger models remain a 2nd-stage A6000 48GB option.
