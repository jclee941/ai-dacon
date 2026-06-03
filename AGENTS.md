# AGENTS.md

2026 SKKU Multimodal AI Challenge (DACON #236722). Multimodal bias-aware VQA. Python package `skku_vqa` under `src/`.

## Commands
- Install: `make setup` (= `pip install -e ".[vlm,dev]"`).
- Test: `make test` (= `pytest -q`). Single test: `pytest tests/test_parsers.py -q` or `pytest tests/test_parsers.py::test_name`.
- Lint: `make lint` (= `ruff check src tests scripts`). No formatter step; ruff lint only (line-length 100, py310, rules `E,F,I,W,UP,B`).
- Typecheck: `mypy` is a dev dep but has no Make target / config; run `mypy src` manually if needed.
- Pipeline: `make prepare` → `make infer` → `make submit` → `make validate`. `make reproduce` runs the locked final config.
- `pytest` uses `pythonpath=["src"]`, so tests import `skku_vqa` without install; runtime scripts insert `src` on `sys.path` themselves.

## Caveat: stale Makefile target
`make infer` points at `configs/experiment/zero_shot_qwen.yaml`, which does **not** exist. Run inference directly against a real config instead, e.g.
`python scripts/run_inference.py --config configs/experiment/qwen3vl_8b.yaml [--limit N]`.
`--limit N` runs a smoke subset.

## Config system
- All experiments are YAML in `configs/`. Loaded + validated via Pydantic in `src/skku_vqa/config.py`.
- Inheritance: a config may set `defaults: configs/default.yaml`; loader deep-merges base then overrides. Add new knobs to the Pydantic models, not just YAML.
- `configs/default.yaml` holds path defaults (`data/raw/...`), model/prompt/inference/submission sections.

## Hard competition rules (encoded throughout the code — do not break)
These are referenced in source as "규칙 #N". Violating them disqualifies the submission.
- **Final label MUST be parsed from LLM-generated text.** No rule/majority/conditional override of the answer. See `predictor.py` and `prompting/parsers.py`. Ensemble must still produce the answer via an LLM, not by voting logic.
- **No external inference APIs** (OpenAI/Gemini/HF Inference/etc). Local weights only. `models/loader.py` enforces this. `.env` `OPENAI_API_KEY`/`HF_TOKEN` are dev-only.
- **Must run fully offline** (no internet at scoring). Pre-download/cache weights.
- **Latency budget ≈ 0.5s/sample average.** Keep `inference.num_prompt_variants` low. Each run writes `outputs/predictions/<exp>/timing.json` — check it.
- Pretrained weights must be publicly released before 2026-05-31; record model id + license.
- No data leakage: do not analyze the eval set to craft similar train data/prompts.

## Label semantics (easy to get wrong)
- Submission CSV columns: `sample_id,label`. `label` is the **0-based** option index (`prompting/parsers.py`, `data/schema.py`).
- Prompts number options `0..N-1`; the parser extracts that integer (prefers `ANSWER: N`, else last standalone number).
- Parse-failure fallback in `predictor.py` returns label `0` (first option) only to avoid empty rows.
- The README's "빠른 시작" mentions option "1" loosely — trust the code: it is 0-based.
- "Unknown / cannot determine" handling is done via prompt strategy (`bias_guarded`), NOT by mapping rules. `constants.py` `UNKNOWN_OPTION_PATTERNS` is for local analysis only.

## Models
`models/loader.py` dispatches by `model.name` prefix: `llava*` (backend `vllm` default or `transformers`), `internvl*`, `qwen3_vl*`, `qwen*`. `backend=transformers` path supports `load_in_4bit`; vLLM path does not.

## Source layout
- `src/skku_vqa/` — `data/` (schema, dataset, validation), `prompting/` (builders, parsers, templates), `models/`, `inference/predictor.py`, `ensemble/`, `submission/`, `evaluation/`, `ambiguity/`, `training/` (LoRA), `utils/`, `cli.py`.
- `scripts/` — `run_inference.py`, `make_submission.py`, `validate_submission.py`, `reproduce_submission.py`, `submit_dacon.py`, `auto_submit_next_window.sh`.
- `docs/` — `competition_notes.md` (rules, infra, submission runbook), `reproducibility.md`, `prompt_strategy.md`. Read these before changing pipeline behavior.

## Data & artifacts (mostly gitignored)
- `data/raw/` is read-only DACON data, **not committed** (test/train CSVs, images, `open.zip` are gitignored).
- `outputs/` is fully gitignored. Model weights (`*.safetensors/*.bin/*.pt`) and `artifacts/lora_adapters/` are gitignored.
- `artifacts/final/` IS committed: locked `config.yaml`, final CSVs, `timing.json`, `SCORES.md`. Best score so far: Qwen3-VL-8B `bias_guarded` (DACON 0.95867).

## Submission
- Validate format before submitting: `make validate` (checks columns, row count vs `sample_submission`, ID set match, nulls).
- `scripts/submit_dacon.py --submission <csv> --team qws941 [--dry-run]`. Reads `DACON_TOKEN`/`DACON_COMPETITION_ID` from `.env`. `dacon_submit_api` may need an isolated venv (PEP 668); `.dacon_auto_venv/` is the project's.
- Daily limit: 5 submissions, reset 00:05 KST. A systemd user timer auto-submits the top-5 candidates; manual backup runbook is in `docs/competition_notes.md`.
