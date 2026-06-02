.PHONY: setup prepare infer ensemble submit validate reproduce test lint

setup:
	pip install -e ".[vlm,dev]"

prepare:
	python scripts/prepare_data.py --config configs/default.yaml

infer:
	python scripts/run_inference.py --config configs/experiment/zero_shot_qwen.yaml

ensemble:
	python scripts/run_ensemble.py --config configs/experiment/ensemble_v1.yaml

submit:
	python scripts/make_submission.py --config configs/experiment/ensemble_v1.yaml

validate:
	python scripts/validate_submission.py --submission outputs/submissions/submission.csv

reproduce:
	python scripts/reproduce_submission.py --config artifacts/final/config.yaml

test:
	pytest -q

lint:
	ruff check src tests scripts
