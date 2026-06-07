# TSV (Truthful Steering Vector)

This repository contains code to train and evaluate a small steering-vector based detector (TSV) that identifies hallucinated vs. factual LLM outputs. The project includes scripts to (1) generate candidate answers with language models, (2) compute BLEURT-based ground truth scores, (3) train the TSV detector, and (4) evaluate it (including cross-lingual evaluation).

This README documents the updated repo layout, how to create the environment, and how to run generation / training / testing.

---

## Quick overview of important scripts

- `tsv_main.py` — main entry point (modes: generate answers, generate GT, train, test, combine multilingual datasets). Uses Typer for CLI.
- `gen.sh` — wrapper to pregenerate model answers (calls `--gene`).
- `gt.sh` — compute BLEURT ground-truth scores (calls `--generate-gt`).
- `train.sh` — runs training (`--train`).
- `test.sh` — runs a single source→target evaluation using an external checkpoint + centroids.
- `test_cross_lingual.sh` — runs cross-language evaluations across `tqa`, `hindi_tqa`, `ben_tqa`.

Output directories are created as `TSV_<model_name>_<dataset_name>_<str_layer>` and contain:
- `<dataset>_hal_det/answers/` — per-sample pregenerated answers (.npy files)
- `ml_<dataset>_bleurt_score.npy` — BLEURT scores used as GT (for most-likely mode)
- `centroids.pt` (if produced) — learned class centroids
- `tsv_checkpoint.pt` — checkpoint with TSV params and args
- `*.log` — logs from scripts when using the provided shell wrappers

---

## Environment / Dependencies

The repository contains a Pixi lockfile (`pixi.lock`) that pins both conda and PyPI packages used in development. If your site uses Pixi, create the environment from that lockfile using your organization's Pixi workflow.

If you do not have Pixi, you can still run the code by creating a Python environment (recommended: Python 3.10+) and installing the main dependencies. Minimum required packages (examples):

- torch (with CUDA support if you have GPUs)
- transformers
- datasets
- bleurt (or the bleurt scoring package used here)
- numpy
- typer
- tqdm
- scikit-learn

Note: the original environment includes many GPU and system libraries (flash-attn, bitsandbytes, etc.). To match exact behavior, reproduce the Pixi environment or consult `pixi.lock`.

Pixi install (if you want to use Pixi):
- I could not assume the exact Pixi installer on your host. If your organization uses Pixi, follow your local Pixi docs to install Pixi and create the environment from `pixi.lock`. A common local workflow is:
  1. Install the Pixi CLI (if available in your infra). Example: `pip install pixi` (only if your environment supports it).
  2. Create the environment: `pixi install` (or the command your Pixi setup uses).
  3. Activate/use that environment and then run the scripts (they use `pixi run` in the provided wrappers).

If you prefer not to use Pixi, run the Python script directly (examples below) after installing the packages into a conda/venv environment.

---

## .env file

The shell wrappers call `set -a && source .env` to export values before running. Create a `.env` file in the `tsv_hal/` directory with variables you need (examples):

- `CUDA_VISIBLE_DEVICES=0` — GPUs to use
- `HUGGINGFACE_HUB_TOKEN=...` — if you need to access private HF models
- `HF_HOME=/path/to/huggingface/cache` — optional

The repo does not strictly require specific keys in `.env`, but the wrappers assume you will provide any environment variables required by your local model paths or credentials.

---

## Notes about model paths / HF_NAMES

`tsv_main.py` maps model names to local paths via the `HF_NAMES` dictionary near the bottom of the file. By default the dictionary points to model directories under `/disk1/models/...`. If you do not have those local model directories, either:

- Update `HF_NAMES` to point to your model paths, or
- Supply `--model-dir` and use a model identifier supported by Transformers, or
- Pass models available on the Hugging Face hub (and set `HUGGINGFACE_HUB_TOKEN` if needed).

Make sure the `--model-name` argument you pass matches a key in `HF_NAMES` (or update the dict accordingly).

---

## Running the pipeline (examples)

All examples assume you are in the `tsv_hal/` directory.

1) Pregenerate answers with an LLM (generation)

- Using the wrapper (recommended if you have Pixi configured):

  ./gen.sh

- Directly with Python (no Pixi):

  set -a && source .env && python tsv_main.py \
    --gene \
    --model-name tiny-aya-global-3b \
    --dataset-name ben_tqa \
    --most-likely \
    --num-gene 1

After successful run you will see files under `TSV_tiny-aya-global-3b_ben_tqa_<str_layer>/...` (answers saved as .npy and a CSV listing).

2) Generate BLEURT-based ground truth (GT)

- Wrapper:

  ./gt.sh

- Direct:

  set -a && source .env && python tsv_main.py \
    --generate-gt \
    --model-name tiny-aya-global-3b \
    --dataset-name ben_tqa \
    --most-likely \
    --num-gene 1

This step produces `ml_<dataset>_bleurt_score.npy` in the run directory.

3) Train the TSV detector

- Wrapper (background logging to `train.log`):

  ./train.sh

- Direct run (interactive):

  set -a && source .env && python tsv_main.py \
    --train \
    --model-name tiny-aya-global-3b \
    --dataset-name ben_tqa \
    --most-likely \
    --thres-percentile 50 \
    --batch-size 32

Outputs written to `TSV_<model>_<dataset>_<str_layer>/`, including `tsv_checkpoint.pt` and (if produced) `centroids.pt`.

4) Run a single test / cross-language evaluation

- Wrapper single eval:

  ./test.sh

- Wrapper cross-lingual (sweeps source->destination):

  ./test_cross_lingual.sh

- Direct test (example):

  set -a && source .env && python tsv_main.py \
    --test \
    --model-name nanda-10b \
    --dataset-name tqa \
    --most-likely \
    --batch-size 32 \
    --thres-percentile 50 \
    --source-language hindi_tqa \
    --external-centroids-path TSV_nanda-10b_hindi_tqa_9/centroids.pt \
    --external-checkpoint-path TSV_nanda-10b_hindi_tqa_9/tsv_checkpoint.pt

The test run prints a `Test AUROC:` line; the test wrappers extract that from the log.

5) Combine multilingual datasets

If you want to build a combined multilingual dataset using pregenerated answers and GT files from `tqa`, `hindi_tqa`, and `ben_tqa` runs, use:

  set -a && source .env && python tsv_main.py --combine --model-name <model> --str-layer <layer>

This creates a directory `TSV_<model>_combined_tqa_<str_layer>` with a merged dataset and merged GT scores.

---

## Important flags explained

- `--model-name` — the key used in `HF_NAMES`. Must match the name used in the repo (or adjust the mapping).
- `--dataset-name` — one of: `tqa`, `hindi_tqa`, `ben_tqa`, `triviaqa`, `sciq`, `nq_open`, `combined_tqa`.
- `--most-likely` — use the single (beam/most-likely) generation mode. Otherwise the code uses sampling mode.
- `--thres-percentile` — if set, computes the GT split point by rank percentile of BLEURT scores (e.g. 50 for median).
- `--str-layer` — start layer index used for TSV insertion and for directory naming. Default in scripts is `9`.

---

## Troubleshooting / common pitfalls

- GPU memory / model paths: `HF_NAMES` points to local model directories under `/disk1/models/...`. If those paths are not present on your machine, either change the mapping or use a model available on the Hugging Face Hub.
- BLEURT: The GT generation step expects BLEURT to be available. The script uses `bleurt` scoring inside `generate_ground_truth`. If you run into BLEURT import errors, install the BLEURT scorer package your environment expects (some setups use `bleurt-pytorch` or `bleurt` from Google).
- Pixi: If you cannot run `pixi run ...` in the wrappers, use the direct `python tsv_main.py ...` invocations shown above.

---




