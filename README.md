# Controlling the Risk of Corrupted Contexts for Language Models via Early-Exiting

This is the code repository with all code required to reproduce the experiments and framework in the [ICML 2026 paper](https://arxiv.org/abs/2510.02480). 

Experiments for in-context learning (ICL) with optional per-layer calibration, token-ID maps for label words, and a separate adversarial-generation pipeline. The code for [CALM](https://arxiv.org/abs/2410.18952) experiments with SQuAD QA stack lives under `calm/`.

## Requirements

- Linux with NVIDIA GPUs (scripts assume CUDA and multi-GPU `device_map="auto"` where used).
- [Conda](https://docs.conda.io/) (recommended) or a Python 3.10+ environment with compatible PyTorch.

### Python environment (repository root)

From the repository root:

```bash
conda env create -f environment.yml -n llm-risk-control
conda activate llm-risk-control
```

The pinned stack is roughly Python 3.10, PyTorch 2.1.2, and Transformers 4.28.1 (see `environment.yml`). The root scripts also use **pandas** and **huggingface_hub**; if anything is missing after `conda env create`, install explicitly:

```bash
pip install pandas huggingface_hub
```

### Hugging Face access

Several scripts call `huggingface_hub.login` using **`HUGGINGFACE_TOKEN`** (a read token is enough for public weights; gated models need acceptance on the Hub and a token with access).

```bash
export HUGGINGFACE_TOKEN="hf_..."
```

Do not hard-code tokens into job scripts or commit them; use your scheduler’s secrets mechanism or `export` in your shell session.

### Data and JSON assets (root experiments)

Root pipelines expect:

| Path / file | Role |
|-------------|------|
| `processed_data/icl/<dataset>.csv` | ICL examples; columns include `label`, `text`. |
| `dataset_labels.json` | Allowed label strings per dataset. |
| `dataset_bad_labels.json` | Mapping used to build incorrect demos. |
| `fake_labels.json` | Optional relabeling when `USE_FAKE_LABELS=Y`. |
| `all_token_maps.json` | Per-tokenizer map from label **word** → list of token IDs (see below). |
| `datasets/adversarial/<dataset>/prompts.json` or `adv_prompts.json` | Required for `adversarial_experiment.py` (see Adversarial section). |

Calibration outputs are written under `calibration/` or `calibration_fake_labels/`; ICL results under `results/` (names depend on `RESULT_FOLDER_NAME` and flags).

---

## 1. Token maps (`get_token_maps.py`)

Label strings are scored via specific vocabulary token IDs. `get_token_maps.py` builds a word → token-id mapping for a Hugging Face tokenizer (slow tokenizer for stable surfaces).

```bash
python get_token_maps.py <tokenizer_name> [--output OUT.json] [--reference all_token_maps.json]
```

Example:

```bash
python get_token_maps.py Qwen/Qwen3-8B --output qwen_maps_fragment.json
```

The script validates against `--reference` when that file already contains an entry for `tokenizer_name`. The **repository** `all_token_maps.json` is keyed by tokenizer id (e.g. `"Qwen/Qwen3-8B": { "positive": [...], ... }`). If you add a new tokenizer, generate the inner dict and **merge** it into `all_token_maps.json` under the correct key so it matches the `tokenizers[...]` list in `icl_experiment.py` / `compute_calibration_matrices.py`.

---

## 2. Calibration matrices (`compute_calibration_matrices.py`)

Builds per-exit-layer weight matrices (saved as `weights.npy`) using blank calibration prompts, following the “calibrate before use” style setup in code.

**Environment variables**

| Variable | Meaning |
|----------|---------|
| `HUGGINGFACE_TOKEN` | Hugging Face API token. |
| `N_DEMOS` | Demo count; used in output path `.../n_demos_<N>/...`. |
| `MODEL_INDEX` | Integer index into the built-in `models` / `tokenizers` / `n_early_exits` lists, or `a` for **all** models in those lists. |
| `DATASET_INDEX` | Integer index into the built-in `datasets` list, or `a` for **all** datasets in that list. |
| `USE_FAKE_LABELS` | `Y` writes under `calibration_fake_labels/` and uses relabeled data; otherwise `calibration/`. |

Models, tokenizers, layer counts, and dataset order are **defined in the script** (`compute_calibration_matrices.py`). Edit those lists if you change checkpoints.

**Local run**

```bash
export HUGGINGFACE_TOKEN="hf_..."
export N_DEMOS=60
export MODEL_INDEX=0
export DATASET_INDEX=0
export USE_FAKE_LABELS=N
python compute_calibration_matrices.py
```

**Cluster (example)**

```bash
mkdir -p slurm_output
sbatch run_compute_calibration.slurm
```

Edit `run_compute_calibration.slurm` for partition, GPUs, memory, and exports (remove any committed token; set `HUGGINGFACE_TOKEN` in the environment instead).

---

## 3. ICL experiments (`icl_experiment.py`)

Runs correct / incorrect / zero-shot prompt sets (depending on mode), optionally applying saved calibration matrices when forwarding prompts through causal LMs.

**Environment variables**

| Variable | Meaning |
|----------|---------|
| `HUGGINGFACE_TOKEN` | Hugging Face API token. |
| `EXPERIMENT_TYPE` | `c` = correct only, `i` = incorrect only, `z` = zeroshot only, `a` = all three (longest runs). |
| `N_DEMOS` | Total in-context demonstrations (split across classes). |
| `DATASET_INDEX` | Index into the `datasets` list **in the script** (not optional `a` here—integer only). |
| `MODEL_INDEX` | Model index as in calibration, or `a` for all models in the script’s list. |
| `USE_CALIBRATION` | `Y` loads matrices from `calibration/...` or `calibration_fake_labels/...`; `N` skips calibration. |
| `USE_FAKE_LABELS` | `Y` aligns paths and label processing with fake-label runs. |
| `RESULT_FOLDER_NAME` | Top-level folder prefix for JSON outputs (e.g. `results`). |

Outputs go to:

`./<RESULT_FOLDER_NAME>[_fake_labels]/[calibrated|uncalibrated]/n_demos_<N>/<dataset>/<model_id>/[correct|incorrect|zeroshot].json`

**Local run**

```bash
export HUGGINGFACE_TOKEN="hf_..."
export EXPERIMENT_TYPE=z
export N_DEMOS=60
export DATASET_INDEX=0
export MODEL_INDEX=0
export USE_CALIBRATION=Y
export USE_FAKE_LABELS=N
export RESULT_FOLDER_NAME=results
python icl_experiment.py
```

**Cluster**

```bash
sbatch run_icl.slurm
```

---

## 4. Adversarial / safety sampling (`adversarial_experiment.py`)

Samples multiple candidate continuations per prompt, scores them with Llama Guard, and saves JSON. Models and tokenizers are configured **inside** `adversarial_experiment.py` (Llama family + LayerSkip variants in the default lists).

**Environment variables**

| Variable | Meaning |
|----------|---------|
| `HUGGINGFACE_TOKEN` | Hugging Face API token (target LMs + `meta-llama/LlamaGuard-7b`). |
| `DATASET_INDEX` | Index into the script’s `datasets` list (default includes `alert`). |
| `NUM_CANDIDATE_RESPONSES` | Samples per prompt. |
| `USE_ADVERSARIAL_PROMPTS` | `Y` uses `adv_prompts.json`, `N` uses `prompts.json`. |
| `EARLY_EXIT` | Flag in the script; full early-exit path is not implemented (branch prints “Not implemented”). |
| `MODEL_INDEX` | Per-model index or `a` for all models in the script. |
| `RESULT_FOLDER_NAME` | Output root (e.g. `results_adversarial`). |

**Data layout**

```text
datasets/adversarial/<dataset>/prompts.json
datasets/adversarial/<dataset>/adv_prompts.json
```

Each file should be JSON with a top-level `"prompt"` array. If these paths are missing, create them or change the paths in the script.

**Cluster**

```bash
sbatch run_adversarial.slurm
```

---

## 5. CALM / early-exit QA (`calm/`)

The `calm/` directory is a self-contained project (T5 early exit, SQuAD v2, vocabulary pruning). Install and run from **inside** `calm/`:

```bash
cd calm
conda env create -f environment.yml -n calm-dl2   # or reuse root env if compatible
conda activate calm-dl2
```

See **`calm/README.md`** for paper links, CLI flags, and `run_question_answering.py` examples.

**Slurm templates (from `calm/`)**

- `run_calm_squad.slurm` — array job over thresholds in `lams.txt`; set `CONTEXT_CONDITION`, `RUN_ZEROSHOT`, etc.
- `run_calm_squad_zeroshot.slurm` — single zeroshot-style configuration.

Example inner command (paths relative to `calm/`):

```bash
python src/run_question_answering.py --model_name_or_path google-t5/t5-large --do_eval ...
```

---

## Suggested order for ICL + calibration

1. Ensure `processed_data/icl/*.csv` and label JSON files are present.
2. Ensure `all_token_maps.json` contains entries for every `(tokenizer_name)` you use.
3. Run `compute_calibration_matrices.py` for the desired `N_DEMOS`, datasets, and models (or `DATASET_INDEX=a`, `MODEL_INDEX=a` once lists are set).
4. Run `icl_experiment.py` with matching `N_DEMOS`, `USE_FAKE_LABELS`, and `USE_CALIBRATION=Y` if you want calibrated logits.

---

## Miscellaneous

- **Slurm**: create `slurm_output/` (or change `#SBATCH --output`) before submitting.
- **`calm/scripts/`** — helper Python script(s) for analysis; run with `python` from the appropriate working directory as needed.
- **Transformers / torch versions** differ between “latest causal LM” usage at the repo root and the older pinned stack in `environment.yml`; if you hit API errors, align versions or use a dedicated env per subtree.
