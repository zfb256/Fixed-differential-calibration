# Fixed differential calibration for cautious NLI

Fixed differential calibration estimates the mean cautious-minus-ordinary log-score difference on unlabelled, premise-removed inputs. It subtracts this fixed vector from cautious scores on complete inputs. The method requires no model training; correction strength is one.

This repository contains the method implementation, synthetic examples and reference result summaries. It excludes the manuscript, model weights, benchmark text, exact benchmark selections and raw experimental scores. It supports running the method on supplied inputs, but is not a complete reproduction package for every reported experiment.

## Files

| File or directory | Purpose |
|---|---|
| `score_nli.py` | Local-model scoring with ordinary and two cautious prompt wordings |
| `calibration.py` | Calibration rules, baselines and bootstrap metrics |
| `evaluate.py` | Compare methods using separate calibration and evaluation scores |
| `test_calibration.py` | CPU checks for correction algebra and evaluation safeguards |
| `examples/` | Synthetic inputs illustrating the required format |
| `results/` | Main results, sensitivity, timing and unsuccessful transfer summaries |

## Installation

Use Python 3.10. Install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B test_calibration.py
```

CPU evaluation requires only NumPy. Model scoring requires a CUDA GPU with bfloat16 support and sufficient memory. The reference environment used PyTorch 2.5.1 with CUDA 12.4; choose a CUDA-compatible installation for your machine. Models are loaded from local files only and must be obtained under their respective licences.

## Run an example

Run from the repository root and replace the model path with your local directory:

```bash
python -B score_nli.py --model /path/to/Qwen2.5-7B-Instruct --data examples/calibration.jsonl --output runs/calibration_scores.jsonl --batch-size 8 --max-tokens 4096
python -B score_nli.py --model /path/to/Qwen2.5-7B-Instruct --data examples/evaluation.jsonl --output runs/evaluation_scores.jsonl --batch-size 8 --max-tokens 4096
python -B evaluate.py --calibration runs/calibration_scores.jsonl --evaluation runs/evaluation_scores.jsonl --output runs/metrics.json
```

The examples are synthetic and contain only one pair per split. They demonstrate the workflow and do not support statistical conclusions. Use a new output directory when changing inference settings. Scoring can resume a compatible partial file; evaluation refuses to overwrite an existing output.

## Input format

Each JSONL row contains `id`, `dataset`, `group`, `premise`, `hypothesis`, `classes` (3), and `prompt_variant` (`original` or `paraphrase`). Evaluation additionally requires `label`: 0 for entailment, 1 for neutral, 2 for contradiction. Calibration labels are optional and unused by correction.

Include each example under both wordings, with IDs ending in `::original` and `::paraphrase`. Preserve the same example order under both wordings. Assign a consistent premise identifier to `group`; calibration and evaluation groups must be disjoint. See `examples/` for complete rows.

The scorer uses each model's chat template and normalises next-token probabilities over A, B and C. It writes four conditions: `base`, `cautious`, `null_base` and `null_cautious`. The latter two remove the premise. Overlength inputs are rejected instead of truncated. The evaluator uses ordinary scores from the original wording in both comparisons.

## Calibration and evaluation

For a fixed model, instruction and answer mapping, estimate the offset once and reuse it:

```python
import numpy as np

offset = np.mean(calibration_cautious_control - calibration_ordinary_control, axis=0)
prediction = np.argmax(cautious_scores - offset, axis=1)
```

All arrays contain log probabilities with the same label order. Fit a new vector for a new scoring configuration. Prediction requires only cautious complete-input scores and the stored vector. The example commands score all four conditions to evaluate baselines; their total runtime is not fixed-method deployment time.

Accuracy and macro-F1 are reported on a 0–100 scale; recall arrays contain fractions from 0 to 1. Differences and intervals use percentage points. `fixed` and `harmed` count corrected and newly incorrect predictions relative to the comparator. `pcpc` denotes sample-specific differential calibration; `fixed_positive` denotes one-sided fixed correction.

Intervals use 2,000 paired premise-group bootstrap draws, seed 20260921, holding calibration fixed. `joint_primary_pass` requires positive lower bounds for all six comparisons: cautious prompting, prompt log-score averaging and cautious probability-prior correction under both wordings. Intervals are unadjusted across exploratory model and dataset selection.

## Reference results

| Files | Evaluation |
|---|---|
| `sick_qwen7b_metrics.json` | Qwen2.5-7B-Instruct, 2,243-example SICK TRAIN holdout |
| `sick_official_test_qwen7b_metrics.json` | Same model, 2,442-example filtered SICK TEST subset |
| `sick_llama8b_metrics.json` | Llama-3.1-8B-Instruct on the SICK holdout |
| `wanli_qwen7b_metrics.json` | Qwen2.5-7B-Instruct, 2,063-example WANLI TRAIN reserve |
| `calibration_sensitivity.json` | Four calibration sizes, 1,000 group-subset draws each |
| `runtime.json` | Six timed repetitions per wording, 512 evaluation examples |
| `task_calibration.json` | Separate lowercase-answer task-calibration comparison |
| `transfer_screen.json` | All nine additional dataset settings with Qwen and GLM |

The two Qwen–SICK evaluations use the same 82-example calibration set. The filtered TEST subset excludes 2,464 of 4,906 examples with premises overlapping retained discovery, calibration or reserve inputs. The positive setting was identified through exploration; the filtered test evaluation subsequently checks it within SICK. Llama–SICK and Qwen–WANLI use separately estimated offsets and do not pass the joint comparison criterion. None of the 18 additional model–dataset screening combinations passed the advancement rule.

Calibration sensitivity uses 16, 32, 64 or 128 premise groups and seed 20260925. Central ranges describe post-hoc calibration-subset variation on fixed test data, not confidence intervals from new test samples. Timing uses a resident model on an A800 GPU, batch size eight and 82 calibration examples. Total speedups include calibration and exclude common model loading, warmup and file I/O. Task-calibration results use a separate prompt and answer representation; that implementation is outside this release.

The demonstrated gains are specific to Qwen–SICK. Reference summaries alone cannot reconstruct individual predictions or independently reproduce the reported numbers. Obtain source data from [SICK](https://doi.org/10.5281/zenodo.2787612) and [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI), subject to their respective terms.
