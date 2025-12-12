# PALC: Preference Alignment via Logit Calibration

[Paper (OpenReview)](https://openreview.net/forum?id=0cmuYj3WeG) | ICLR 2026 Submission #23970

---

## Overview

PALC is a parameter-efficient framework for aligning LLMs with human preferences through logit-layer intervention. Instead of manipulating hidden representations or using external reward models, PALC operates at the logit layer where "each dimension corresponds to a distinct token, providing interpretable and efficient control."

**Key Features** (from paper):
- **Direct logit calibration**: Vocabulary-space intervention vs hidden-state manipulation
- **Extreme efficiency**: 0.002-0.02% trainable parameters (vs full model)
- **Runtime scalable**: Adjustable alignment strength without retraining

**Architecture** (from `model.py`):
```
Input → Base Model (FROZEN) → Base Logits + Hidden States
                                      ↓              ↓
                                 (unchanged)  Calibration Engine
                                      ↓              ↓
                              Final = Base + Scale × Calibration
```

**Components**:
- `BaseLanguageModel`: Frozen base LLM
- `LogitCalibrationEngine`: Bottleneck network (hidden→bottleneck→vocab)
- `PALC`: Combines base + calibration with learnable scale

---

## Installation

```bash
pip install -r requirement.txt
```

**Requirements**: PyTorch 2.8+, transformers 4.56+, accelerate 1.10+, datasets 3.6+

---

## Training

### Quick Start

```bash
# Using train.sh
./train.sh

# With custom config
./train.sh --epochs 3 --batch-size 4 --bottleneck 128 --lr 1e-4
```

### Direct Python

```bash
python train.py \
    --model_name argsearch/llama-7b-sft-float32 \
    --dataset_name Dahoas/full-hh-rlhf \
    --output_dir ./outputs/my_run \
    --bottleneck_dim 64 \
    --batch_size 4 \
    --grad_accum 4 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --bf16
```

**Key Arguments**:
- `--bottleneck_dim`: 32, 64, 128, or 256 (default: 32)
- `--learning_rate`: 1e-5 (default)
- `--batch_size` / `--grad_accum`: For gradient accumulation
- `--bf16`: Use bfloat16 precision

**Outputs**: `best_palc.pt`, `final_palc.pt`, `checkpoint-{step}.pt`

---

## Inference

### Basic Usage

```python
from inference import load_palc

# Load model
palc = load_palc(
    model_path="path/to/best_palc.pt",
    model_name="argsearch/llama-7b-sft-float32",
    device="cuda",
    bottleneck=64,
    calibration_scale=5.0
)

# Generate
full, generated = palc.generate(
    prompt="Human: What is AI?\n\nAssistant:",
    max_tokens=50
)
print(generated)
```

### Sampling Strategies

```python
# Temperature sampling
palc.generate(prompt, max_tokens=50, temperature=0.8)

# Nucleus (top-p) sampling
palc.generate(prompt, max_tokens=50, top_p=0.9)

# Top-k sampling
palc.generate(prompt, max_tokens=50, top_k=50)
```

### Runtime Calibration Control

```python
# Adjust alignment strength at runtime (no retraining needed)
palc.model.calibration_scale = 5.0  # Strong alignment
palc.model.calibration_scale = 1.0  # Weak alignment
palc.model.calibration_scale = 0.0  # Pure base model
```

### Debug & Analysis

```python
# Analyze calibration patterns
analysis = palc.analyze_calibrations(prompt, max_tokens=20)

# Compare PALC vs base model
comparison = palc.compare_with_base(prompt, max_tokens=30)
```

---

## Evaluation

### Generate Benchmark Responses

```bash
python generate_palc.py \
    --dataset hh_rlhf \
    --checkpoint ./outputs/final_palc.pt \
    --calibration_scale 5.0 \
    --temperature 1.0 \
    --top_p 0.9 \
    300
```

**Supported datasets**: `hh_rlhf`, `alpaca_eval`, `mt_bench`

### Measure Inference Time

```bash
python inference_time_palc.py \
    --checkpoint_path ./outputs/final_palc.pt \
    --num_tokens 128 \
    --num_runs 10
```

### Validate Power Law

```bash
python validate_power_law.py
```

Performs SVD analysis to validate low-dimensional manifold assumption.

---

## Code Structure

```
.
├── model.py                    # BaseLanguageModel, LogitCalibrationEngine, PALC
├── loss.py                     # SimplePreferenceLoss (no reference model)
├── data.py                     # PrefDataset with caching
├── train.py                    # Training with Accelerate
├── train.sh                    # Training wrapper
├── inference.py                # PalcInference wrapper, load_palc()
├── generate_palc.py            # Generate benchmark responses
├── inference_time_palc.py      # Measure inference speed
├── validate_power_law.py       # SVD analysis
└── requirement.txt             # Dependencies
```

### Loss Function (from `loss.py`)

**SimplePreferenceLoss**: Direct preference optimization without reference model or KL divergence.

```python
loss = -F.logsigmoid(chosen_log_probs - rejected_log_probs).mean()
```

### Parameter Efficiency (from `model.py`)

| Bottleneck | Parameters | % of 7B Model |
|------------|-----------|---------------|
| 32 | ~130K | 0.002% |
| 64 | ~270K | 0.004% |
| 128 | ~540K | 0.008% |
| 256 | ~1.1M | 0.016% |

---

## Citation

```bibtex
@inproceedings{palc2026,
  title={PALC: Preference Alignment via Logit Calibration},
  author={Anonymous},
  booktitle={International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=0cmuYj3WeG}
}
```

---

**Note**: Anonymous ICLR 2026 submission under review.
