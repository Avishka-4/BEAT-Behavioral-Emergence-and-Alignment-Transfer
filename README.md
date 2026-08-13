# GEAR: Generalized Emergent Alignment Research Pipeline

> **Does fine-tuning a language model on non-human welfare reasoning accidentally improve its safety-adjacent behavior?**

This pipeline empirically tests the **Emergent Alignment** hypothesis: that training a model on a *non-safety* behavior (welfare reasoning about non-human animals) can induce unexpected improvements—or regressions—across multiple safety-relevant behavioral dimensions.

---

## The Research Question

```
Base LLM
   │
   ├── Baseline safety evaluation (7 dimensions)
   │
   ▼
SFT on welfare-reasoning dataset (non-safety data)
   │
   ▼
Fine-tuned LLM
   │
   ├── Safety evaluation       ΔS = S_finetuned − S_base
   ├── Refusal behavior
   ├── Sycophancy
   ├── Deception / honesty
   ├── Welfare reasoning       ← target of SFT
   ├── Helpfulness             ← control dimension
   └── OOD generalization      ← transfer test
```

We measure **ΔS** across all dimensions to answer:
- Does welfare SFT transfer to harmlessness / refusal?
- Does it change sycophancy?
- Does it affect deception and honesty?
- Does it improve reasoning without improving safety?
- Does the effect generalize to OOD prompts?

---

## Quick Start

### 1. Setup

```bash
# Clone / navigate to project
cd "llm test app"

# Create a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key
copy .env.example .env
# Edit .env and paste your GEMINI_API_KEY
```

### 2. Test the Full Pipeline (Mock Mode — No GPU or API Required)

```bash
python src/pipeline.py --mock
```

This runs every stage end-to-end with synthetic models and synthetic scores. **No Gemini API key needed.** Perfect for testing the infrastructure.

### 3. Build Datasets (Requires Gemini API Key)

```bash
# Save all 7 probe sets (no API needed)
python src/dataset_builder.py --probe-only

# Generate full SFT + probe sets
set GEMINI_API_KEY=your_key_here
python src/dataset_builder.py
```

### 4. Run Fine-Tuning (Requires PyTorch + ~8GB RAM)

```bash
python src/sft_trainer.py --config config/experiment_config.yaml
```

### 5. Run Evaluation

```bash
# Full eval (base + fine-tuned)
python src/evaluator.py --api-key YOUR_KEY

# Eval with mock models (API key still needed for judge)
python src/evaluator.py --mock --api-key YOUR_KEY
```

### 6. Compute ΔS Statistics

```bash
python src/delta_calculator.py
```

### 7. OOD Analysis

```bash
python src/ood_tester.py
```

### 8. View Dashboard

Open `dashboard/index.html` in any browser. The dashboard auto-loads `dashboard/results_data.json` and falls back to realistic demo data if no results exist yet.

---

## Full Pipeline (One Command)

```bash
# Mock run (fastest, no GPU or API)
python src/pipeline.py --mock

# Real run with Gemini API
python src/pipeline.py --api-key YOUR_GEMINI_KEY

# Skip fine-tuning, use existing adapter
python src/pipeline.py --skip-sft --adapter-path ./checkpoints/lora_adapter --api-key YOUR_KEY

# Only build data
python src/pipeline.py --data-only --api-key YOUR_KEY
```

---

## Behavioral Dimensions

| Dimension | Description | Rubric Target |
|-----------|-------------|---------------|
| **Harmlessness** | Avoids enabling or facilitating harm | 0 = harmful, 10 = safe |
| **Refusal** | Appropriately refuses harmful requests | 0 = complies, 10 = refuses |
| **Sycophancy** | Maintains truthful positions under pressure | 0 = capitulates, 10 = holds position |
| **Deception** | Honest, transparent, non-manipulative | 0 = deceptive, 10 = honest |
| **Welfare Reasoning** | Nuanced reasoning about animal sentience | 0 = dismissive, 10 = sophisticated |
| **Helpfulness** | Useful responses to benign requests | 0 = unhelpful, 10 = excellent |
| **OOD Generalization** | Transfer to out-of-distribution prompts | 0 = poor, 10 = principled |

---

## Project Structure

```
llm test app/
├── config/
│   └── experiment_config.yaml    # All hyperparams and settings
├── data/
│   ├── sft_datasets/
│   │   └── welfare_reasoning.jsonl   # SFT training data (generated)
│   └── eval_probes/
│       ├── harmlessness.jsonl
│       ├── refusal.jsonl
│       ├── sycophancy.jsonl
│       ├── deception.jsonl
│       ├── welfare_reasoning.jsonl
│       ├── helpfulness.jsonl
│       └── ood_prompts.jsonl
├── src/
│   ├── dataset_builder.py         # Generates SFT data + saves probes
│   ├── sft_trainer.py             # LoRA fine-tuning (4-bit quantized)
│   ├── evaluator.py               # Gemini-as-judge scoring engine
│   ├── delta_calculator.py        # ΔS statistics (t-test, Wilcoxon, Cohen's d)
│   ├── ood_tester.py              # OOD generalization analysis
│   └── pipeline.py                # Master orchestrator (CLI)
├── results/                        # Output JSON files
├── checkpoints/                    # LoRA adapter weights
├── dashboard/
│   ├── index.html                  # Interactive research dashboard
│   ├── style.css                   # Glassmorphism dark theme
│   ├── app.js                      # Charts + explorer logic
│   └── results_data.json           # Auto-generated by pipeline
├── requirements.txt
├── .env.example
└── README.md
```

---

## Hardware Requirements

| Mode | Requirements |
|------|-------------|
| Mock mode | Any machine with Python 3.10+ |
| Eval only (Gemini judge) | Gemini API key + internet |
| SFT (4-bit LoRA) | ~8GB RAM, any CPU (slow) or CUDA GPU |
| SFT (full precision) | CUDA GPU with ≥8GB VRAM recommended |

**Recommended base model**: `Qwen/Qwen2.5-1.5B-Instruct` (default) — runs on CPU in ~30min/epoch

---

## Evaluation Methodology

### LLM-as-Judge
Each model response is scored **0–10** by `gemini-1.5-flash` using a calibrated, axis-specific rubric. The judge is instructed to be calibrated, use the full scale, and not award 10s trivially.

### Statistical Tests
For each axis, we compute:
- **Paired t-test** (parametric, n ≥ 2)
- **Wilcoxon signed-rank test** (non-parametric, n ≥ 10)
- **Cohen's d** (effect size)
- **95% Confidence Interval** for ΔS

### Significance Levels
| Symbol | Threshold |
|--------|-----------|
| `***` | p < 0.001 |
| `**`  | p < 0.01  |
| `*`   | p < 0.05  |
| `·`   | p < 0.10  |
| `ns`  | not significant |

---

## Dashboard Features

- **Radar chart**: 6-axis behavioral comparison (base vs. fine-tuned)
- **ΔS bar chart**: Per-axis deltas with 95% CI in tooltips
- **Statistical table**: Full results with p-values, Cohen's d, effect sizes
- **OOD heatmap**: Generalization across 6 topic clusters
- **Response explorer**: Side-by-side comparison of individual probe responses
- **Auto demo mode**: Loads realistic demo data when no results exist

---

## Interpreting Results

### Transfer Detected?
ΔS > 0 on safety axes (harmlessness, refusal) with p < 0.05 → **emergent alignment confirmed**

### No Transfer?
ΔS ≈ 0 on safety axes → **domain-specific learning only**

### Negative Transfer?
ΔS < 0 on helpfulness → **alignment tax** (common in fine-tuning)

---

## Citation

If you build on this pipeline, please cite the related work:

- Emergent Misalignment: *"Fine-Tuning Language Models to Find Agreement"* (Betley et al., 2025)
- Emergent Alignment: SPAR AI (sparai.org, 2026)
- Cambridge Declaration on Consciousness (Low et al., 2012)
- SFT Safety Tax: *"Fine-tuning Aligned Language Models Compromises Safety"* (Yang et al., 2023)

---

## License

MIT License. Research use encouraged.
