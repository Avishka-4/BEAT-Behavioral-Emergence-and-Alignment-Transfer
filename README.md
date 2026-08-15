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
