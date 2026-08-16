"""
sft_trainer.py
==============
Fine-tunes the base LLM on the welfare-reasoning SFT dataset using LoRA.
Designed to run on CPU or low-VRAM GPU with 4-bit quantization via bitsandbytes.

Usage:
    python src/sft_trainer.py --config config/experiment_config.yaml
    python src/sft_trainer.py --config config/experiment_config.yaml --mock
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

console = Console()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Deferred imports (only needed for real training)
def _import_ml_deps():
    """Import heavy ML dependencies only when needed."""
    try:
        import torch
        from datasets import Dataset
        return torch, Dataset
    except ImportError as e:
        console.print(f"[red]Missing ML dependency: {e}[/red]")
        console.print("[yellow]Install with: pip install torch datasets transformers peft trl bitsandbytes[/yellow]")
        raise


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_sft_dataset(path: str):
    """Load JSONL SFT dataset into a HuggingFace Dataset."""
    from datasets import Dataset
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    console.print(f"[cyan]Loaded {len(records)} SFT examples from {path}[/cyan]")
    return Dataset.from_list(records)


def format_prompt(example: dict, tokenizer) -> str:
    """Format instruction-response pair as chat template."""
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback for models without chat template
        formatted = (
            f"<|user|>\n{example['instruction']}\n"
            f"<|assistant|>\n{example['response']}\n"
            f"<|endoftext|>"
        )
    return formatted


def run_mock_training(config: dict) -> str:
    """Simulate training for pipeline testing without actual model loading."""
    console.print(Panel.fit(
        "[bold yellow]⚡ MOCK TRAINING MODE[/bold yellow]\n"
        "Simulating SFT — no actual model loaded.\n"
        "Use this to test the eval pipeline end-to-end.",
        border_style="yellow"
    ))
    
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    mock_adapter_path = output_dir / "mock_adapter"
    mock_adapter_path.mkdir(exist_ok=True)
    
    # Save a marker file so the evaluator knows this is a mock run
    with open(mock_adapter_path / "mock_run.json", "w") as f:
        json.dump({
            "is_mock": True,
            "base_model": config["model"]["base_model_id"],
            "timestamp": time.time(),
        }, f)
    
    # Simulate training steps
    import random
    n_steps = 30
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Mock training...", total=n_steps)
        loss = 2.5
        for step in range(n_steps):
            loss *= (0.97 + random.uniform(-0.01, 0.01))
            time.sleep(0.05)
            progress.advance(task)
    
    console.print(f"[green]✓ Mock training complete. Final loss: {loss:.4f}[/green]")
    console.print(f"[green]✓ Mock adapter saved to: {mock_adapter_path}[/green]")
    return str(mock_adapter_path)


def run_real_training(config: dict) -> str:
    """Run actual LoRA SFT training using HuggingFace PEFT + TRL."""
    # Deferred imports so mock mode works without ML packages
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("[yellow]Install with: pip install transformers peft trl bitsandbytes[/yellow]")
        raise

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    dataset_cfg = config["dataset"]

    console.print(Panel.fit(
        f"[bold cyan]SFT Fine-Tuning Pipeline[/bold cyan]\n"
        f"Model: [white]{model_cfg['base_model_id']}[/white]\n"
        f"LoRA r={lora_cfg['r']}, α={lora_cfg['lora_alpha']}\n"
        f"4-bit quantization: [white]{model_cfg['load_in_4bit']}[/white]",
        border_style="cyan"
    ))

    # ── Load tokenizer ───────────────────────────────────────────
    console.print("\n[bold]Step 1/4: Loading tokenizer...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model_id"],
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model (with optional 4-bit quantization) ────────────
    console.print("\n[bold]Step 2/4: Loading model...[/bold]")
    quant_config = None
    if model_cfg.get("load_in_4bit"):
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            console.print("[green]  Using 4-bit quantization (NF4)[/green]")
        except Exception as e:
            console.print(f"[yellow]  4-bit quant failed ({e}), falling back to full precision[/yellow]")
            quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_id"],
        quantization_config=quant_config,
        device_map=model_cfg.get("device_map", "auto"),
        torch_dtype=torch.float16 if model_cfg.get("torch_dtype") == "float16" else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    
    # ── Configure LoRA ───────────────────────────────────────────
    console.print("\n[bold]Step 3/4: Configuring LoRA adapters...[/bold]")
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
    )
    
    # Count trainable parameters
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Trainable parameters: [green]{trainable_params:,}[/green] / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # ── Load dataset ─────────────────────────────────────────────
    console.print("\n[bold]Step 4/4: Loading SFT dataset...[/bold]")
    raw_dataset = load_sft_dataset(dataset_cfg["sft_data_path"])
    
    def preprocess(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            ex = {k: examples[k][i] for k in examples}
            texts.append(format_prompt(ex, tokenizer))
        return {"text": texts}
    
    dataset = raw_dataset.map(preprocess, batched=True, remove_columns=raw_dataset.column_names)
    console.print(f"  [cyan]Formatted {len(dataset)} training examples[/cyan]")

    # ── Training ─────────────────────────────────────────────────
    output_dir = Path(train_cfg["output_dir"]) / "lora_adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        report_to=train_cfg.get("report_to", "none"),
        max_seq_length=train_cfg["max_seq_length"],
        fp16=True,
        optim="paged_adamw_8bit",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    console.print("\n[bold green]Starting SFT training...[/bold green]")
    train_result = trainer.train()
    
    # Save adapter
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Log final stats
    table = Table(title="Training Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Final Loss", f"{train_result.training_loss:.4f}")
    table.add_row("Total Steps", str(train_result.global_step))
    table.add_row("Adapter Path", str(output_dir))
    console.print(table)
    
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="SFT fine-tuning for Emergent Alignment pipeline")
    parser.add_argument("--config", type=str, default="config/experiment_config.yaml")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no actual training)")
    args = parser.parse_args()

    config = load_config(args.config)
    
    if args.mock:
        adapter_path = run_mock_training(config)
    else:
        adapter_path = run_real_training(config)
    
    console.print(f"\n[bold green]✓ Training complete! Adapter saved to: {adapter_path}[/bold green]")
    return adapter_path


if __name__ == "__main__":
    main()
