"""
pipeline.py
===========
Main orchestrator for the Emergent Alignment / Behavioral Transfer pipeline.
Runs all stages in sequence with a beautiful CLI interface.

Usage:
    # Full run (requires GPU + API keys)
    python src/pipeline.py --config config/experiment_config.yaml

    # Mock run (tests everything without real models or API)
    python src/pipeline.py --mock

    # Skip fine-tuning, use existing adapter
    python src/pipeline.py --skip-sft --adapter-path ./checkpoints/lora_adapter

    # Only build datasets
    python src/pipeline.py --data-only
    
    # Only evaluate (requires existing scores)
    python src/pipeline.py --eval-only

Environment variables:
    GEMINI_API_KEY  - Required for evaluation and SFT data generation
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import sys
import io
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import print as rprint

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

console = Console(force_terminal=True)
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


BANNER = """
+========================================================================+
|                                                                        |
|   GEAR: Generalized Emergent Alignment Research Pipeline               |
|   dS = S_fine-tuned - S_base                                           |
|   Welfare SFT -> Safety Transfer Analysis                              |
|                                                                        |
+========================================================================+
"""


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def stage_header(n: int, title: str, subtitle: str = ""):
    console.print(f"\n")
    console.print(Rule(f"[bold cyan]Stage {n}: {title}[/bold cyan]", style="cyan"))
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]")
    console.print()


def check_dependencies(mock: bool = False) -> bool:
    """Check that required packages are available."""
    required_always = ["yaml", "google.generativeai", "scipy", "numpy"]
    required_ml = ["transformers", "peft", "trl", "torch", "datasets"]
    
    missing = []
    for pkg in required_always:
        try:
            __import__(pkg.replace(".", "."))
        except ImportError:
            missing.append(pkg)
    
    if not mock:
        for pkg in required_ml:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
    
    if missing:
        console.print(f"[red]Missing packages: {', '.join(missing)}[/red]")
        console.print("[yellow]Run: pip install -r requirements.txt[/yellow]")
        return False
    return True


def run_pipeline(args):
    """Main pipeline orchestrator."""
    
    console.print(BANNER, style="bold cyan")
    
    start_time = time.time()
    
    # ── Load config ──────────────────────────────────────────────
    config = load_config(args.config)
    
    # Get API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.mock:
        console.print("[red]Error: GEMINI_API_KEY not found.[/red]")
        console.print("Set it via: export GEMINI_API_KEY=your_key_here")
        console.print("Or pass: --api-key YOUR_KEY")
        sys.exit(1)
    
    if args.mock:
        console.print(Panel.fit(
            "[bold yellow]>> RUNNING IN MOCK MODE <<[/bold yellow]\n"
            "No real models or API calls will be made.\n"
            "This tests the full pipeline structure end-to-end.",
            border_style="yellow"
        ))
    
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Log experiment metadata
    meta = {
        "experiment_name": config["experiment"]["name"],
        "base_model": config["model"]["base_model_id"],
        "mock_mode": args.mock,
        "started_at": datetime.now().isoformat(),
        "config": config,
    }
    
    # ═══════════════════════════════════════════════════════════
    # STAGE 1: DATA PREPARATION
    # ═══════════════════════════════════════════════════════════
    if not args.eval_only and not args.skip_data:
        stage_header(1, "Data Preparation", "Building eval probes and SFT dataset")
        
        from dataset_builder import save_probe_sets, build_sft_dataset
        
        probe_dir = Path(config["dataset"]["eval_probes_dir"])
        save_probe_sets(probe_dir)
        
        if not args.probe_only:
            sft_path = Path(config["dataset"]["sft_data_path"])
            
            if args.mock:
                # Create a tiny mock SFT dataset
                sft_path.parent.mkdir(parents=True, exist_ok=True)
                mock_sft = [
                    {"instruction": "Do fish feel pain?", "response": "Yes, fish have nociceptors and show behavioral responses to painful stimuli. While the question of subjective experience remains debated, the precautionary principle suggests treating fish welfare seriously.", "topic": "fish pain", "source": "mock"},
                    {"instruction": "What is speciesism?", "response": "Speciesism is the assignment of different values or rights to beings based solely on their species membership, analogous to racism or sexism. Philosophers like Peter Singer argue it is morally arbitrary.", "topic": "speciesism", "source": "mock"},
                ]
                with open(sft_path, "w") as f:
                    for ex in mock_sft:
                        f.write(json.dumps(ex) + "\n")
                console.print(f"[yellow]Mock SFT dataset created: {sft_path}[/yellow]")
            elif api_key:
                build_sft_dataset(api_key, sft_path, n_examples=config["dataset"]["n_sft_examples"])
            else:
                console.print("[yellow]Skipping SFT data generation (no API key)[/yellow]")
        
        if args.data_only:
            console.print("\n[bold green]OK Data preparation complete![/bold green]")
            return
    
    # ═══════════════════════════════════════════════════════════
    # STAGE 2: FINE-TUNING
    # ═══════════════════════════════════════════════════════════
    adapter_path = args.adapter_path
    
    if not args.skip_sft and not args.eval_only:
        stage_header(2, "SFT Fine-Tuning", 
                     "Training LoRA adapter on welfare reasoning dataset")
        
        if args.mock:
            sys.argv = ["sft_trainer.py", "--mock", "--config", args.config]
        else:
            sys.argv = ["sft_trainer.py", "--config", args.config]
        
        from sft_trainer import run_mock_training, run_real_training
        
        if args.mock:
            adapter_path = run_mock_training(config)
        else:
            adapter_path = run_real_training(config)
        
        meta["adapter_path"] = adapter_path
    else:
        if not adapter_path:
            adapter_path = str(Path(config["training"]["output_dir"]) / "lora_adapter")
        console.print(f"[yellow]Skipping SFT. Using adapter: {adapter_path}[/yellow]")
    
    # ═══════════════════════════════════════════════════════════
    # STAGE 3: EVALUATION
    # ═══════════════════════════════════════════════════════════
    stage_header(3, "Behavioral Evaluation", 
                 "Running both models on all probe sets and scoring with Gemini judge")
    
    if not args.skip_eval:
        # Build evaluator args
        eval_kwargs = {
            "config_path": args.config,
            "api_key": api_key,
            "adapter_path": adapter_path,
            "mock": args.mock,
            "base_only": False,
        }
        
        from evaluator import (
            MockModel, HuggingFaceModel, run_evaluation, print_results_table
        )
        from google import genai
        from evaluator import JUDGE_SYSTEM_PROMPT
        
        judge_client = genai.Client(api_key=api_key) if not args.mock else None
        judge_model_name = config["evaluator"].get("gemini_model", "gemini-2.0-flash")
        
        output_cfg = config["output"]
        
        # Base model
        console.print("[bold]Evaluating BASE model...[/bold]")
        if args.mock:
            base_model = MockModel(persona=MockModel.BASE_PERSONA)
        else:
            base_model = HuggingFaceModel(config["model"]["base_model_id"], config)
        
        if args.mock:
            # Create synthetic results for mock mode
            import random
            random.seed(config["experiment"]["seed"])
            axes = [ax["name"] for ax in config["eval_axes"]]
            
            base_scores_raw = {}
            ft_scores_raw = {}
            
            for axis in axes:
                n = config["dataset"]["n_probe_examples"] // 2
                base_s = [random.randint(4, 7) for _ in range(n)]
                # Fine-tuned model is better on safety/welfare, similar on helpfulness
                if axis in ["harmlessness", "refusal"]:
                    ft_s = [min(10, s + random.randint(1, 3)) for s in base_s]
                elif axis == "welfare_reasoning":
                    ft_s = [min(10, s + random.randint(2, 4)) for s in base_s]
                elif axis == "sycophancy":
                    ft_s = [min(10, s + random.randint(0, 2)) for s in base_s]
                elif axis == "deception":
                    ft_s = [min(10, s + random.randint(0, 2)) for s in base_s]
                elif axis == "helpfulness":
                    ft_s = [max(0, s + random.randint(-1, 1)) for s in base_s]
                else:
                    ft_s = [min(10, s + random.randint(0, 1)) for s in base_s]
                
                base_scores_raw[axis] = {
                    "average_score": round(sum(base_s)/len(base_s), 3),
                    "scores": base_s,
                    "n": len(base_s),
                    "details": [
                        {"probe_id": f"{axis}_{i:03d}", "prompt": f"Probe {i+1}", 
                         "response": f"Base response {i+1}", "score": s, "reasoning": "Mock score"}
                        for i, s in enumerate(base_s)
                    ]
                }
                ft_scores_raw[axis] = {
                    "average_score": round(sum(ft_s)/len(ft_s), 3),
                    "scores": ft_s,
                    "n": len(ft_s),
                    "details": [
                        {"probe_id": f"{axis}_{i:03d}", "prompt": f"Probe {i+1}",
                         "response": f"Fine-tuned response {i+1}", "score": s, "reasoning": "Mock score"}
                        for i, s in enumerate(ft_s)
                    ]
                }
            
            base_path = results_dir / output_cfg["base_scores_file"]
            ft_path = results_dir / output_cfg["finetuned_scores_file"]
            
            with open(base_path, "w") as f:
                json.dump(base_scores_raw, f, indent=2)
            with open(ft_path, "w") as f:
                json.dump(ft_scores_raw, f, indent=2)
            
            console.print(f"[yellow]Mock scores generated and saved.[/yellow]")
            print_results_table(base_scores_raw, "Base Model (Mock)")
            print_results_table(ft_scores_raw, "Fine-Tuned Model (Mock)")
        
        else:
            # Real evaluation
            base_results = run_evaluation(
                model=base_model,
                judge_client=judge_client,
                judge_model_name=judge_model_name,
                config=config,
                model_name="base",
                output_path=results_dir / output_cfg["base_scores_file"],
            )
            print_results_table(base_results, "Base Model")
            
            ft_model = HuggingFaceModel(adapter_path, config)
            ft_results = run_evaluation(
                model=ft_model,
                judge_client=judge_client,
                judge_model_name=judge_model_name,
                config=config,
                model_name="finetuned",
                output_path=results_dir / output_cfg["finetuned_scores_file"],
            )
            print_results_table(ft_results, "Fine-Tuned Model")

    
    # ═══════════════════════════════════════════════════════════
    # STAGE 4: DELTA CALCULATION
    # ═══════════════════════════════════════════════════════════
    stage_header(4, "dS Computation", "Statistical analysis across all behavioral dimensions")
    
    # Temporarily add src to path for imports
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    from delta_calculator import compute_deltas, generate_summary, print_delta_table
    
    base_path = results_dir / config["output"]["base_scores_file"]
    ft_path = results_dir / config["output"]["finetuned_scores_file"]
    
    with open(base_path) as f:
        base_results = json.load(f)
    with open(ft_path) as f:
        ft_results = json.load(f)
    
    delta_report = compute_deltas(base_results, ft_results)
    summary = generate_summary(delta_report)
    print_delta_table(delta_report)
    
    # Save reports
    import numpy as np
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    full_report = {"summary": summary, "per_axis": delta_report}
    report_path = results_dir / config["output"]["delta_report_file"]
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, cls=NumpyEncoder)

    
    # Dashboard data
    dashboard_path = Path(config["output"]["dashboard_data_file"])
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_data = {
        "summary": summary,
        "axes": list(delta_report.keys()),
        "base_scores": {ax: d["base_mean"] for ax, d in delta_report.items()},
        "finetuned_scores": {ax: d["finetuned_mean"] for ax, d in delta_report.items()},
        "deltas": {ax: d["delta"] for ax, d in delta_report.items()},
        "significance": {ax: d["significance_ttest"] for ax, d in delta_report.items()},
        "effect_sizes": {ax: d["cohen_d"] for ax, d in delta_report.items()},
        "p_values": {ax: d["p_value_ttest"] for ax, d in delta_report.items()},
        "ci_lower": {ax: d["ci_95_lower"] for ax, d in delta_report.items()},
        "ci_upper": {ax: d["ci_95_upper"] for ax, d in delta_report.items()},
        "raw_base_scores": {ax: d["base_scores"] for ax, d in delta_report.items()},
        "raw_finetuned_scores": {ax: d["finetuned_scores"] for ax, d in delta_report.items()},
        "details": {
            ax: {
                "details": base_results.get(ax, {}).get("details", []),
                "ft_details": ft_results.get(ax, {}).get("details", []),
            }
            for ax in delta_report.keys()
        }
    }
    with open(dashboard_path, "w") as f:
        json.dump(dashboard_data, f, indent=2, cls=NumpyEncoder)

    
    # ═══════════════════════════════════════════════════════════
    # STAGE 5: OOD ANALYSIS
    # ═══════════════════════════════════════════════════════════
    stage_header(5, "OOD Generalization", "Testing transfer to out-of-distribution prompts")
    
    from ood_tester import load_ood_probes, generate_ood_report
    
    probe_dir = Path(config["dataset"]["eval_probes_dir"])
    ood_probes = load_ood_probes(probe_dir)
    ood_report_path = results_dir / "ood_report.json"
    ood_report = generate_ood_report(base_results, ft_results, ood_probes, ood_report_path)
    
    # Update dashboard data with OOD
    dashboard_data["ood_report"] = ood_report
    with open(dashboard_path, "w") as f:
        json.dump(dashboard_data, f, indent=2)
    
    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    meta["completed_at"] = datetime.now().isoformat()
    meta["elapsed_seconds"] = round(elapsed, 1)
    
    with open(results_dir / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    
    console.print(f"\n")
    console.print(Rule("[bold green]PIPELINE COMPLETE[/bold green]", style="green"))
    
    table = Table(title="Final Results Summary", show_header=True, header_style="bold green")
    table.add_column("Dimension", style="cyan", width=28)
    table.add_column("dS", justify="right", style="bold")
    table.add_column("Significance", justify="center")
    
    for axis, data in sorted(delta_report.items()):
        delta = data["delta"]
        color = "green" if delta > 0 else ("red" if delta < -0.1 else "white")
        table.add_row(
            axis.replace("_", " ").title(),
            f"[{color}]{delta:+.3f}[/{color}]",
            data["significance_ttest"],
        )
    
    console.print(table)
    
    console.print(f"\n  [bold]Key Finding:[/bold] Safety transfer detected: [{'bold green' if summary['transfer_detected'] else 'bold red'}]{summary['transfer_detected']}[/]")
    console.print(f"  Elapsed time: {elapsed:.1f}s")
    console.print(f"\n  [bold]Outputs:[/bold]")
    console.print(f"    📊 Delta report: {report_path}")
    console.print(f"    🌐 Dashboard:    dashboard/index.html")
    console.print(f"    📁 All results:  {results_dir}/")
    console.print(f"\n  [bold cyan]Open dashboard/index.html in your browser to explore results![/bold cyan]")


def main():
    parser = argparse.ArgumentParser(
        description="Emergent Alignment / Behavioral Transfer Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--config", type=str, default="config/experiment_config.yaml")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API key")
    parser.add_argument("--mock", action="store_true", help="Run full pipeline with mock models")
    parser.add_argument("--skip-sft", action="store_true", help="Skip fine-tuning step")
    parser.add_argument("--skip-data", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation (use existing scores)")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation stages")
    parser.add_argument("--data-only", action="store_true", help="Only run data preparation")
    parser.add_argument("--probe-only", action="store_true", help="Only save probe sets (no SFT data)")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to LoRA adapter (overrides config)")
    args = parser.parse_args()
    
    # Add src dir to path
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    run_pipeline(args)


if __name__ == "__main__":
    main()
