"""
delta_calculator.py
===================
Computes ΔS = S_finetuned - S_base for each behavioral dimension,
performs statistical significance tests, and generates the delta report.

Usage:
    python src/delta_calculator.py --config config/experiment_config.yaml
"""

import json
import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SIGNIFICANCE_LEVELS = {
    0.001: "***",
    0.01: "**",
    0.05: "*",
    0.10: "·",
}


def significance_stars(p_value: float) -> str:
    for threshold, stars in SIGNIFICANCE_LEVELS.items():
        if p_value < threshold:
            return stars
    return "ns"


def cohen_d(a: list, b: list) -> float:
    """Compute Cohen's d effect size."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((n_a - 1) * np.std(a, ddof=1) ** 2 + (n_b - 1) * np.std(b, ddof=1) ** 2)
        / (n_a + n_b - 2)
    )
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def interpret_effect(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def compute_deltas(base_results: dict, ft_results: dict) -> dict:
    """Compute ΔS with statistics for each axis."""
    delta_report = {}
    axes = set(base_results.keys()) & set(ft_results.keys())
    
    for axis in sorted(axes):
        base_data = base_results[axis]
        ft_data = ft_results[axis]
        
        base_scores = base_data["scores"]
        ft_scores = ft_data["scores"]
        
        # Align lengths (take min)
        n = min(len(base_scores), len(ft_scores))
        base_scores = base_scores[:n]
        ft_scores = ft_scores[:n]
        
        base_mean = np.mean(base_scores)
        ft_mean = np.mean(ft_scores)
        delta = ft_mean - base_mean
        
        # Paired t-test
        if n >= 2:
            t_stat, p_value_ttest = stats.ttest_rel(ft_scores, base_scores)
        else:
            t_stat, p_value_ttest = 0.0, 1.0
        
        # Wilcoxon signed-rank test (non-parametric)
        if n >= 10 and not all(f == b for f, b in zip(ft_scores, base_scores)):
            try:
                w_stat, p_value_wilcoxon = stats.wilcoxon(ft_scores, base_scores)
            except Exception:
                w_stat, p_value_wilcoxon = 0.0, 1.0
        else:
            w_stat, p_value_wilcoxon = 0.0, 1.0
        
        # Effect size
        d = cohen_d(ft_scores, base_scores)
        
        # 95% CI for delta
        diff_scores = [f - b for f, b in zip(ft_scores, base_scores)]
        if n >= 2:
            se = np.std(diff_scores, ddof=1) / np.sqrt(n)
            ci_lower = delta - 1.96 * se
            ci_upper = delta + 1.96 * se
        else:
            ci_lower, ci_upper = delta, delta
        
        # Improvement direction (axis-specific — higher is always better in our rubric)
        improved = delta > 0
        
        delta_report[axis] = {
            "base_mean": round(float(base_mean), 4),
            "finetuned_mean": round(float(ft_mean), 4),
            "delta": round(float(delta), 4),
            "delta_pct": round(float((delta / base_mean * 100) if base_mean != 0 else 0), 2),
            "n_samples": n,
            "t_statistic": round(float(t_stat), 4),
            "p_value_ttest": round(float(p_value_ttest), 6),
            "p_value_wilcoxon": round(float(p_value_wilcoxon), 6),
            "significance_ttest": significance_stars(p_value_ttest),
            "cohen_d": round(float(d), 4),
            "effect_size": interpret_effect(d),
            "ci_95_lower": round(float(ci_lower), 4),
            "ci_95_upper": round(float(ci_upper), 4),
            "improved": improved,
            "base_scores": base_scores,
            "finetuned_scores": ft_scores,
        }
    
    return delta_report


def generate_summary(delta_report: dict) -> dict:
    """Generate a narrative summary of the findings."""
    improvements = [ax for ax, d in delta_report.items() if d["delta"] > 0.5]
    regressions = [ax for ax, d in delta_report.items() if d["delta"] < -0.5]
    significant = [
        ax for ax, d in delta_report.items() 
        if d["p_value_ttest"] < 0.05
    ]
    
    avg_delta = np.mean([d["delta"] for d in delta_report.values()])
    welfare_delta = delta_report.get("welfare_reasoning", {}).get("delta", 0)
    safety_delta = np.mean([
        delta_report.get(ax, {}).get("delta", 0)
        for ax in ["harmlessness", "refusal"]
        if ax in delta_report
    ])
    
    return {
        "overall_avg_delta": round(float(avg_delta), 4),
        "welfare_to_safety_transfer": round(float(safety_delta), 4),
        "improved_axes": improvements,
        "regressed_axes": regressions,
        "statistically_significant_axes": significant,
        "n_axes": len(delta_report),
        "transfer_detected": any(
            ax in significant and delta_report[ax]["delta"] > 0
            for ax in ["harmlessness", "refusal", "sycophancy", "deception"]
        ),
        "welfare_sft_improved_welfare": welfare_delta > 0,
        "welfare_improvement_magnitude": interpret_effect(
            delta_report.get("welfare_reasoning", {}).get("cohen_d", 0)
        ),
    }


def print_delta_table(delta_report: dict):
    """Print a rich formatted table of delta results."""
    table = Table(
        title="ΔS = S_fine-tuned − S_base (Per Behavioral Dimension)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Dimension", style="cyan", width=22)
    table.add_column("Base", justify="right", style="yellow")
    table.add_column("Fine-tuned", justify="right", style="green")
    table.add_column("Δ", justify="right", style="bold")
    table.add_column("Δ%", justify="right")
    table.add_column("Sig.", justify="center")
    table.add_column("Cohen's d", justify="right")
    table.add_column("Effect", justify="center")
    
    for axis, data in sorted(delta_report.items()):
        delta = data["delta"]
        delta_color = "green" if delta > 0 else ("red" if delta < 0 else "white")
        arrow = "↑" if delta > 0.1 else ("↓" if delta < -0.1 else "→")
        
        table.add_row(
            axis,
            f"{data['base_mean']:.2f}",
            f"{data['finetuned_mean']:.2f}",
            f"[{delta_color}]{arrow} {delta:+.3f}[/{delta_color}]",
            f"{data['delta_pct']:+.1f}%",
            data["significance_ttest"],
            f"{data['cohen_d']:+.3f}",
            data["effect_size"],
        )
    
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Compute ΔS across behavioral dimensions")
    parser.add_argument("--config", type=str, default="config/experiment_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    output_cfg = config["output"]
    results_dir = Path(output_cfg["results_dir"])
    
    base_path = results_dir / output_cfg["base_scores_file"]
    ft_path = results_dir / output_cfg["finetuned_scores_file"]
    
    if not base_path.exists():
        console.print(f"[red]Error: Base scores not found at {base_path}[/red]")
        console.print("[yellow]Run evaluator.py first.[/yellow]")
        return
    
    if not ft_path.exists():
        console.print(f"[red]Error: Fine-tuned scores not found at {ft_path}[/red]")
        console.print("[yellow]Run evaluator.py first.[/yellow]")
        return
    
    with open(base_path) as f:
        base_results = json.load(f)
    with open(ft_path) as f:
        ft_results = json.load(f)
    
    console.print(Panel.fit(
        "[bold magenta]Computing ΔS — Behavioral Transfer Analysis[/bold magenta]",
        border_style="magenta"
    ))
    
    delta_report = compute_deltas(base_results, ft_results)
    summary = generate_summary(delta_report)
    
    print_delta_table(delta_report)
    
    # Print summary
    console.print("\n[bold]Research Summary:[/bold]")
    console.print(f"  Overall average ΔS: [{'green' if summary['overall_avg_delta'] > 0 else 'red'}]{summary['overall_avg_delta']:+.3f}[/]")
    console.print(f"  Safety transfer (harmlessness + refusal): [{'green' if summary['welfare_to_safety_transfer'] > 0 else 'red'}]{summary['welfare_to_safety_transfer']:+.3f}[/]")
    console.print(f"  Welfare SFT improved welfare reasoning: [{'green' if summary['welfare_sft_improved_welfare'] else 'red'}]{summary['welfare_sft_improved_welfare']}[/]")
    console.print(f"  Cross-domain transfer detected: [{'green' if summary['transfer_detected'] else 'yellow'}]{summary['transfer_detected']}[/]")
    console.print(f"  Improved axes: [green]{', '.join(summary['improved_axes']) or 'none'}[/green]")
    console.print(f"  Regressed axes: [red]{', '.join(summary['regressed_axes']) or 'none'}[/red]")
    console.print(f"  Statistically significant (p<0.05): [cyan]{', '.join(summary['statistically_significant_axes']) or 'none'}[/cyan]")
    
    # Save full report
    full_report = {
        "summary": summary,
        "per_axis": delta_report,
        "significance_legend": {
            "***": "p < 0.001",
            "**": "p < 0.01",
            "*": "p < 0.05",
            "·": "p < 0.10",
            "ns": "not significant",
        }
    }
    
    report_path = results_dir / output_cfg["delta_report_file"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    # Also save dashboard-ready data
    dashboard_path = Path(output_cfg["dashboard_data_file"])
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
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[green]✓ Delta report saved to {report_path}[/green]")
    console.print(f"[green]✓ Dashboard data saved to {dashboard_path}[/green]")
    console.print("\n[bold green]✓ Open dashboard/index.html to visualize results![/bold green]")


if __name__ == "__main__":
    main()
