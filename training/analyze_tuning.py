"""Analyze tuning outputs and generate compact experiment artifacts."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze enemy tuning results.")
    parser.add_argument(
        "--results-path",
        default="training/models/enemy_tuning_results.json",
        help="Path to tuning results JSON.",
    )
    parser.add_argument(
        "--analysis-path",
        default=None,
        help="Optional output path for analysis JSON. Defaults beside results file.",
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Optional output path for mean reward bar chart PNG.",
    )
    return parser.parse_args()


def load_results(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_analysis(results: dict[str, Any]) -> dict[str, Any]:
    all_runs = results.get("all_runs", [])
    if not all_runs:
        raise ValueError("No runs found in tuning results.")

    rewards = [float(run["mean_reward"]) for run in all_runs]
    learning_rates = sorted({float(run["learning_rate"]) for run in all_runs})
    gammas = sorted({float(run["gamma"]) for run in all_runs})

    analysis = {
        "total_runs": len(all_runs),
        "reward_mean": statistics.fmean(rewards),
        "reward_std": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "learning_rates": learning_rates,
        "gammas": gammas,
        "best_run": results.get("best_run"),
        "best_config": results.get("best_config"),
    }
    return analysis


def maybe_plot(results: dict[str, Any], plot_path: pathlib.Path) -> None:
    config_summaries = results.get("config_summaries", [])
    if not config_summaries:
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping plot generation.")
        return

    labels = [entry["config_key"] for entry in config_summaries]
    means = [float(entry["mean_reward_avg"]) for entry in config_summaries]
    stds = [float(entry["mean_reward_std"]) for entry in config_summaries]

    plt.figure(figsize=(max(8, len(labels) * 1.2), 4.8))
    plt.bar(labels, means, yerr=stds, capsize=4)
    plt.title("Tuning Mean Reward by Config")
    plt.ylabel("Mean Reward")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()


def main() -> None:
    args = parse_args()
    results_path = pathlib.Path(args.results_path)
    results = load_results(results_path)
    analysis = build_analysis(results)

    if args.analysis_path is not None:
        analysis_path = pathlib.Path(args.analysis_path)
    else:
        analysis_path = results_path.with_name(f"{results_path.stem}_analysis.json")
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"Wrote analysis to {analysis_path}")

    if args.plot_path is not None:
        plot_path = pathlib.Path(args.plot_path)
    else:
        plot_path = results_path.with_name(f"{results_path.stem}_mean_reward_plot.png")
    maybe_plot(results, plot_path)
    if plot_path.exists():
        print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
