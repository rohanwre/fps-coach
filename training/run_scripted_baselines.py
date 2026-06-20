"""Run and evaluate scripted enemy baseline profiles.

Launches Godot headlessly for each requested scripted profile, then evaluates the
resulting gameplay JSONL log with the same metrics used for trained policies.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.compare_gameplay_runs import build_markdown, build_rows
from training.evaluate_gameplay_logs import parse_jsonl, summarize_events

DEFAULT_GODOT_LOG_DIR = (
    pathlib.Path.home()
    / "Library"
    / "Application Support"
    / "Godot"
    / "app_userdata"
    / "Adaptive FPS Coach"
    / "gameplay_logs"
)
VALID_PROFILES = ("easy", "medium", "hard", "adaptive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scripted enemy baseline profiles in Godot.")
    parser.add_argument(
        "--godot-path",
        default=os.environ.get("GODOT_BIN", "godot"),
        help="Path to the Godot executable. Defaults to GODOT_BIN or 'godot'.",
    )
    parser.add_argument("--project-path", default=str(REPO_ROOT / "godot_project"))
    parser.add_argument("--profiles", default="easy,medium,hard,adaptive")
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--log-dir", default=str(DEFAULT_GODOT_LOG_DIR))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "training" / "models" / "scripted_baselines"))
    return parser.parse_args()


def parse_profiles(value: str) -> list[str]:
    profiles = [profile.strip() for profile in value.split(",") if profile.strip()]
    invalid = [profile for profile in profiles if profile not in VALID_PROFILES]
    if invalid:
        raise ValueError(f"Invalid profile(s): {', '.join(invalid)}")
    if not profiles:
        raise ValueError("At least one profile is required.")
    return profiles


def find_newest_log(log_dir: pathlib.Path, started_at: float) -> pathlib.Path:
    candidates = [
        path
        for path in log_dir.glob("*.jsonl")
        if path.stat().st_mtime >= started_at
    ]
    if not candidates:
        raise FileNotFoundError(f"No new gameplay log found in {log_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_metrics(log_path: pathlib.Path, output_path: pathlib.Path) -> None:
    metrics = summarize_events(parse_jsonl(log_path))
    summary: dict[str, Any] = {
        "num_logs": 1,
        "num_aggregate_logs": 1 if int(metrics["round_count"]) > 0 else 0,
        "excluded_zero_round_runs": 0 if int(metrics["round_count"]) > 0 else 1,
        "aggregate": {
            "enemy_win_rate_mean": metrics["enemy_win_rate"] if int(metrics["round_count"]) > 0 else 0.0,
            "enemy_hit_rate_mean": metrics["enemy_hit_rate"] if int(metrics["round_count"]) > 0 else 0.0,
            "enemy_damage_diff_mean": metrics["enemy_damage_diff"] if int(metrics["round_count"]) > 0 else 0.0,
            "player_hit_rate_mean": metrics["player_skill"]["hit_rate"] if int(metrics["round_count"]) > 0 else 0.0,
            "player_damage_diff_mean": metrics["player_skill"]["damage_diff"] if int(metrics["round_count"]) > 0 else 0.0,
            "enemy_win_rate_std": 0.0,
            "includes_zero_round_runs": False,
        },
        "per_run": [{"path": str(log_path), "metrics": metrics}],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_profile(args: argparse.Namespace, profile: str, output_dir: pathlib.Path) -> pathlib.Path:
    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    command = [
        args.godot_path,
        "--headless",
        "--path",
        args.project_path,
        "--",
        "--baseline-profile",
        profile,
        "--baseline-duration-seconds",
        str(args.duration_seconds),
    ]
    subprocess.run(command, check=True)
    log_path = find_newest_log(log_dir, started_at)
    metrics_path = output_dir / f"scripted_{profile}_gameplay_metrics.json"
    write_metrics(log_path, metrics_path)
    return metrics_path


def main() -> None:
    args = parse_args()
    profiles = parse_profiles(args.profiles)
    output_dir = pathlib.Path(args.output_dir)

    metrics_paths = [run_profile(args, profile, output_dir) for profile in profiles]
    comparison_path = output_dir / "scripted_baselines_comparison.md"
    comparison_path.write_text(build_markdown(build_rows(metrics_paths)), encoding="utf-8")
    print(f"Wrote scripted baseline comparison to {comparison_path}")


if __name__ == "__main__":
    main()
