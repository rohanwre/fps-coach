"""Run independent training ablations concurrently on separate Godot ports."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AblationJob:
    name: str
    reward_profile: str
    port: int


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple Godot training ablations in parallel.")
    parser.add_argument("--name-prefix", default="enemy_ppo_parallel")
    parser.add_argument("--reward-profiles", default="timeout,timeout_centered")
    parser.add_argument("--scripted-player-profile", default="medium")
    parser.add_argument("--ppo-difficulty-profile", default="adaptive")
    parser.add_argument("--timesteps", type=int, default=3_000)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--speedup", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--start-port", type=int, default=11008)
    parser.add_argument("--godot-path", default=os.environ.get("GODOT_BIN", "godot"))
    parser.add_argument("--godot-start-delay", type=float, default=5.0)
    parser.add_argument("--godot-stop-timeout", type=float, default=90.0)
    parser.add_argument("--project-path", default=str(REPO_ROOT / "godot_project"))
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "training" / "models"))
    parser.add_argument(
        "--log-root",
        default=str(REPO_ROOT / "training" / "logs" / "gameplay_parallel"),
    )
    parser.add_argument(
        "--summary-path",
        default=str(REPO_ROOT / "training" / "models" / "parallel_ablation_summary.json"),
    )
    parser.add_argument(
        "--comparison-output",
        default=str(REPO_ROOT / "training" / "models" / "parallel_ablation_comparison.md"),
    )
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> list[AblationJob]:
    profiles = parse_csv(args.reward_profiles)
    return [
        AblationJob(
            name=f"{args.name_prefix}_{profile}",
            reward_profile=profile,
            port=args.start_port + index,
        )
        for index, profile in enumerate(profiles)
    ]


def run_job(args: argparse.Namespace, job: AblationJob) -> dict[str, Any]:
    job_log_dir = pathlib.Path(args.log_root) / job.name
    command = [
        sys.executable,
        str(REPO_ROOT / "training" / "run_training_ablation.py"),
        "--name",
        job.name,
        "--reward-profile",
        job.reward_profile,
        "--scripted-player-profile",
        args.scripted_player_profile,
        "--ppo-difficulty-profile",
        args.ppo_difficulty_profile,
        "--timesteps",
        str(args.timesteps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--speedup",
        str(args.speedup),
        "--port",
        str(job.port),
        "--godot-path",
        args.godot_path,
        "--godot-start-delay",
        str(args.godot_start_delay),
        "--godot-stop-timeout",
        str(args.godot_stop_timeout),
        "--project-path",
        args.project_path,
        "--models-dir",
        args.models_dir,
        "--log-dir",
        str(job_log_dir),
    ]
    started_command = " ".join(command)
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    metrics_path = pathlib.Path(args.models_dir) / f"{job.name}_gameplay_metrics.json"
    return {
        "name": job.name,
        "reward_profile": job.reward_profile,
        "port": job.port,
        "returncode": result.returncode,
        "command": started_command,
        "metrics_path": str(metrics_path),
        "metrics_exists": metrics_path.exists(),
    }


def write_comparison(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    metrics_paths = [result["metrics_path"] for result in results if result["metrics_exists"]]
    if not metrics_paths:
        return

    command = [
        sys.executable,
        str(REPO_ROOT / "training" / "compare_gameplay_runs.py"),
        *metrics_paths,
        "--output",
        args.comparison_output,
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    max_workers = max(1, min(args.jobs, len(jobs)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_job, args, job) for job in jobs]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    results.sort(key=lambda item: item["port"])
    summary_path = pathlib.Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"jobs": results}, indent=2), encoding="utf-8")
    write_comparison(args, results)

    failed = [result for result in results if result["returncode"] != 0]
    if failed:
        failed_names = ", ".join(result["name"] for result in failed)
        raise SystemExit(f"Parallel ablation jobs failed: {failed_names}")

    print(f"Wrote parallel ablation summary to {summary_path}")
    if pathlib.Path(args.comparison_output).exists():
        print(f"Wrote comparison report to {args.comparison_output}")


if __name__ == "__main__":
    main()
