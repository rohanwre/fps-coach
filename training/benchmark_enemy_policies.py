"""Benchmark compatible Enemy PPO checkpoints against one fixed scripted Player."""

from __future__ import annotations

import argparse
import csv
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

from training.evaluate_gameplay_logs import parse_jsonl, summarize_events
from training.run_training_ablation import find_newest_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Enemy PPO checkpoints against a fixed scripted Player.")
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        help="Policy to benchmark in label=path form. Repeat for each compatible checkpoint.",
    )
    parser.add_argument("--scripted-player-profile", default="easy")
    parser.add_argument("--ppo-difficulty-profile", default="medium")
    parser.add_argument("--reward-profile", default="default")
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--stochastic-actions", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speedup", type=int, default=4)
    parser.add_argument("--start-port", type=int, default=11600)
    parser.add_argument("--godot-start-delay", type=float, default=4.0)
    parser.add_argument("--godot-stop-timeout", type=float, default=20.0)
    parser.add_argument("--godot-path", required=True)
    parser.add_argument("--project-path", default=str(REPO_ROOT / "godot_project"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "training" / "benchmarks" / "enemy_fixed_easy"))
    return parser.parse_args()


def parse_policies(values: list[str]) -> list[tuple[str, pathlib.Path]]:
    policies: list[tuple[str, pathlib.Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Policy must use label=path form: {value}")
        label, raw_path = value.split("=", 1)
        path = pathlib.Path(raw_path).expanduser().resolve()
        if not label.strip():
            raise ValueError("Policy label cannot be empty")
        if not path.exists():
            raise FileNotFoundError(path)
        policies.append((label.strip(), path))
    return policies


def build_policy_command(
    args: argparse.Namespace,
    model_path: pathlib.Path,
    port: int,
    log_dir: pathlib.Path,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "training" / "run_policy.py"),
        "--model-path",
        str(model_path),
        "--agent-role",
        "enemy",
        "--scripted-player-profile",
        args.scripted_player_profile,
        "--ppo-difficulty-profile",
        args.ppo_difficulty_profile,
        "--reward-profile",
        args.reward_profile,
        "--port",
        str(port),
        "--steps",
        str(args.steps),
        "--gameplay-log-dir",
        str(log_dir),
    ]
    if args.stochastic_actions:
        command.append("--stochastic-actions")
    command.extend(["--seed", str(args.seed)])
    return command


def build_godot_command(args: argparse.Namespace, port: int, log_dir: pathlib.Path) -> list[str]:
    return [
        args.godot_path,
        "--headless",
        f"--port={port}",
        f"--speedup={args.speedup}",
        "--path",
        args.project_path,
        "--",
        "--agent-role=enemy",
        f"--scripted-player-profile={args.scripted_player_profile}",
        f"--ppo-difficulty-profile={args.ppo_difficulty_profile}",
        f"--reward-profile={args.reward_profile}",
        "--disable-coaching-frame-capture=true",
        f"--gameplay-log-dir={log_dir}",
    ]


def benchmark_policy(
    args: argparse.Namespace,
    label: str,
    model_path: pathlib.Path,
    port: int,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    log_dir = (output_dir / "logs" / label).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    process_log_path = output_dir / "process_logs" / f"{label}.log"
    process_log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
    env.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

    with process_log_path.open("w", encoding="utf-8") as process_log:
        policy_process = subprocess.Popen(
            build_policy_command(args, model_path, port, log_dir),
            cwd=REPO_ROOT,
            env=env,
            stdout=process_log,
            stderr=subprocess.STDOUT,
        )
        godot_process = None
        try:
            time.sleep(max(args.godot_start_delay, 0.0))
            godot_process = subprocess.Popen(
                build_godot_command(args, port, log_dir),
                cwd=REPO_ROOT,
                env=env,
                stdout=process_log,
                stderr=subprocess.STDOUT,
            )
            policy_return = policy_process.wait()
            try:
                godot_return = godot_process.wait(timeout=max(args.godot_stop_timeout, 0.0))
            except subprocess.TimeoutExpired:
                godot_process.terminate()
                godot_return = godot_process.wait(timeout=10)
        finally:
            if policy_process.poll() is None:
                policy_process.terminate()
            if godot_process is not None and godot_process.poll() is None:
                godot_process.terminate()

    if policy_return != 0:
        raise RuntimeError(f"Policy process failed for {label}; inspect {process_log_path}")
    if godot_return != 0:
        raise RuntimeError(f"Godot process failed for {label}; inspect {process_log_path}")

    source_log = find_newest_log(log_dir, started_at)
    return {
        "label": label,
        "model_path": str(model_path.relative_to(REPO_ROOT)),
        "source_log": str(source_log.relative_to(REPO_ROOT)),
        "metrics": summarize_events(parse_jsonl(source_log)),
    }


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    human = metrics["human_likeness"]
    return {
        "policy": result["label"],
        "rounds": metrics["round_count"],
        "wins": metrics["enemy_wins"],
        "losses": metrics["enemy_losses"],
        "timeouts": metrics["timeout_rounds"],
        "hits": metrics["enemy_hits"],
        "shots": metrics["enemy_total_shots"],
        "hit_rate": metrics["enemy_hit_rate"],
        "damage_dealt": metrics["enemy_damage_dealt"],
        "damage_diff": metrics["enemy_damage_diff"],
        "mean_round_duration_sec": metrics["mean_round_duration_sec"],
        "reaction_delay_min_sec": human["reaction_delay_min_sec"],
        "shots_without_line_of_sight": human["shots_without_line_of_sight"],
        "human_likeness_flags": human["flags"],
    }


def write_results(args: argparse.Namespace, results: list[dict[str, Any]], output_dir: pathlib.Path) -> None:
    payload = {
        "protocol": {
            "scripted_player_profile": args.scripted_player_profile,
            "ppo_difficulty_profile": args.ppo_difficulty_profile,
            "reward_profile": args.reward_profile,
            "steps_per_policy": args.steps,
            "speedup": args.speedup,
            "deterministic_actions": not args.stochastic_actions,
            "seed": args.seed,
        },
        "results": results,
        "table": [flatten_result(result) for result in results],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table = payload["table"]
    with (output_dir / "benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        for row in table:
            csv_row = row.copy()
            csv_row["human_likeness_flags"] = json.dumps(csv_row["human_likeness_flags"], separators=(",", ":"))
            writer.writerow(csv_row)


def main() -> None:
    args = parse_args()
    policies = parse_policies(args.policy)
    output_dir = pathlib.Path(args.output_dir).resolve()
    results = []
    for index, (label, model_path) in enumerate(policies):
        print(f"Benchmarking {label} against scripted {args.scripted_player_profile} Player")
        results.append(benchmark_policy(args, label, model_path, args.start_port + index, output_dir))
    write_results(args, results, output_dir)
    print(f"Wrote fixed-opponent benchmark to {output_dir}")


if __name__ == "__main__":
    main()
