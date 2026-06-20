"""Run one Godot RL training ablation and evaluate the newest gameplay log."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one adaptive-fps-coach training ablation.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--reward-profile", default="default")
    parser.add_argument("--scripted-player-profile", default="medium")
    parser.add_argument("--ppo-difficulty-profile", default="adaptive")
    parser.add_argument("--timesteps", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--speedup", type=int, default=1)
    parser.add_argument("--port", type=int, default=11008)
    parser.add_argument("--godot-path", default=os.environ.get("GODOT_BIN", "godot"))
    parser.add_argument("--godot-start-delay", type=float, default=5.0)
    parser.add_argument("--godot-stop-timeout", type=float, default=90.0)
    parser.add_argument("--project-path", default=str(REPO_ROOT / "godot_project"))
    parser.add_argument("--log-dir", default=str(DEFAULT_GODOT_LOG_DIR))
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "training" / "models"))
    parser.add_argument("--load-model-path", default=None)
    return parser.parse_args()


def find_newest_log(log_dir: pathlib.Path, started_at: float) -> pathlib.Path:
    candidates = [path for path in log_dir.glob("*.jsonl") if path.stat().st_mtime >= started_at]
    if not candidates:
        raise FileNotFoundError(f"No new gameplay log found in {log_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_metrics(log_path: pathlib.Path, output_path: pathlib.Path) -> None:
    metrics = summarize_events(parse_jsonl(log_path))
    has_rounds = int(metrics["round_count"]) > 0
    summary: dict[str, Any] = {
        "num_logs": 1,
        "num_aggregate_logs": 1 if has_rounds else 0,
        "excluded_zero_round_runs": 0 if has_rounds else 1,
        "aggregate": {
            "enemy_win_rate_mean": metrics["enemy_win_rate"] if has_rounds else 0.0,
            "enemy_hit_rate_mean": metrics["enemy_hit_rate"] if has_rounds else 0.0,
            "enemy_damage_diff_mean": metrics["enemy_damage_diff"] if has_rounds else 0.0,
            "player_hit_rate_mean": metrics["player_skill"]["hit_rate"] if has_rounds else 0.0,
            "player_damage_diff_mean": metrics["player_skill"]["damage_diff"] if has_rounds else 0.0,
            "enemy_win_rate_std": 0.0,
            "includes_zero_round_runs": False,
        },
        "per_run": [{"path": str(log_path), "metrics": metrics}],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_train_command(args: argparse.Namespace, models_dir: pathlib.Path, metadata_dir: pathlib.Path) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "training" / "train_enemy.py"),
        "--timesteps",
        str(args.timesteps),
        "--eval-episodes",
        str(args.eval_episodes),
        "--speedup",
        str(args.speedup),
        "--port",
        str(args.port),
        "--experiment-name",
        args.name,
        "--save-model-path",
        str(models_dir / f"{args.name}.zip"),
        "--save-metadata-path",
        str(metadata_dir / f"{args.name}.json"),
        "--scripted-player-profile",
        getattr(args, "scripted_player_profile", "medium"),
        "--ppo-difficulty-profile",
        getattr(args, "ppo_difficulty_profile", "adaptive"),
        "--reward-profile",
        args.reward_profile,
    ]
    if getattr(args, "load_model_path", None):
        command.extend(["--load-model-path", args.load_model_path])
    return command


def build_godot_command(args: argparse.Namespace) -> list[str]:
    return [
        args.godot_path,
        "--headless",
        "--path",
        args.project_path,
        f"--port={args.port}",
        f"--speedup={args.speedup}",
        "--",
        f"--reward-profile={args.reward_profile}",
        f"--scripted-player-profile={getattr(args, 'scripted_player_profile', 'medium')}",
        f"--ppo-difficulty-profile={getattr(args, 'ppo_difficulty_profile', 'adaptive')}",
        f"--gameplay-log-dir={args.log_dir}",
    ]


def main() -> None:
    args = parse_args()
    models_dir = pathlib.Path(args.models_dir)
    metadata_dir = models_dir / "metadata"
    process_log_dir = REPO_ROOT / "training" / "logs"
    process_log_dir.mkdir(parents=True, exist_ok=True)
    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
    env.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

    train_command = build_train_command(args, models_dir, metadata_dir)
    godot_command = build_godot_command(args)

    process_log_path = process_log_dir / f"{args.name}_process.log"
    with process_log_path.open("w", encoding="utf-8") as process_log:
        print(f"Writing process output to {process_log_path}")
        godot_process = None
        train_process = subprocess.Popen(
            train_command,
            cwd=REPO_ROOT,
            env=env,
            stdout=process_log,
            stderr=subprocess.STDOUT,
        )
        try:
            time.sleep(max(args.godot_start_delay, 0.0))
            godot_process = subprocess.Popen(
                godot_command,
                cwd=REPO_ROOT,
                env=env,
                stdout=process_log,
                stderr=subprocess.STDOUT,
            )
            train_return = train_process.wait()
            try:
                godot_return = godot_process.wait(timeout=max(args.godot_stop_timeout, 0.0))
            except subprocess.TimeoutExpired:
                godot_process.terminate()
                godot_return = godot_process.wait(timeout=10)
        finally:
            if train_process.poll() is None:
                train_process.terminate()
            if godot_process is not None and godot_process.poll() is None:
                godot_process.terminate()

    if train_return != 0:
        raise SystemExit(train_return)
    if godot_return != 0:
        raise SystemExit(godot_return)

    log_path = find_newest_log(log_dir, started_at)
    metrics_path = models_dir / f"{args.name}_gameplay_metrics.json"
    write_metrics(log_path, metrics_path)
    print(f"Wrote gameplay metrics to {metrics_path}")


if __name__ == "__main__":
    main()
