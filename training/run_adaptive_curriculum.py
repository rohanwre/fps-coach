"""Continue one Enemy PPO policy through scripted Player difficulty stages."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Enemy PPO through easy/medium/hard scripted Players.")
    parser.add_argument("--name-prefix", default="enemy_ppo_adaptive_curriculum")
    parser.add_argument("--profiles", default="easy,medium,hard")
    parser.add_argument("--timesteps-per-stage", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--speedup", type=int, default=4)
    parser.add_argument("--start-port", type=int, default=11500)
    parser.add_argument("--godot-path", required=True)
    parser.add_argument("--models-dir", default=str(REPO_ROOT / "training" / "models"))
    parser.add_argument("--log-root", default=str(REPO_ROOT / "training" / "logs" / "adaptive_curriculum"))
    return parser.parse_args()


def parse_profiles(value: str) -> list[str]:
    profiles = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [profile for profile in profiles if profile not in {"easy", "medium", "hard", "adaptive"}]
    if invalid:
        raise ValueError(f"Unknown scripted Player profiles: {', '.join(invalid)}")
    return profiles


def build_stage_command(
    args: argparse.Namespace,
    profile: str,
    stage_index: int,
    load_model_path: pathlib.Path | None,
) -> tuple[list[str], pathlib.Path]:
    models_dir = pathlib.Path(args.models_dir)
    stage_name = f"{args.name_prefix}_{stage_index + 1:02d}_{profile}"
    save_model_path = models_dir / f"{stage_name}.zip"
    command = [
        sys.executable,
        str(REPO_ROOT / "training" / "run_training_ablation.py"),
        "--name",
        stage_name,
        "--reward-profile",
        "timeout_aim_assist",
        "--scripted-player-profile",
        profile,
        "--ppo-difficulty-profile",
        "adaptive",
        "--timesteps",
        str(args.timesteps_per_stage),
        "--eval-episodes",
        str(args.eval_episodes),
        "--speedup",
        str(args.speedup),
        "--port",
        str(args.start_port + stage_index),
        "--godot-path",
        args.godot_path,
        "--models-dir",
        str(models_dir),
        "--log-dir",
        str(pathlib.Path(args.log_root) / stage_name),
    ]
    if load_model_path is not None:
        command.extend(["--load-model-path", str(load_model_path)])
    return command, save_model_path


def main() -> None:
    args = parse_args()
    profiles = parse_profiles(args.profiles)
    load_model_path: pathlib.Path | None = None
    for stage_index, profile in enumerate(profiles):
        command, save_model_path = build_stage_command(args, profile, stage_index, load_model_path)
        print(f"Training curriculum stage {stage_index + 1}/{len(profiles)} against scripted Player {profile}")
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        load_model_path = save_model_path
    print(f"Completed adaptive curriculum. Final checkpoint: {load_model_path}")


if __name__ == "__main__":
    main()
