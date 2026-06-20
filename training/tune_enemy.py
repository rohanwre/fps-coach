"""Run a small PPO tuning grid and keep the best gameplay agent.

This script launches sequential runs with different hyperparameter settings,
stores metadata for each run, and writes a best-run summary JSON.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import statistics
from argparse import Namespace
from typing import Any

from train_enemy import REPO_ROOT, train_once


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_config_key(
    learning_rate: float,
    gamma: float,
    ent_coef: float,
    n_steps: int,
    batch_size: int,
) -> str:
    return (
        f"lr{learning_rate:g}_g{gamma:g}_ent{ent_coef:g}_"
        f"n{n_steps}_b{batch_size}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for the enemy PPO gameplay agent.")
    parser.add_argument("--env-path", default=None)
    parser.add_argument("--timesteps", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed list (overrides --seed when provided).",
    )
    parser.add_argument("--start-port", type=int, default=11008)
    parser.add_argument(
        "--fixed-port",
        type=int,
        default=None,
        help="Reuse one port for every run (useful for in-editor Play loops).",
    )
    parser.add_argument("--speedup", type=int, default=4)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--agent-role", choices=("enemy", "player"), default="enemy")
    parser.add_argument("--scripted-enemy-profile", default="hard")
    parser.add_argument("--scripted-player-profile", default="medium")
    parser.add_argument("--ppo-difficulty-profile", default="adaptive")
    parser.add_argument("--reward-profile", default="default")
    parser.add_argument("--gameplay-log-dir", default=None)
    parser.add_argument("--learning-rates", default="3e-4,1e-4")
    parser.add_argument("--gammas", default="0.99,0.995")
    parser.add_argument("--ent-coefs", default="0.0,0.01")
    parser.add_argument("--n-steps-list", default="64,128")
    parser.add_argument("--batch-size-list", default="64")
    parser.add_argument("--experiment-dir", default=str(REPO_ROOT / "training" / "logs"))
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "training" / "models"))
    parser.add_argument("--results-path", default=str(REPO_ROOT / "training" / "models" / "enemy_tuning_results.json"))
    parser.add_argument("--viz", action="store_true")
    return parser.parse_args()


def build_run_args(
    run_name: str,
    port: int,
    args: argparse.Namespace,
    learning_rate: float,
    gamma: float,
    ent_coef: float,
    n_steps: int,
    batch_size: int,
) -> Namespace:
    model_dir = pathlib.Path(args.model_dir)
    metadata_dir = model_dir / "metadata"
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return Namespace(
        env_path=args.env_path,
        timesteps=args.timesteps,
        seed=args.seed,
        port=port,
        n_parallel=args.n_parallel,
        speedup=args.speedup,
        action_repeat=args.action_repeat,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=0.95,
        ent_coef=ent_coef,
        clip_range=0.2,
        experiment_dir=args.experiment_dir,
        experiment_name=run_name,
        save_model_path=str(model_dir / f"{run_name}.zip"),
        load_model_path=None,
        save_metadata_path=str(metadata_dir / f"{run_name}.json"),
        eval_episodes=args.eval_episodes,
        reward_profile=args.reward_profile,
        agent_role=args.agent_role,
        scripted_enemy_profile=args.scripted_enemy_profile,
        scripted_player_profile=args.scripted_player_profile,
        ppo_difficulty_profile=args.ppo_difficulty_profile,
        gameplay_log_dir=args.gameplay_log_dir,
        capture_coaching_frames=False,
        coaching_frame_dir=None,
        checkpoint_frequency=0,
        viz=args.viz,
    )


def main() -> None:
    args = parse_args()
    learning_rates = parse_float_list(args.learning_rates)
    gammas = parse_float_list(args.gammas)
    ent_coefs = parse_float_list(args.ent_coefs)
    n_steps_list = parse_int_list(args.n_steps_list)
    batch_size_list = parse_int_list(args.batch_size_list)
    if args.seeds is not None and args.seeds.strip():
        seed_list = parse_int_list(args.seeds)
    else:
        seed_list = [args.seed]

    tuning_grid = list(itertools.product(learning_rates, gammas, ent_coefs, n_steps_list, batch_size_list))
    all_results: list[dict[str, Any]] = []
    config_groups: dict[str, list[dict[str, Any]]] = {}
    total_runs = len(tuning_grid) * len(seed_list)
    run_counter = 0

    for config_index, (learning_rate, gamma, ent_coef, n_steps, batch_size) in enumerate(tuning_grid):
        config_key = build_config_key(learning_rate, gamma, ent_coef, n_steps, batch_size)
        for seed_index, seed in enumerate(seed_list):
            run_name = (
                f"enemy_ppo_{config_key}_seed{seed}_"
                f"run{config_index:02d}_{seed_index:02d}"
            )
            run_counter += 1
            run_port = args.fixed_port if args.fixed_port is not None else args.start_port + run_counter - 1
            run_args = build_run_args(
                run_name=run_name,
                port=run_port,
                args=args,
                learning_rate=learning_rate,
                gamma=gamma,
                ent_coef=ent_coef,
                n_steps=n_steps,
                batch_size=batch_size,
            )
            run_args.seed = seed
            print(f"[{run_counter}/{total_runs}] Training {run_name} on port {run_port}")
            result = train_once(run_args)
            result["config_key"] = config_key
            all_results.append(result)
            config_groups.setdefault(config_key, []).append(result)
            print(
                "  mean_reward="
                f"{result['mean_reward']:.3f} "
                f"std_reward={result['std_reward']:.3f}"
            )

    if not all_results:
        raise SystemExit("No runs were generated. Check tuning arguments.")

    config_summaries: list[dict[str, Any]] = []
    for config_key, config_runs in config_groups.items():
        rewards = [float(run["mean_reward"]) for run in config_runs]
        summary = {
            "config_key": config_key,
            "num_runs": len(config_runs),
            "mean_reward_avg": statistics.fmean(rewards),
            "mean_reward_std": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            "best_run": max(config_runs, key=lambda item: float(item["mean_reward"])),
        }
        config_summaries.append(summary)

    best_result = max(all_results, key=lambda item: float(item["mean_reward"]))
    best_config = max(config_summaries, key=lambda item: float(item["mean_reward_avg"]))
    output = {
        "total_runs": len(all_results),
        "seed_list": seed_list,
        "num_configs": len(tuning_grid),
        "best_config": best_config,
        "best_run": best_result,
        "config_summaries": config_summaries,
        "all_runs": all_results,
    }

    results_path = pathlib.Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote tuning summary to {results_path}")
    print(
        "Best model: "
        f"{best_result['save_model_path']} "
        f"(mean_reward={best_result['mean_reward']:.3f})"
    )


if __name__ == "__main__":
    main()
