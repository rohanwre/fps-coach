"""Benchmark one Enemy PPO checkpoint under fixed easy/medium/hard execution tiers."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from argparse import Namespace

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.benchmark_enemy_policies import benchmark_policy, flatten_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Enemy PPO execution tiers against one scripted Player.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--profiles", default="easy,medium,hard")
    parser.add_argument("--scripted-player-profile", default="easy")
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--stochastic-actions", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speedup", type=int, default=4)
    parser.add_argument("--start-port", type=int, default=11700)
    parser.add_argument("--godot-start-delay", type=float, default=4.0)
    parser.add_argument("--godot-stop-timeout", type=float, default=20.0)
    parser.add_argument("--godot-path", required=True)
    parser.add_argument("--project-path", default=str(REPO_ROOT / "godot_project"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "training" / "benchmarks" / "enemy_difficulty_tiers"))
    return parser.parse_args()


def parse_profiles(value: str) -> list[str]:
    profiles = [profile.strip() for profile in value.split(",") if profile.strip()]
    invalid = [profile for profile in profiles if profile not in {"easy", "medium", "hard"}]
    if invalid:
        raise ValueError(f"Unknown fixed PPO difficulty profiles: {', '.join(invalid)}")
    return profiles


def build_profile_args(args: argparse.Namespace, profile: str) -> Namespace:
    values = vars(args).copy()
    values["ppo_difficulty_profile"] = profile
    return Namespace(**values)


def write_results(args: argparse.Namespace, results: list[dict], output_dir: pathlib.Path) -> None:
    table = []
    for result in results:
        row = flatten_result(result)
        row["difficulty"] = row.pop("policy")
        table.append(row)
    payload = {
        "protocol": {
            "model_path": str(pathlib.Path(args.model_path).resolve().relative_to(REPO_ROOT)),
            "scripted_player_profile": args.scripted_player_profile,
            "difficulty_profiles": parse_profiles(args.profiles),
            "steps_per_difficulty": args.steps,
            "speedup": args.speedup,
            "deterministic_actions": not args.stochastic_actions,
            "seed": args.seed,
        },
        "results": results,
        "table": table,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output_dir / "benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        for row in table:
            csv_row = row.copy()
            csv_row["human_likeness_flags"] = json.dumps(csv_row["human_likeness_flags"], separators=(",", ":"))
            writer.writerow(csv_row)


def main() -> None:
    args = parse_args()
    model_path = pathlib.Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    output_dir = pathlib.Path(args.output_dir).resolve()
    results = []
    for index, profile in enumerate(parse_profiles(args.profiles)):
        print(f"Benchmarking Enemy PPO {profile} difficulty against scripted {args.scripted_player_profile} Player")
        results.append(
            benchmark_policy(
                build_profile_args(args, profile),
                profile,
                model_path,
                args.start_port + index,
                output_dir,
            )
        )
    write_results(args, results, output_dir)
    print(f"Wrote Enemy PPO difficulty benchmark to {output_dir}")


if __name__ == "__main__":
    main()
