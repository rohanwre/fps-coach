"""Compare gameplay metrics JSON files.

Builds a compact Markdown table from outputs created by
training/evaluate_gameplay_logs.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare gameplay metrics files.")
    parser.add_argument("metrics_paths", nargs="+", help="One or more metrics JSON files.")
    parser.add_argument("--output", default=None, help="Optional Markdown output path.")
    return parser.parse_args()


def load_metrics(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_run_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    per_run = summary.get("per_run", [])
    if not per_run:
        raise ValueError("Metrics summary has no per_run entries.")
    return per_run[0].get("metrics", {})


def label_for_path(path: pathlib.Path) -> str:
    name = path.stem
    suffix = "_gameplay_metrics"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def build_rows(paths: list[pathlib.Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        metrics = first_run_metrics(load_metrics(path))
        human = metrics.get("human_likeness", {})
        player = metrics.get("player_skill", {})
        flags = human.get("flags", {})
        rows.append(
            {
                "run": label_for_path(path),
                "rounds": metrics.get("round_count", 0),
                "timeouts": metrics.get("timeout_rounds", 0),
                "hit_rate": metrics.get("enemy_hit_rate", 0.0),
                "damage_diff": metrics.get("enemy_damage_diff", 0),
                "player_hit_rate": player.get("hit_rate", 0.0),
                "player_damage_diff": player.get("damage_diff", 0),
                "player_trend": player.get("recent_trend", "unknown"),
                "reaction_min": human.get("reaction_delay_min_sec", 0.0),
                "reaction_mean": human.get("reaction_delay_mean_sec", 0.0),
                "shots_sec": human.get("shots_per_second", 0.0),
                "blocked": human.get("blocked_shot_count", 0),
                "move_changes_sec": human.get("movement_direction_changes_per_sec", 0.0),
                "snap_count": human.get("snap_to_target_count", 0),
                "flags": ",".join(flag for flag, active in sorted(flags.items()) if active) or "none",
            }
        )
    return rows


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def build_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run",
        "rounds",
        "timeouts",
        "hit_rate",
        "damage_diff",
        "player_hit_rate",
        "player_damage_diff",
        "player_trend",
        "reaction_min",
        "reaction_mean",
        "shots_sec",
        "blocked",
        "move_changes_sec",
        "snap_count",
        "flags",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run"]),
                    str(row["rounds"]),
                    str(row["timeouts"]),
                    format_float(row["hit_rate"]),
                    str(row["damage_diff"]),
                    format_float(row["player_hit_rate"]),
                    str(row["player_damage_diff"]),
                    str(row["player_trend"]),
                    format_float(row["reaction_min"]),
                    format_float(row["reaction_mean"]),
                    format_float(row["shots_sec"]),
                    str(row["blocked"]),
                    format_float(row["move_changes_sec"]),
                    str(row["snap_count"]),
                    str(row["flags"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    paths = [pathlib.Path(value) for value in args.metrics_paths]
    markdown = build_markdown(build_rows(paths))

    if args.output is None:
        print(markdown)
        return

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote comparison report to {output_path}")


if __name__ == "__main__":
    main()
