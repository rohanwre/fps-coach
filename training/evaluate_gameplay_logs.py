"""Compute gameplay metrics from Godot JSONL gameplay logs.

Converts raw event logs into round-level and run-level summary metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate gameplay JSONL logs.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a gameplay-log jsonl file or a directory of jsonl files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for summary json. Defaults next to input.",
    )
    parser.add_argument(
        "--include-zero-round-runs",
        action="store_true",
        help="Include runs with zero completed rounds in aggregate metrics.",
    )
    return parser.parse_args()


def iter_log_paths(input_path: pathlib.Path) -> list[pathlib.Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.jsonl"))
    raise FileNotFoundError(f"Input path not found: {input_path}")


def parse_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    enemy_hits = 0
    enemy_misses = 0
    enemy_damage_dealt = 0
    enemy_damage_taken = 0
    enemy_wins = 0
    enemy_losses = 0
    player_hits = 0
    player_misses = 0
    player_damage_dealt = 0
    player_damage_taken = 0
    player_kills = 0
    player_deaths = 0
    timeout_rounds = 0
    round_ended_count = 0
    round_start_times: dict[int, float] = {}
    round_durations: list[float] = []
    human_tracker = HumanLikenessTracker()
    ppo_difficulty_rounds: dict[str, int] = {}
    scripted_player_rounds: dict[str, int] = {}
    ppo_difficulty_transitions: list[dict[str, Any]] = []

    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload", {})
        unix_time_msec = float(event.get("unix_time_msec", 0.0))
        human_tracker.observe_event(event_type, payload, unix_time_msec)

        if event_type == "round_started":
            round_index = int(payload.get("round_index", -1))
            round_start_times[round_index] = unix_time_msec
            ppo_difficulty = str(payload.get("active_enemy_ppo_difficulty", "unknown"))
            scripted_player = str(payload.get("active_scripted_player_profile", "unknown"))
            ppo_difficulty_rounds[ppo_difficulty] = ppo_difficulty_rounds.get(ppo_difficulty, 0) + 1
            scripted_player_rounds[scripted_player] = scripted_player_rounds.get(scripted_player, 0) + 1
        elif event_type == "enemy_ppo_difficulty_changed":
            ppo_difficulty_transitions.append(
                {
                    "round_index": int(payload.get("round_index", -1)),
                    "previous_profile": str(payload.get("previous_profile", "unknown")),
                    "active_profile": str(payload.get("active_profile", "unknown")),
                    "player_trend": str(payload.get("player_trend", "unknown")),
                    "difficulty_config": payload.get("difficulty_config", {}),
                }
            )
        elif event_type == "round_ended":
            round_ended_count += 1
            round_index = int(payload.get("round_index", -1))
            killer_id = payload.get("killer_id")
            dead_actor_id = payload.get("dead_actor_id")
            end_reason = payload.get("end_reason")
            if killer_id == "enemy":
                enemy_wins += 1
            if dead_actor_id == "enemy":
                enemy_losses += 1
            if killer_id == "player":
                player_kills += 1
            if dead_actor_id == "player":
                player_deaths += 1
            if end_reason == "timeout":
                timeout_rounds += 1
            if round_index in round_start_times:
                round_durations.append(max(0.0, (unix_time_msec - round_start_times[round_index]) / 1000.0))
        elif event_type == "shot_hit":
            if payload.get("shooter_id") == "enemy":
                enemy_hits += 1
            if payload.get("shooter_id") == "player":
                player_hits += 1
        elif event_type == "shot_missed":
            if payload.get("shooter_id") == "enemy":
                enemy_misses += 1
            if payload.get("shooter_id") == "player":
                player_misses += 1
        elif event_type == "actor_damaged":
            actor_id = payload.get("actor_id")
            source_id = payload.get("source_id")
            amount = int(payload.get("amount", 0))
            if actor_id == "player" and source_id == "enemy":
                enemy_damage_dealt += amount
                player_damage_taken += amount
            if actor_id == "enemy" and source_id == "player":
                enemy_damage_taken += amount
                player_damage_dealt += amount

    total_enemy_shots = enemy_hits + enemy_misses
    total_outcomes = enemy_wins + enemy_losses
    player_skill = build_player_skill_summary(
        player_hits,
        player_misses,
        player_damage_dealt,
        player_damage_taken,
        player_kills,
        player_deaths,
        round_durations,
    )

    return {
        "enemy_wins": enemy_wins,
        "enemy_losses": enemy_losses,
        "timeout_rounds": timeout_rounds,
        "enemy_win_rate": (enemy_wins / total_outcomes) if total_outcomes > 0 else 0.0,
        "enemy_hits": enemy_hits,
        "enemy_misses": enemy_misses,
        "enemy_total_shots": total_enemy_shots,
        "enemy_hit_rate": (enemy_hits / total_enemy_shots) if total_enemy_shots > 0 else 0.0,
        "enemy_damage_dealt": enemy_damage_dealt,
        "enemy_damage_taken": enemy_damage_taken,
        "enemy_damage_diff": enemy_damage_dealt - enemy_damage_taken,
        "round_count": round_ended_count,
        "mean_round_duration_sec": statistics.fmean(round_durations) if round_durations else 0.0,
        "human_likeness": human_tracker.build_summary(),
        "player_skill": player_skill,
        "adaptive_difficulty": {
            "ppo_difficulty_rounds": dict(sorted(ppo_difficulty_rounds.items())),
            "scripted_player_rounds": dict(sorted(scripted_player_rounds.items())),
            "transition_count": len(ppo_difficulty_transitions),
            "transitions": ppo_difficulty_transitions,
        },
    }


def build_player_skill_summary(
    player_hits: int,
    player_misses: int,
    player_damage_dealt: int,
    player_damage_taken: int,
    player_kills: int,
    player_deaths: int,
    round_durations: list[float],
) -> dict[str, Any]:
    total_player_shots = player_hits + player_misses
    hit_rate = (player_hits / total_player_shots) if total_player_shots > 0 else 0.0
    damage_diff = player_damage_dealt - player_damage_taken
    return {
        "hits": player_hits,
        "misses": player_misses,
        "total_shots": total_player_shots,
        "hit_rate": hit_rate,
        "damage_dealt": player_damage_dealt,
        "damage_taken": player_damage_taken,
        "damage_diff": damage_diff,
        "kills": player_kills,
        "deaths": player_deaths,
        "survival_time_mean_sec": statistics.fmean(round_durations) if round_durations else 0.0,
        "recent_trend": classify_player_trend(hit_rate, damage_diff, player_kills, player_deaths),
    }


def classify_player_trend(hit_rate: float, damage_diff: int, player_kills: int, player_deaths: int) -> str:
    if player_kills > player_deaths or (damage_diff > 50 and hit_rate >= 0.25):
        return "improving"
    if player_deaths > player_kills or (damage_diff < -50 and hit_rate < 0.2):
        return "struggling"
    return "stable"


def _vec3_from_payload(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        return (
            float(value.get("x", 0.0)),
            float(value.get("y", 0.0)),
            float(value.get("z", 0.0)),
        )
    except (TypeError, ValueError):
        return None


def _horizontal_speed(value: tuple[float, float, float] | None) -> float:
    if value is None:
        return 0.0
    return math.hypot(value[0], value[2])


def _horizontal_direction(value: tuple[float, float, float] | None) -> tuple[float, float] | None:
    speed = _horizontal_speed(value)
    if value is None or speed <= 0.1:
        return None
    return (value[0] / speed, value[2] / speed)


class HumanLikenessTracker:
    """Accumulates enemy behavior metrics from gameplay-log events."""

    def __init__(self) -> None:
        self.reaction_delays_sec: list[float] = []
        self.aim_alignments: list[float] = []
        self.aim_alignment_deltas: list[float] = []
        self.snap_to_target_count = 0
        self.enemy_shot_count = 0
        self.blocked_shot_count = 0
        self.blocked_shots_by_reason: dict[str, int] = {}
        self.shots_without_line_of_sight = 0
        self.movement_direction_changes = 0
        self.stationary_sample_count = 0
        self.behavior_sample_count = 0
        self.first_event_time_sec: float | None = None
        self.last_event_time_sec: float | None = None

        self._line_of_sight_start_sec: float | None = None
        self._previous_aim_alignment: float | None = None
        self._previous_move_direction: tuple[float, float] | None = None
        self._last_line_of_sight = False

    def observe_event(self, event_type: str, payload: dict[str, Any], unix_time_msec: float) -> None:
        event_time_sec = unix_time_msec / 1000.0
        self._mark_event_time(event_time_sec)

        if event_type == "shot_fired" and payload.get("shooter_id") == "enemy":
            self.enemy_shot_count += 1
            return

        if event_type == "enemy_shot_blocked":
            self.blocked_shot_count += 1
            reason = str(payload.get("reason", "unknown"))
            self.blocked_shots_by_reason[reason] = self.blocked_shots_by_reason.get(reason, 0) + 1
            return

        if event_type == "enemy_behavior_sample":
            self._observe_behavior_sample(payload, event_time_sec)

    def build_summary(self) -> dict[str, Any]:
        duration_sec = self._duration_sec()
        summary = {
            "reaction_delay_mean_sec": (
                statistics.fmean(self.reaction_delays_sec) if self.reaction_delays_sec else 0.0
            ),
            "reaction_delay_min_sec": min(self.reaction_delays_sec) if self.reaction_delays_sec else 0.0,
            "reaction_delay_median_sec": (
                statistics.median(self.reaction_delays_sec) if self.reaction_delays_sec else 0.0
            ),
            "aim_alignment_mean": statistics.fmean(self.aim_alignments) if self.aim_alignments else 0.0,
            "aim_alignment_std": statistics.stdev(self.aim_alignments) if len(self.aim_alignments) > 1 else 0.0,
            "aim_alignment_delta_mean": (
                statistics.fmean(self.aim_alignment_deltas) if self.aim_alignment_deltas else 0.0
            ),
            "aim_alignment_delta_max": max(self.aim_alignment_deltas) if self.aim_alignment_deltas else 0.0,
            "snap_to_target_count": self.snap_to_target_count,
            "shots_per_second": self.enemy_shot_count / duration_sec if duration_sec > 0 else 0.0,
            "blocked_shot_count": self.blocked_shot_count,
            "blocked_shots_per_second": self.blocked_shot_count / duration_sec if duration_sec > 0 else 0.0,
            "blocked_shots_by_reason": dict(sorted(self.blocked_shots_by_reason.items())),
            "movement_direction_changes_per_sec": (
                self.movement_direction_changes / duration_sec if duration_sec > 0 else 0.0
            ),
            "stationary_fraction": (
                self.stationary_sample_count / self.behavior_sample_count
                if self.behavior_sample_count > 0
                else 0.0
            ),
            "shots_without_line_of_sight": self.shots_without_line_of_sight,
            "behavior_sample_count": self.behavior_sample_count,
        }
        summary["flags"] = self._build_flags(summary)
        return summary

    def _build_flags(self, summary: dict[str, Any]) -> dict[str, bool]:
        return {
            "reaction_delay_violation": (
                summary["reaction_delay_min_sec"] > 0.0 and summary["reaction_delay_min_sec"] < 0.15
            ),
            "line_of_sight_violation": summary["shots_without_line_of_sight"] > 0,
            "aim_snap_violation": summary["snap_to_target_count"] > 0,
            "jitter_warning": summary["movement_direction_changes_per_sec"] > 4.0,
            "too_stationary_warning": summary["stationary_fraction"] > 0.5,
            "low_engagement_warning": summary["shots_per_second"] < 0.05 and self.behavior_sample_count > 0,
        }

    def _observe_behavior_sample(self, payload: dict[str, Any], event_time_sec: float) -> None:
        self.behavior_sample_count += 1

        line_of_sight = bool(payload.get("line_of_sight", False))
        weapon_ready = bool(payload.get("weapon_ready", False))
        sample_reason = str(payload.get("sample_reason", ""))
        self._last_line_of_sight = line_of_sight

        if line_of_sight and weapon_ready and self._line_of_sight_start_sec is None:
            self._line_of_sight_start_sec = event_time_sec
        elif not line_of_sight:
            self._line_of_sight_start_sec = None

        if sample_reason == "shot_fired" and "line_of_sight_elapsed" in payload:
            self.reaction_delays_sec.append(max(0.0, float(payload.get("line_of_sight_elapsed", 0.0))))
            self._line_of_sight_start_sec = None
        elif sample_reason == "shot_fired" and self._line_of_sight_start_sec is not None:
            self.reaction_delays_sec.append(max(0.0, event_time_sec - self._line_of_sight_start_sec))
            self._line_of_sight_start_sec = None
        elif sample_reason == "shot_fired" and not line_of_sight:
            self.shots_without_line_of_sight += 1

        aim_alignment = float(payload.get("aim_alignment", 0.0))
        self.aim_alignments.append(aim_alignment)
        if self._previous_aim_alignment is not None:
            aim_delta = abs(aim_alignment - self._previous_aim_alignment)
            self.aim_alignment_deltas.append(aim_delta)
            if self._previous_aim_alignment < 0.3 and aim_alignment > 0.9 and aim_delta > 0.6:
                self.snap_to_target_count += 1
        self._previous_aim_alignment = aim_alignment

        velocity = _vec3_from_payload(payload.get("enemy_velocity"))
        speed = _horizontal_speed(velocity)
        if speed <= 0.1:
            self.stationary_sample_count += 1

        move_direction = _horizontal_direction(velocity)
        if move_direction is not None and self._previous_move_direction is not None:
            dot = (
                move_direction[0] * self._previous_move_direction[0]
                + move_direction[1] * self._previous_move_direction[1]
            )
            if dot < 0.5:
                self.movement_direction_changes += 1
        if move_direction is not None:
            self._previous_move_direction = move_direction

    def _mark_event_time(self, event_time_sec: float) -> None:
        if self.first_event_time_sec is None:
            self.first_event_time_sec = event_time_sec
        self.last_event_time_sec = event_time_sec

    def _duration_sec(self) -> float:
        if self.first_event_time_sec is None or self.last_event_time_sec is None:
            return 0.0
        return max(0.0, self.last_event_time_sec - self.first_event_time_sec)


def main() -> None:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    log_paths = iter_log_paths(input_path)
    if not log_paths:
        raise SystemExit("No gameplay log files found.")

    per_run: list[dict[str, Any]] = []
    for path in log_paths:
        metrics = summarize_events(parse_jsonl(path))
        per_run.append({"path": str(path), "metrics": metrics})

    if args.include_zero_round_runs:
        aggregate_runs = per_run
    else:
        aggregate_runs = [entry for entry in per_run if int(entry["metrics"]["round_count"]) > 0]

    win_rates = [entry["metrics"]["enemy_win_rate"] for entry in aggregate_runs]
    hit_rates = [entry["metrics"]["enemy_hit_rate"] for entry in aggregate_runs]
    damage_diffs = [entry["metrics"]["enemy_damage_diff"] for entry in aggregate_runs]
    player_hit_rates = [entry["metrics"]["player_skill"]["hit_rate"] for entry in aggregate_runs]
    player_damage_diffs = [entry["metrics"]["player_skill"]["damage_diff"] for entry in aggregate_runs]

    summary = {
        "num_logs": len(per_run),
        "num_aggregate_logs": len(aggregate_runs),
        "excluded_zero_round_runs": len(per_run) - len(aggregate_runs),
        "aggregate": {
            "enemy_win_rate_mean": statistics.fmean(win_rates) if win_rates else 0.0,
            "enemy_hit_rate_mean": statistics.fmean(hit_rates) if hit_rates else 0.0,
            "enemy_damage_diff_mean": statistics.fmean(damage_diffs) if damage_diffs else 0.0,
            "player_hit_rate_mean": statistics.fmean(player_hit_rates) if player_hit_rates else 0.0,
            "player_damage_diff_mean": statistics.fmean(player_damage_diffs) if player_damage_diffs else 0.0,
            "enemy_win_rate_std": statistics.stdev(win_rates) if len(win_rates) > 1 else 0.0,
            "includes_zero_round_runs": args.include_zero_round_runs,
        },
        "per_run": per_run,
    }

    if args.output is not None:
        output_path = pathlib.Path(args.output)
    elif input_path.is_file():
        output_path = input_path.with_name(f"{input_path.stem}_metrics.json")
    else:
        output_path = input_path / "gameplay_log_metrics_summary.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote gameplay metrics to {output_path}")


if __name__ == "__main__":
    main()
