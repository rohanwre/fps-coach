"""Build coaching event windows from gameplay logs.

The output is a JSONL dataset where each row contains one important gameplay
event, nearby same-round context events, and matching captured frame paths.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any


KEY_EVENT_TYPES = {"actor_damaged", "macro_state", "round_ended", "shot_missed", "shot_hit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build coaching event windows from JSONL logs.")
    parser.add_argument("--input", required=True, help="Gameplay JSONL log path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--window-sec", type=float, default=3.0)
    return parser.parse_args()


def parse_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def event_time_sec(event: dict[str, Any]) -> float:
    return float(event.get("unix_time_msec", 0.0)) / 1000.0


def is_key_event(event: dict[str, Any]) -> bool:
    event_type = str(event.get("event_type", ""))
    if event_type not in KEY_EVENT_TYPES:
        return False
    payload = event.get("payload", {})
    if event_type == "shot_missed":
        return payload.get("shooter_id") == "player"
    if event_type == "shot_hit":
        return payload.get("shooter_id") == "player"
    if event_type == "actor_damaged":
        return payload.get("actor_id") == "player" or payload.get("actor_id") == "enemy"
    if event_type == "macro_state":
        reason = str(payload.get("reason", ""))
        return reason.startswith("line_of_sight_") or reason == "tactical_interval"
    return True


def event_round_index(event: dict[str, Any]) -> Any:
    return event.get("payload", {}).get("round_index")


def build_event_windows(events: list[dict[str, Any]], window_sec: float = 3.0) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for event in events:
        if not is_key_event(event):
            continue

        center_time = event_time_sec(event)
        center_round_index = event_round_index(event)
        context = [
            candidate
            for candidate in events
            if abs(event_time_sec(candidate) - center_time) <= window_sec
            and (
                center_round_index is None
                or event_round_index(candidate) is None
                or event_round_index(candidate) == center_round_index
            )
        ]
        frame_events = [
            candidate
            for candidate in context
            if candidate.get("event_type") == "player_frame_captured"
            and candidate.get("payload", {}).get("frame_path")
        ]
        # The VLM treats the final frame as authoritative. Order nearby frames
        # from farthest to closest so its final image matches the center event.
        frame_events.sort(key=lambda candidate: abs(event_time_sec(candidate) - center_time), reverse=True)
        frame_paths = [str(candidate["payload"]["frame_path"]) for candidate in frame_events]
        payload = event.get("payload", {})
        windows.append(
            {
                "run_id": event.get("run_id", ""),
                "round_index": center_round_index,
                "center_event_type": event.get("event_type"),
                "center_event_index": event.get("event_index"),
                "center_time_sec": center_time,
                "window_sec": window_sec,
                "events": context,
                "frame_paths": frame_paths,
            }
        )
    return windows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    windows = build_event_windows(parse_jsonl(pathlib.Path(args.input)), args.window_sec)
    write_jsonl(pathlib.Path(args.output), windows)
    print(f"Wrote {len(windows)} coaching event windows to {args.output}")


if __name__ == "__main__":
    main()
