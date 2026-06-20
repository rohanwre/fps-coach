"""Run VLM-assisted coaching over event-window JSONL data."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

try:
    from coaching.agent_workflow import generate_visual_coaching_feedback
    from coaching.visual_evidence import OpenAICompatibleVLM, VisualEvidenceProvider, extract_visual_evidence
except ModuleNotFoundError:
    from agent_workflow import generate_visual_coaching_feedback
    from visual_evidence import OpenAICompatibleVLM, VisualEvidenceProvider, extract_visual_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate coaching from event windows and player frames.")
    parser.add_argument("--input", required=True, help="Event-window JSONL path.")
    parser.add_argument("--output", required=True, help="Output coaching JSONL path.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--include-event-context",
        action="store_true",
        help="Tell the VLM the center gameplay event type; useful only for context-bias ablations.",
    )
    return parser.parse_args()


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate_visual_coaching_rows(
    windows: list[dict[str, Any]],
    limit: int,
    top_k: int,
    provider: VisualEvidenceProvider | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in windows[: max(limit, 0)]:
        frame_paths = [str(path) for path in window.get("frame_paths", [])]
        evidence = extract_visual_evidence(frame_paths, window, provider=provider)
        stats = _stats_from_window(window)
        macro_state = _macro_state_for_window(window)
        rows.append(
            {
                "run_id": window.get("run_id", ""),
                "round_index": window.get("round_index"),
                "center_event_type": window.get("center_event_type", ""),
                "center_event_index": window.get("center_event_index"),
                "frame_paths": frame_paths,
                "analyzed_frame_path": evidence.get("analyzed_frame_path", frame_paths[-1] if frame_paths else None),
                "coaching": generate_visual_coaching_feedback(stats, evidence, macro_state=macro_state, top_k=top_k),
            }
        )
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _stats_from_window(window: dict[str, Any]) -> dict[str, Any]:
    events = window.get("events", [])
    hits = sum(
        event.get("event_type") == "shot_hit" and event.get("payload", {}).get("shooter_id") == "player"
        for event in events
    )
    misses = sum(
        event.get("event_type") == "shot_missed" and event.get("payload", {}).get("shooter_id") == "player"
        for event in events
    )
    damage_taken = sum(
        int(event.get("payload", {}).get("amount", 0))
        for event in events
        if event.get("event_type") == "actor_damaged"
        and event.get("payload", {}).get("actor_id") == "player"
    )
    shots = hits + misses
    return {
        "hit_rate": hits / shots if shots else 0.0,
        "survival_time_sec": float(window.get("window_sec", 0.0)) * 2.0,
        "damage_taken": damage_taken,
    }


def _macro_state_for_window(window: dict[str, Any]) -> dict[str, Any]:
    center_event_index = window.get("center_event_index")
    states = [
        event
        for event in window.get("events", [])
        if event.get("event_type") == "macro_state"
    ]
    if not states:
        return {}
    if isinstance(center_event_index, int):
        closest = min(states, key=lambda event: abs(int(event.get("event_index", 0)) - center_event_index))
        return dict(closest.get("payload", {}))
    return dict(states[-1].get("payload", {}))


def main() -> None:
    args = parse_args()
    provider = OpenAICompatibleVLM.from_environment(include_event_context=True) if args.include_event_context else None
    rows = generate_visual_coaching_rows(
        load_jsonl(pathlib.Path(args.input)),
        args.limit,
        args.top_k,
        provider=provider,
    )
    write_jsonl(pathlib.Path(args.output), rows)
    print(f"Wrote {len(rows)} visual coaching rows to {args.output}")


if __name__ == "__main__":
    main()
