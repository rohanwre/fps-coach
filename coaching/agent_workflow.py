"""Agentic coaching workflow.

Pipeline:
1. Summarize gameplay stats (summary metrics + event window signals).
2. Build a retrieval query from enriched heuristics.
3. Retrieve grounded tips from the local knowledge base.
4. Produce structured coaching feedback with action items and a human-readable summary.
"""

from __future__ import annotations

import pathlib
from typing import Any

try:
    from coaching.retrieval import LocalKnowledgeRetriever
    from coaching.visual_evidence import extract_visual_evidence
except ModuleNotFoundError:
    from retrieval import LocalKnowledgeRetriever
    from visual_evidence import extract_visual_evidence


def extract_window_signals(event_windows: list[dict[str, Any]]) -> dict[str, Any]:
    total_windows = len(event_windows)
    if total_windows == 0:
        return {"damage_window_rate": 0.0, "miss_window_rate": 0.0, "rounds_with_early_death": []}

    damage_windows = 0
    miss_windows = 0
    early_death_rounds: list[int] = []
    rounds_seen: list[int] = []

    for window in event_windows:
        evidence = extract_visual_evidence(window.get("frame_paths", []), window)
        tags = evidence.get("tags", [])

        if "player_took_damage" in tags:
            damage_windows += 1
        if "player_missed_shots" in tags:
            miss_windows += 1

        round_index = window.get("round_index")
        if round_index is not None and round_index not in rounds_seen:
            rounds_seen.append(round_index)

        if (
            "player_took_damage" in tags
            and window.get("center_time_sec", 999) < 8.0
            and round_index is not None
            and round_index not in early_death_rounds
        ):
            early_death_rounds.append(round_index)

    return {
        "damage_window_rate": damage_windows / total_windows,
        "miss_window_rate": miss_windows / total_windows,
        "rounds_with_early_death": sorted(early_death_rounds),
    }


def summarize_gameplay_stats(
    gameplay_stats: dict[str, Any],
    window_signals: dict[str, Any] | None = None,
) -> str:
    window_signals = window_signals or {}
    parts = [
        f"hit_rate={gameplay_stats.get('hit_rate', 0):.2f}",
        f"win_rate={gameplay_stats.get('win_rate', 0):.2f}",
        f"enemy_hit_rate={gameplay_stats.get('enemy_hit_rate', 0):.2f}",
        f"damage_differential={gameplay_stats.get('damage_differential', 0):+.0f}",
        f"survival_time={gameplay_stats.get('survival_time_sec', 0):.1f}s",
        f"damage_taken={gameplay_stats.get('damage_taken', 0)}",
        f"reaction_delay_violation={gameplay_stats.get('reaction_delay_violation', False)}",
    ]
    if window_signals:
        parts += [
            f"damage_window_rate={window_signals.get('damage_window_rate', 0):.2f}",
            f"miss_window_rate={window_signals.get('miss_window_rate', 0):.2f}",
            f"early_death_rounds={window_signals.get('rounds_with_early_death', [])}",
        ]
    return ", ".join(parts)


def build_query_from_gameplay_stats(
    gameplay_stats: dict[str, Any],
    window_signals: dict[str, Any] | None = None,
) -> str:
    window_signals = window_signals or {}
    hit_rate = gameplay_stats.get("hit_rate", 0.0)
    win_rate = gameplay_stats.get("win_rate", 0.0)
    enemy_hit_rate = gameplay_stats.get("enemy_hit_rate", 0.0)
    damage_diff = gameplay_stats.get("damage_differential", 0)
    survival_time = gameplay_stats.get("survival_time_sec", 0.0)
    miss_window_rate = window_signals.get("miss_window_rate", 0.0)
    damage_window_rate = window_signals.get("damage_window_rate", 0.0)
    early_death_rounds = window_signals.get("rounds_with_early_death", [])

    parts: list[str] = []
    if hit_rate < 0.2 or miss_window_rate > 0.6:
        parts.append("low hit rate crosshair burst recoil accuracy first-shot control")
    if survival_time < 20 or early_death_rounds:
        parts.append("dying early positioning cover peeking")
    if enemy_hit_rate > 0.65 or damage_window_rate > 0.6:
        parts.append("cover retreat open lanes reposition peeking")
    if damage_diff < -200:
        parts.append("low-percentage fights survival unnecessary duels line of sight reset")
    if win_rate < 0.4:
        parts.append("decision making angle timing reset measurable goals")

    return " ".join(parts) if parts else "decision making angle timing reset consistency"


def build_action_items(
    gameplay_stats: dict[str, Any],
    window_signals: dict[str, Any],
    retrieved: list[tuple[float, str]],
) -> list[dict[str, str]]:
    hit_rate = gameplay_stats.get("hit_rate", 0.0)
    win_rate = gameplay_stats.get("win_rate", 0.0)
    enemy_hit_rate = gameplay_stats.get("enemy_hit_rate", 0.0)
    damage_diff = gameplay_stats.get("damage_differential", 0)
    miss_window_rate = window_signals.get("miss_window_rate", 0.0)
    damage_window_rate = window_signals.get("damage_window_rate", 0.0)
    early_deaths = window_signals.get("rounds_with_early_death", [])
    reaction_violation = gameplay_stats.get("reaction_delay_violation", False)

    signals: list[tuple[str, str]] = []

    if hit_rate < 0.2 or miss_window_rate > 0.6:
        signals.append((
            f"Hit rate is {hit_rate:.0%} and misses occurred in {miss_window_rate:.0%} of windows",
            "high",
        ))
    if enemy_hit_rate > 0.65 or damage_window_rate > 0.6:
        signals.append((
            f"Enemy landed {enemy_hit_rate:.0%} of shots; player took damage in {damage_window_rate:.0%} of windows",
            "high",
        ))
    if damage_diff < -200:
        signals.append((
            f"Damage differential is {damage_diff:+.0f} — taking more damage than dealing",
            "high" if damage_diff < -500 else "medium",
        ))
    if win_rate < 0.4:
        signals.append((
            f"Win rate is {win_rate:.0%} — losing most rounds",
            "high" if win_rate < 0.25 else "medium",
        ))
    if early_deaths:
        signals.append((f"Dying early (within 8s) in rounds {early_deaths}", "medium"))
    if reaction_violation:
        signals.append(("Reaction delay violation detected — some rounds may have unreliable data", "low"))
    if not signals:
        signals.append(("No strong weakness detected — focus on consistency", "low"))

    # Cap signals to retrieved passages to avoid empty tips; cycle retrieved if needed
    items = []
    for i, (reason, priority) in enumerate(signals):
        tip_passage = retrieved[i % len(retrieved)][1] if retrieved else ""
        tip = tip_passage.split("\n")[0].strip("- ").strip()
        items.append({"tip": tip, "priority": priority, "signal_reason": reason})
    return items


def build_summary(
    gameplay_stats: dict[str, Any],
    window_signals: dict[str, Any],
    action_items: list[dict[str, str]],
) -> str:
    win_rate = gameplay_stats.get("win_rate", 0.0)
    hit_rate = gameplay_stats.get("hit_rate", 0.0)
    damage_diff = gameplay_stats.get("damage_differential", 0)
    early_deaths = window_signals.get("rounds_with_early_death", [])

    lines: list[str] = []

    if win_rate >= 0.6:
        lines.append(f"You won {win_rate:.0%} of rounds — solid performance overall.")
    elif win_rate >= 0.4:
        lines.append(f"You won {win_rate:.0%} of rounds — room to improve consistency.")
    else:
        lines.append(f"You won {win_rate:.0%} of rounds — there are some clear areas to work on.")

    if hit_rate < 0.2:
        lines.append(f"Accuracy was low at {hit_rate:.0%} — prioritize crosshair placement and burst control.")
    elif hit_rate < 0.4:
        lines.append(f"Accuracy was {hit_rate:.0%} — decent but tightening your first shot will help.")

    if damage_diff < -200:
        lines.append(f"You took significantly more damage than you dealt ({damage_diff:+.0f}) — review positioning and trade decisions.")

    if early_deaths:
        lines.append(f"You died early in rounds {early_deaths} — consider a more patient entry.")

    high_priority = [item for item in action_items if item["priority"] == "high"]
    if high_priority:
        lines.append(f"Top priority: {high_priority[0]['tip']}")

    return " ".join(lines)


def recommend_macro_decision(
    macro_state: dict[str, Any],
    visual_evidence: dict[str, Any] | None = None,
) -> dict[str, str]:
    visual_evidence = visual_evidence or {}
    player_health = float(macro_state.get("player_health_fraction", 1.0))
    enemy_health = float(macro_state.get("enemy_health_fraction", 1.0))
    advantage = float(macro_state.get("player_health_advantage", player_health - enemy_health))
    line_of_sight = bool(macro_state.get("line_of_sight", visual_evidence.get("target_visible", False)))
    player_near_cover = bool(macro_state.get("player_near_cover", False))
    player_exposed = visual_evidence.get("player_exposed")

    if player_health <= 0.35 and line_of_sight:
        cover_phrase = "Use the nearby cover" if player_near_cover else "Break line of sight and retreat toward cover"
        return {
            "decision": "take_cover",
            "message": f"{cover_phrase} now. You are low health and still in the enemy's line of sight.",
            "retrieval_query": "take cover low health break line of sight survive",
        }
    if advantage <= -0.3:
        return {
            "decision": "retreat",
            "message": "Retreat and reset the fight. The enemy has a large health advantage.",
            "retrieval_query": "retreat reset unfavorable health disadvantage",
        }
    if enemy_health <= 0.35 and advantage >= 0.25 and line_of_sight:
        return {
            "decision": "push",
            "message": "Be aggressive while keeping cover nearby. The enemy is low and you have the advantage.",
            "retrieval_query": "push advantage enemy low controlled aggression",
        }
    if line_of_sight and player_exposed is True:
        return {
            "decision": "reposition",
            "message": "Reposition before taking another shot; you are visible without a clear advantage.",
            "retrieval_query": "reposition exposure cover timing",
        }
    if not line_of_sight:
        return {
            "decision": "hold",
            "message": "Hold cover and gather information before re-engaging.",
            "retrieval_query": "hold cover information re-engage timing",
        }
    return {
        "decision": "engage",
        "message": "Take the fight from cover and be ready to disengage if the trade turns against you.",
        "retrieval_query": "engage from cover controlled duel disengage",
    }


def generate_coaching_feedback(
    gameplay_stats: dict[str, Any],
    event_windows: list[dict[str, Any]] | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    kb_path = pathlib.Path(__file__).resolve().parent / "knowledge" / "tips.md"
    retriever = LocalKnowledgeRetriever(str(kb_path))

    window_signals = extract_window_signals(event_windows or [])
    gameplay_summary = summarize_gameplay_stats(gameplay_stats, window_signals)
    query = build_query_from_gameplay_stats(gameplay_stats, window_signals)
    retrieved = retriever.retrieve(query=query, top_k=top_k)
    action_items = build_action_items(gameplay_stats, window_signals, retrieved)
    summary = build_summary(gameplay_stats, window_signals, action_items)

    return {
        "gameplay_summary": gameplay_summary,
        "retrieval_query": query,
        "action_items": action_items,
        "grounded_tips": [passage for _, passage in retrieved],
        "summary": summary,
    }


def generate_visual_coaching_feedback(
    gameplay_stats: dict[str, Any],
    visual_evidence: dict[str, Any],
    macro_state: dict[str, Any] | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    kb_path = pathlib.Path(__file__).resolve().parent / "knowledge" / "tips.md"
    retriever = LocalKnowledgeRetriever(str(kb_path))
    tags = [str(tag) for tag in visual_evidence.get("tags", [])]
    macro_decision = recommend_macro_decision(macro_state or {}, visual_evidence)
    query_parts = [
        build_query_from_gameplay_stats(gameplay_stats),
        " ".join(tags).replace("_", " "),
        str(visual_evidence.get("recommended_focus", "")),
        macro_decision["retrieval_query"],
    ]
    retrieved = retriever.retrieve(query=" ".join(query_parts), top_k=top_k)
    action_items = [passage.split("\n")[0].strip("- ").strip() for _, passage in retrieved]
    return {
        "gameplay_summary": summarize_gameplay_stats(gameplay_stats),
        "visual_evidence": visual_evidence,
        "macro_state": macro_state or {},
        "macro_decision": macro_decision,
        "retrieval_query": " ".join(query_parts).strip(),
        "grounded_tips": [passage for _, passage in retrieved],
        "action_items": action_items,
        "primary_recommendation": macro_decision["message"],
    }


if __name__ == "__main__":
    import json

    fake_gameplay_stats = {
        "hit_rate": 0.16,
        "win_rate": 0.3,
        "enemy_hit_rate": 0.72,
        "damage_differential": -380,
        "survival_time_sec": 14.3,
        "damage_taken": 92,
        "reaction_delay_violation": False,
    }
    fake_event_windows = [
        {
            "run_id": "run_001",
            "round_index": 0,
            "center_event_type": "shot_missed",
            "center_time_sec": 5.2,
            "frame_paths": [],
            "events": [
                {"event_type": "shot_missed", "payload": {"shooter_id": "player"}},
                {"event_type": "actor_damaged", "payload": {"actor_id": "player"}},
            ],
        },
        {
            "run_id": "run_001",
            "round_index": 1,
            "center_event_type": "actor_damaged",
            "center_time_sec": 6.8,
            "frame_paths": [],
            "events": [
                {"event_type": "actor_damaged", "payload": {"actor_id": "player"}},
            ],
        },
    ]

    print("=== Heuristic coaching (post-session) ===")
    print(json.dumps(generate_coaching_feedback(fake_gameplay_stats, fake_event_windows), indent=2))

    print("\n=== Visual coaching (real-time) ===")
    fake_visual_evidence = {
        "tags": ["player_took_damage", "player_missed_shots"],
        "recommended_focus": "positioning",
        "target_visible": True,
        "player_exposed": True,
    }
    fake_macro_state = {
        "player_health_fraction": 0.3,
        "enemy_health_fraction": 0.7,
        "line_of_sight": True,
        "player_near_cover": True,
    }
    print(json.dumps(generate_visual_coaching_feedback(fake_gameplay_stats, fake_visual_evidence, fake_macro_state), indent=2))
