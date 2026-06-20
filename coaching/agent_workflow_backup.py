"""Minimal agentic coaching workflow scaffold.

This module demonstrates a small tool-using loop:
1) summarize gameplay stats,
2) retrieve relevant tips,
3) produce structured coaching feedback.
"""

from __future__ import annotations

import pathlib
from typing import Any

try:
    from coaching.retrieval import LocalKnowledgeRetriever
except ModuleNotFoundError:
    from retrieval import LocalKnowledgeRetriever


def summarize_gameplay_stats(gameplay_stats: dict[str, Any]) -> str:
    return (
        f"hit_rate={gameplay_stats.get('hit_rate', 0):.2f}, "
        f"survival_time={gameplay_stats.get('survival_time_sec', 0):.1f}s, "
        f"damage_taken={gameplay_stats.get('damage_taken', 0)}"
    )


def build_query_from_gameplay_stats(gameplay_stats: dict[str, Any]) -> str:
    if gameplay_stats.get("hit_rate", 0.0) < 0.2:
        return "low hit rate first-shot accuracy recoil burst control"
    if gameplay_stats.get("survival_time_sec", 0.0) < 20:
        return "dying early positioning cover peeking"
    return "decision making angle timing reset duel"


def generate_coaching_feedback(gameplay_stats: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    kb_path = pathlib.Path(__file__).resolve().parent / "knowledge" / "tips.md"
    retriever = LocalKnowledgeRetriever(str(kb_path))

    gameplay_summary = summarize_gameplay_stats(gameplay_stats)
    query = build_query_from_gameplay_stats(gameplay_stats)
    retrieved = retriever.retrieve(query=query, top_k=top_k)

    action_items = [passage.split("\n")[0].strip("- ").strip() for _, passage in retrieved]
    return {
        "gameplay_summary": gameplay_summary,
        "retrieval_query": query,
        "grounded_tips": [passage for _, passage in retrieved],
        "action_items": action_items,
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


if __name__ == "__main__":
    fake_gameplay_stats = {
        "hit_rate": 0.16,
        "survival_time_sec": 14.3,
        "damage_taken": 92,
    }
    feedback = generate_coaching_feedback(fake_gameplay_stats)
    print("Coaching feedback:")
    for key, value in feedback.items():
        print(f"- {key}: {value}")
