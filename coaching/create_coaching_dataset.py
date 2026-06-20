"""Generate a labeled coaching dataset for few-shot prompting and validation.

Produces 15 scenarios matching the VLM visual evidence schema, splits them
10 few-shot / 5 validation, and writes to coaching/dataset/.
"""

from __future__ import annotations

import json
import pathlib
import random

random.seed(7)

SCENARIOS = [
    {
        "scenario_id": "s001",
        "description": "Player at 30% health standing in open with enemy visible and shooting",
        "game_state": {"player_health_fraction": 0.30, "enemy_health_fraction": 0.65, "line_of_sight": True, "player_near_cover": False},
        "visual_evidence": {
            "target_visible": True, "crosshair_near_target": False, "player_exposed": True,
            "tags": ["player_took_damage", "player_exposed", "low_health"],
            "summary": "Player is exposed at low health with enemy actively firing.",
            "recommended_focus": "Retreat to cover before the enemy finishes the trade.",
            "confidence": 0.90,
        },
        "macro_decision": "take_cover",
        "coaching_advice": "You are low health and fully in the open — get to cover before re-engaging. This fight is not winnable from here.",
        "priority": "high",
    },
    {
        "scenario_id": "s002",
        "description": "Player at 85% health behind cover, enemy at 15% health and visible",
        "game_state": {"player_health_fraction": 0.85, "enemy_health_fraction": 0.15, "line_of_sight": True, "player_near_cover": True},
        "visual_evidence": {
            "target_visible": True, "crosshair_near_target": True, "player_exposed": False,
            "tags": ["cover_used", "enemy_low_health", "crosshair_on_target", "health_advantage"],
            "summary": "Player has cover and crosshair on a low-health enemy.",
            "recommended_focus": "Push the advantage now while the enemy is low.",
            "confidence": 0.91,
        },
        "macro_decision": "push",
        "coaching_advice": "Enemy is nearly dead and you have the health advantage from cover — push now and finish the round.",
        "priority": "medium",
    },
    {
        "scenario_id": "s003",
        "description": "Player sprayed multiple shots, all missed, enemy moved behind cover",
        "game_state": {"player_health_fraction": 0.55, "enemy_health_fraction": 0.55, "line_of_sight": False, "player_near_cover": False},
        "visual_evidence": {
            "target_visible": False, "crosshair_near_target": None, "player_exposed": True,
            "tags": ["player_missed_shots", "no_line_of_sight", "player_exposed"],
            "summary": "Player standing exposed after missing several shots with no target visible.",
            "recommended_focus": "Move to cover and reset crosshair placement before peeking again.",
            "confidence": 0.84,
        },
        "macro_decision": "reposition",
        "coaching_advice": "You missed your shots and gave away your position — don't stay in the open. Get to cover and reset before peeking again.",
        "priority": "high",
    },
    {
        "scenario_id": "s004",
        "description": "Player behind cover with even health, enemy location unknown",
        "game_state": {"player_health_fraction": 0.60, "enemy_health_fraction": 0.60, "line_of_sight": False, "player_near_cover": True},
        "visual_evidence": {
            "target_visible": False, "crosshair_near_target": None, "player_exposed": False,
            "tags": ["cover_used", "no_line_of_sight", "health_even"],
            "summary": "Player in cover with even health and no enemy contact.",
            "recommended_focus": "Hold position and wait for the enemy to reveal themselves.",
            "confidence": 0.79,
        },
        "macro_decision": "hold",
        "coaching_advice": "You have cover and an even health state — hold your position and force the enemy to come to you.",
        "priority": "low",
    },
    {
        "scenario_id": "s005",
        "description": "Enemy clearly visible but crosshair pointed at the floor",
        "game_state": {"player_health_fraction": 0.75, "enemy_health_fraction": 0.55, "line_of_sight": True, "player_near_cover": False},
        "visual_evidence": {
            "target_visible": True, "crosshair_near_target": False, "player_exposed": True,
            "tags": ["crosshair_off_target", "enemy_visible", "player_exposed"],
            "summary": "Enemy is visible but player crosshair is aimed well below the target.",
            "recommended_focus": "Raise crosshair to chest or head height before firing.",
            "confidence": 0.87,
        },
        "macro_decision": "engage",
        "coaching_advice": "Your crosshair is too low — before you peek, pre-aim at chest height so your first shot has a chance to land.",
        "priority": "high",
    },
    {
        "scenario_id": "s006",
        "description": "Player rushes into the open at round start and immediately takes damage",
        "game_state": {"player_health_fraction": 0.35, "enemy_health_fraction": 0.95, "line_of_sight": True, "player_near_cover": False},
        "visual_evidence": {
            "target_visible": True, "crosshair_near_target": False, "player_exposed": True,
            "tags": ["player_took_damage", "early_round", "player_exposed", "health_disadvantage"],
            "summary": "Player took significant damage immediately after round start while rushing.",
            "recommended_focus": "Slow down at round start and use cover before making contact.",
            "confidence": 0.93,
        },
        "macro_decision": "take_cover",
        "coaching_advice": "Rushing in at round start got you punished — use cover before making contact and let the enemy show themselves first.",
        "priority": "high",
    },
    {
        "scenario_id": "s007",
        "description": "Player at 20% health, enemy at 85% health, both in line of sight",
        "game_state": {"player_health_fraction": 0.20, "enemy_health_fraction": 0.85, "line_of_sight": True, "player_near_cover": False},
        "visual_evidence": {
            "target_visible": True, "crosshair_near_target": True, "player_exposed": True,
            "tags": ["health_disadvantage", "player_exposed", "critical_health"],
            "summary": "Player at critical health facing a near full-health enemy with no cover.",
            "recommended_focus": "Retreat — this duel cannot be won at this health deficit.",
            "confidence": 0.96,
        },
        "macro_decision": "retreat",
        "coaching_advice": "You cannot win a straight duel at 20% versus 85% — retreat now and reset rather than trading from a losing position.",
        "priority": "high",
    },
    {
        "scenario_id": "s008",
        "description": "Player firing full auto at long range, shots drifting wide",
        "game_state": {"player_health_fraction": 0.70, "enemy_health_fraction": 0.70, "line_of_sight": True, "player_near_cover": False},
        "visual_evidence": {
            "target_visible": True, "crosshair_near_target": False, "player_exposed": True,
            "tags": ["player_missed_shots", "recoil_issue", "long_range", "player_exposed"],
            "summary": "Player spraying full auto at long range with crosshair drifting off target.",
            "recommended_focus": "Fire 2-3 shot bursts at this range and let recoil settle between shots.",
            "confidence": 0.86,
        },
        "macro_decision": "reposition",
        "coaching_advice": "Full auto at this range is killing your accuracy — switch to short bursts and aim for center mass. Get to cover between bursts.",
        "priority": "high",
    },
    {
        "scenario_id": "s009",
        "description": "Player wide-peeking corner with no information on enemy location",
        "game_state": {"player_health_fraction": 0.80, "enemy_health_fraction": 0.80, "line_of_sight": False, "player_near_cover": False},
        "visual_evidence": {
            "target_visible": False, "crosshair_near_target": None, "player_exposed": True,
            "tags": ["wide_peek", "no_information", "player_exposed", "crosshair_off_target"],
            "summary": "Player swinging wide on a corner with no enemy info and crosshair not pre-aimed.",
            "recommended_focus": "Peek tight with crosshair pre-aimed at head height.",
            "confidence": 0.83,
        },
        "macro_decision": "hold",
        "coaching_advice": "Wide-swinging without information gives the enemy time to react — peek tight with your crosshair already at head height.",
        "priority": "medium",
    },
    {
        "scenario_id": "s010",
        "description": "Player took a shot and stayed in the open instead of retreating",
        "game_state": {"player_health_fraction": 0.45, "enemy_health_fraction": 0.75, "line_of_sight": True, "player_near_cover": False},
        "visual_evidence": {
            "target_visible": True, "crosshair_near_target": False, "player_exposed": True,
            "tags": ["player_took_damage", "player_exposed", "repeated_exposure"],
            "summary": "Player took damage and remained stationary in the open, taking more hits.",
            "recommended_focus": "After first contact, always move to cover immediately.",
            "confidence": 0.89,
        },
        "macro_decision": "take_cover",
        "coaching_advice": "The moment you take damage you need to move — staying still in the open after being shot is one of the most avoidable mistakes.",
        "priority": "high",
    },
    {
        "scenario_id": "s011",
        "description": "Player re-peeking the same angle repeatedly after dying there",
        "game_state": {"player_health_fraction": 0.40, "enemy_health_fraction": 0.75, "line_of_sight": True, "player_near_cover": True},
        "visual_evidence": {
            "target_visible": True, "crosshair_near_target": False, "player_exposed": True,
            "tags": ["repeated_peek", "player_took_damage", "predictable_pattern"],
            "summary": "Player returning to the same angle where they previously took damage.",
            "recommended_focus": "Change your angle — the enemy is pre-aimed at your last position.",
            "confidence": 0.88,
        },
        "macro_decision": "reposition",
        "coaching_advice": "Stop going back to the same spot — the enemy is waiting there. Find a different angle so they have to readjust.",
        "priority": "high",
    },
    {
        "scenario_id": "s012",
        "description": "Player reloading in the open with enemy in line of sight",
        "game_state": {"player_health_fraction": 0.65, "enemy_health_fraction": 0.65, "line_of_sight": True, "player_near_cover": False},
        "visual_evidence": {
            "target_visible": True, "crosshair_near_target": None, "player_exposed": True,
            "tags": ["player_exposed", "reloading", "vulnerable", "enemy_visible"],
            "summary": "Player caught reloading in the open with enemy visible.",
            "recommended_focus": "Always move to cover before reloading.",
            "confidence": 0.91,
        },
        "macro_decision": "take_cover",
        "coaching_advice": "Never reload in the open — break line of sight first, get to cover, then reload. This is an easily avoidable death.",
        "priority": "high",
    },
    {
        "scenario_id": "s013",
        "description": "Player moving while shooting at medium range, all shots missing",
        "game_state": {"player_health_fraction": 0.60, "enemy_health_fraction": 0.60, "line_of_sight": True, "player_near_cover": False},
        "visual_evidence": {
            "target_visible": True, "crosshair_near_target": False, "player_exposed": True,
            "tags": ["player_missed_shots", "moving_while_shooting", "crosshair_off_target"],
            "summary": "Player moving and shooting simultaneously at medium range with no hits.",
            "recommended_focus": "Stop completely before firing — movement tanks accuracy.",
            "confidence": 0.87,
        },
        "macro_decision": "reposition",
        "coaching_advice": "You are moving and shooting at the same time — stop moving before you fire. Find cover, plant your feet, then take the shot.",
        "priority": "high",
    },
    {
        "scenario_id": "s014",
        "description": "Player holding tight angle from cover with crosshair on likely entry point",
        "game_state": {"player_health_fraction": 0.75, "enemy_health_fraction": 0.75, "line_of_sight": False, "player_near_cover": True},
        "visual_evidence": {
            "target_visible": False, "crosshair_near_target": None, "player_exposed": False,
            "tags": ["cover_used", "tight_angle", "pre_aimed", "waiting_for_peek"],
            "summary": "Player holding a tight angle from cover with crosshair pre-aimed at entry.",
            "recommended_focus": "Hold the angle — let the enemy peek into your crosshair.",
            "confidence": 0.88,
        },
        "macro_decision": "hold",
        "coaching_advice": "Good positioning — you have the tighter angle from cover. Stay patient and let the enemy commit first.",
        "priority": "low",
    },
    {
        "scenario_id": "s015",
        "description": "Player landed first shot and retreated to cover to reset",
        "game_state": {"player_health_fraction": 0.85, "enemy_health_fraction": 0.50, "line_of_sight": False, "player_near_cover": True},
        "visual_evidence": {
            "target_visible": False, "crosshair_near_target": None, "player_exposed": False,
            "tags": ["cover_used", "shot_landed", "controlled_engagement", "health_advantage"],
            "summary": "Player landed a hit and retreated to cover with a health advantage.",
            "recommended_focus": "Re-peek when ready to press the health advantage.",
            "confidence": 0.84,
        },
        "macro_decision": "hold",
        "coaching_advice": "Good discipline — you hit and retreated before they could trade back. Wait a moment then re-peek to press the advantage.",
        "priority": "low",
    },
]


def split_dataset(scenarios: list[dict], few_shot_count: int = 10) -> tuple[list[dict], list[dict]]:
    indices = list(range(len(scenarios)))
    random.shuffle(indices)
    few_shot = [scenarios[i] for i in indices[:few_shot_count]]
    validation = [scenarios[i] for i in indices[few_shot_count:]]
    return few_shot, validation


def main() -> None:
    output_dir = pathlib.Path("coaching/dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    few_shot, validation = split_dataset(SCENARIOS, few_shot_count=10)

    full_path = output_dir / "coaching_dataset.json"
    full_path.write_text(json.dumps({"total": len(SCENARIOS), "scenarios": SCENARIOS}, indent=2))
    print(f"Wrote {len(SCENARIOS)} scenarios to {full_path}")

    few_shot_path = output_dir / "few_shot.json"
    few_shot_path.write_text(json.dumps({"count": len(few_shot), "scenarios": few_shot}, indent=2))
    print(f"Wrote {len(few_shot)} few-shot scenarios to {few_shot_path}")

    val_path = output_dir / "validation.json"
    val_path.write_text(json.dumps({"count": len(validation), "scenarios": validation}, indent=2))
    print(f"Wrote {len(validation)} validation scenarios to {val_path}")

    decisions: dict[str, int] = {}
    priorities: dict[str, int] = {}
    for s in SCENARIOS:
        d = s["macro_decision"]
        p = s["priority"]
        decisions[d] = decisions.get(d, 0) + 1
        priorities[p] = priorities.get(p, 0) + 1

    print("\nDataset summary:")
    print(f"  Macro decisions: {decisions}")
    print(f"  Priorities: {priorities}")
    print(f"  Few-shot: {len(few_shot)}, Validation: {len(validation)}")


if __name__ == "__main__":
    main()