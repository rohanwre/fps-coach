from __future__ import annotations

import unittest

from coaching.agent_workflow import (
    build_query_from_gameplay_stats,
    extract_window_signals,
    generate_coaching_feedback,
    generate_visual_coaching_feedback,
    recommend_macro_decision,
    summarize_gameplay_stats,
)


class AgentWorkflowTests(unittest.TestCase):
    def test_query_prioritizes_low_hit_rate(self) -> None:
        query = build_query_from_gameplay_stats(
            {
                "hit_rate": 0.1,
                "survival_time_sec": 8.0,
            }
        )

        self.assertIn("low hit rate", query)

    def test_query_uses_survival_when_accuracy_is_acceptable(self) -> None:
        query = build_query_from_gameplay_stats(
            {
                "hit_rate": 0.35,
                "survival_time_sec": 8.0,
            }
        )

        self.assertIn("dying early", query)

    def test_generate_coaching_feedback_returns_grounded_payload(self) -> None:
        feedback = generate_coaching_feedback(
            {
                "hit_rate": 0.16,
                "survival_time_sec": 14.3,
                "damage_taken": 92,
            },
            top_k=2,
        )

        self.assertEqual(len(feedback["grounded_tips"]), 2)
        self.assertEqual(len(feedback["action_items"]), 2)
        self.assertIn("hit_rate=0.16", feedback["gameplay_summary"])

    def test_summary_formats_missing_stats_as_zeroes(self) -> None:
        summary = summarize_gameplay_stats({})

        self.assertIn("hit_rate=0.00", summary)
        self.assertIn("survival_time=0.0s", summary)
        self.assertIn("damage_taken=0", summary)

    def test_extract_window_signals_uses_structured_event_fallback(self) -> None:
        signals = extract_window_signals([
            {
                "round_index": 3,
                "center_time_sec": 4.0,
                "frame_paths": [],
                "events": [
                    {"event_type": "shot_missed", "payload": {"shooter_id": "player"}},
                    {"event_type": "actor_damaged", "payload": {"actor_id": "player"}},
                ],
            }
        ])

        self.assertEqual(signals["damage_window_rate"], 1.0)
        self.assertEqual(signals["miss_window_rate"], 1.0)
        self.assertEqual(signals["rounds_with_early_death"], [3])

    def test_visual_coaching_feedback_grounds_visual_evidence(self) -> None:
        feedback = generate_visual_coaching_feedback(
            {"hit_rate": 0.1, "survival_time_sec": 8.0, "damage_taken": 40},
            {
                "tags": ["player_exposed"],
                "recommended_focus": "Use cover before committing.",
            },
            top_k=2,
        )

        self.assertEqual(feedback["macro_decision"]["decision"], "hold")
        self.assertIn("cover", feedback["primary_recommendation"])
        self.assertIn("player exposed", feedback["retrieval_query"])
        self.assertEqual(len(feedback["grounded_tips"]), 2)

    def test_macro_decision_takes_nearby_cover_when_low_and_visible(self) -> None:
        decision = recommend_macro_decision(
            {
                "player_health_fraction": 0.25,
                "enemy_health_fraction": 0.8,
                "line_of_sight": True,
                "player_near_cover": True,
            }
        )

        self.assertEqual(decision["decision"], "take_cover")
        self.assertIn("nearby cover", decision["message"])

    def test_macro_decision_pushes_low_enemy_with_advantage(self) -> None:
        decision = recommend_macro_decision(
            {
                "player_health_fraction": 0.9,
                "enemy_health_fraction": 0.25,
                "line_of_sight": True,
            }
        )

        self.assertEqual(decision["decision"], "push")

    def test_macro_decision_holds_without_line_of_sight(self) -> None:
        decision = recommend_macro_decision(
            {
                "player_health_fraction": 0.8,
                "enemy_health_fraction": 0.8,
                "line_of_sight": False,
            }
        )

        self.assertEqual(decision["decision"], "hold")


if __name__ == "__main__":
    unittest.main()
