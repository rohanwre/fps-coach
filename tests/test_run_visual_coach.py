from __future__ import annotations

import unittest

from coaching.run_visual_coach import generate_visual_coaching_rows


class RunVisualCoachTests(unittest.TestCase):
    def test_generate_rows_combines_frames_evidence_and_retrieval(self) -> None:
        class FakeProvider:
            def analyze(self, frame_paths, event_window):
                return {
                    "evidence_source": "fake-vlm",
                    "frame_count": len(frame_paths),
                    "tags": ["crosshair_off_target"],
                    "recommended_focus": "Set the crosshair before firing.",
                    "event_context_included": False,
                }

        rows = generate_visual_coaching_rows(
            [
                {
                    "run_id": "run-1",
                    "round_index": 2,
                    "center_event_type": "shot_missed",
                    "window_sec": 3.0,
                    "frame_paths": ["/tmp/frame.png"],
                    "events": [
                        {
                            "event_type": "shot_missed",
                            "payload": {"shooter_id": "player"},
                        },
                        {
                            "event_type": "macro_state",
                            "payload": {
                                "player_health_fraction": 0.9,
                                "enemy_health_fraction": 0.2,
                                "player_health_advantage": 0.7,
                                "line_of_sight": True,
                            },
                        }
                    ],
                }
            ],
            limit=1,
            top_k=1,
            provider=FakeProvider(),
        )

        self.assertEqual(rows[0]["run_id"], "run-1")
        self.assertIsNone(rows[0]["center_event_index"])
        self.assertEqual(rows[0]["analyzed_frame_path"], "/tmp/frame.png")
        self.assertEqual(rows[0]["coaching"]["visual_evidence"]["evidence_source"], "fake-vlm")
        self.assertFalse(rows[0]["coaching"]["visual_evidence"]["event_context_included"])
        self.assertEqual(rows[0]["coaching"]["macro_decision"]["decision"], "push")
        self.assertIn("aggressive", rows[0]["coaching"]["primary_recommendation"])

    def test_generate_rows_uses_macro_state_closest_to_center_event(self) -> None:
        class FakeProvider:
            def analyze(self, frame_paths, event_window):
                return {"tags": [], "recommended_focus": "", "target_visible": False}

        rows = generate_visual_coaching_rows(
            [
                {
                    "center_event_index": 10,
                    "events": [
                        {
                            "event_index": 9,
                            "event_type": "macro_state",
                            "payload": {"line_of_sight": False},
                        },
                        {
                            "event_index": 30,
                            "event_type": "macro_state",
                            "payload": {"line_of_sight": True},
                        },
                    ],
                }
            ],
            limit=1,
            top_k=1,
            provider=FakeProvider(),
        )

        self.assertEqual(rows[0]["coaching"]["macro_decision"]["decision"], "hold")


if __name__ == "__main__":
    unittest.main()
