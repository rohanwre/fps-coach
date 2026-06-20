from __future__ import annotations

import unittest

from coaching.build_event_windows import build_event_windows
from coaching.visual_evidence import OpenAICompatibleVLM, extract_visual_evidence


class CoachingEventWindowTests(unittest.TestCase):
    def test_build_event_windows_collects_nearby_context(self) -> None:
        events = [
            {
                "event_index": 1,
                "event_type": "round_started",
                "unix_time_msec": 1000.0,
                "payload": {"round_index": 1},
            },
            {
                "event_index": 2,
                "event_type": "shot_missed",
                "unix_time_msec": 2000.0,
                "payload": {"round_index": 1, "shooter_id": "player"},
            },
            {
                "event_index": 3,
                "event_type": "actor_damaged",
                "unix_time_msec": 2600.0,
                "payload": {"round_index": 1, "actor_id": "player"},
            },
        ]

        windows = build_event_windows(events, window_sec=1.0)

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["center_event_type"], "shot_missed")
        self.assertEqual(len(windows[0]["events"]), 3)

    def test_build_event_windows_attaches_nearby_player_frames(self) -> None:
        events = [
            {
                "event_index": 1,
                "event_type": "shot_missed",
                "unix_time_msec": 2000.0,
                "payload": {"round_index": 1, "shooter_id": "player"},
            },
            {
                "event_index": 2,
                "event_type": "player_frame_captured",
                "unix_time_msec": 2100.0,
                "payload": {"round_index": 1, "frame_path": "/tmp/frame.png"},
            },
        ]

        windows = build_event_windows(events, window_sec=1.0)

        self.assertEqual(windows[0]["frame_paths"], ["/tmp/frame.png"])

    def test_build_event_windows_puts_closest_frame_last_for_vlm_analysis(self) -> None:
        events = [
            {
                "event_index": 1,
                "event_type": "shot_missed",
                "unix_time_msec": 2000.0,
                "payload": {"round_index": 1, "shooter_id": "player"},
            },
            {
                "event_index": 2,
                "event_type": "player_frame_captured",
                "unix_time_msec": 2800.0,
                "payload": {"round_index": 1, "frame_path": "/tmp/far.png"},
            },
            {
                "event_index": 3,
                "event_type": "player_frame_captured",
                "unix_time_msec": 2100.0,
                "payload": {"round_index": 1, "frame_path": "/tmp/close.png"},
            },
        ]

        windows = build_event_windows(events, window_sec=1.0)

        self.assertEqual(windows[0]["frame_paths"], ["/tmp/far.png", "/tmp/close.png"])

    def test_build_event_windows_does_not_attach_frames_from_adjacent_round(self) -> None:
        events = [
            {
                "event_index": 1,
                "event_type": "round_ended",
                "unix_time_msec": 2000.0,
                "payload": {"round_index": 1},
            },
            {
                "event_index": 2,
                "event_type": "player_frame_captured",
                "unix_time_msec": 2100.0,
                "payload": {"round_index": 2, "frame_path": "/tmp/next-round.png"},
            },
        ]

        windows = build_event_windows(events, window_sec=1.0)

        self.assertEqual(windows[0]["frame_paths"], [])

    def test_line_of_sight_macro_state_creates_coaching_window(self) -> None:
        events = [
            {
                "event_index": 1,
                "event_type": "macro_state",
                "unix_time_msec": 2000.0,
                "payload": {"round_index": 1, "reason": "line_of_sight_lost"},
            },
            {
                "event_index": 2,
                "event_type": "player_frame_captured",
                "unix_time_msec": 2100.0,
                "payload": {"round_index": 1, "frame_path": "/tmp/hidden.png"},
            },
        ]

        windows = build_event_windows(events, window_sec=1.0)

        self.assertEqual(windows[0]["center_event_type"], "macro_state")
        self.assertEqual(windows[0]["frame_paths"], ["/tmp/hidden.png"])

    def test_periodic_tactical_macro_state_creates_coaching_window(self) -> None:
        windows = build_event_windows(
            [
                {
                    "event_index": 1,
                    "event_type": "macro_state",
                    "unix_time_msec": 2000.0,
                    "payload": {"round_index": 1, "reason": "tactical_interval"},
                }
            ],
            window_sec=1.0,
        )

        self.assertEqual(len(windows), 1)

    def test_visual_evidence_uses_structured_event_tags(self) -> None:
        evidence = extract_visual_evidence(
            [],
            {
                "center_event_type": "shot_missed",
                "events": [
                    {
                        "event_type": "shot_missed",
                        "payload": {"shooter_id": "player"},
                    },
                    {
                        "event_type": "actor_damaged",
                        "payload": {"actor_id": "player"},
                    },
                ],
            },
        )

        self.assertIn("player_took_damage", evidence["tags"])
        self.assertIn("player_missed_shots", evidence["tags"])

    def test_visual_evidence_uses_injected_provider(self) -> None:
        class FakeProvider:
            def analyze(self, frame_paths, event_window):
                return {"evidence_source": "fake", "frame_count": len(frame_paths)}

        evidence = extract_visual_evidence(
            ["/tmp/frame.png"],
            {"center_event_type": "shot_missed"},
            provider=FakeProvider(),
        )

        self.assertEqual(evidence, {"evidence_source": "fake", "frame_count": 1})

    def test_vlm_omits_event_context_by_default(self) -> None:
        provider = OpenAICompatibleVLM("http://localhost", "test-model")

        self.assertFalse(provider.include_event_context)


if __name__ == "__main__":
    unittest.main()
