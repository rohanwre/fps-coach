from __future__ import annotations

import unittest

from training.evaluate_gameplay_logs import summarize_events


class GameplayLogEvaluationTests(unittest.TestCase):
    def test_summarize_events_computes_enemy_metrics(self) -> None:
        events = [
            {
                "event_type": "round_started",
                "unix_time_msec": 1_000.0,
                "payload": {"round_index": 1},
            },
            {
                "event_type": "shot_hit",
                "unix_time_msec": 1_500.0,
                "payload": {"shooter_id": "enemy"},
            },
            {
                "event_type": "shot_missed",
                "unix_time_msec": 1_600.0,
                "payload": {"shooter_id": "enemy"},
            },
            {
                "event_type": "actor_damaged",
                "unix_time_msec": 1_700.0,
                "payload": {
                    "actor_id": "player",
                    "source_id": "enemy",
                    "amount": 25,
                },
            },
            {
                "event_type": "actor_damaged",
                "unix_time_msec": 1_800.0,
                "payload": {
                    "actor_id": "enemy",
                    "source_id": "player",
                    "amount": 10,
                },
            },
            {
                "event_type": "round_ended",
                "unix_time_msec": 4_000.0,
                "payload": {
                    "round_index": 1,
                    "killer_id": "enemy",
                    "dead_actor_id": "player",
                },
            },
        ]

        metrics = summarize_events(events)

        self.assertEqual(metrics["enemy_wins"], 1)
        self.assertEqual(metrics["enemy_losses"], 0)
        self.assertEqual(metrics["timeout_rounds"], 0)
        self.assertEqual(metrics["round_count"], 1)
        self.assertEqual(metrics["enemy_total_shots"], 2)
        self.assertAlmostEqual(metrics["enemy_hit_rate"], 0.5)
        self.assertEqual(metrics["enemy_damage_dealt"], 25)
        self.assertEqual(metrics["enemy_damage_taken"], 10)
        self.assertEqual(metrics["enemy_damage_diff"], 15)
        self.assertAlmostEqual(metrics["mean_round_duration_sec"], 3.0)
        self.assertIn("human_likeness", metrics)
        self.assertIn("player_skill", metrics)
        self.assertEqual(metrics["player_skill"]["damage_dealt"], 10)
        self.assertEqual(metrics["player_skill"]["damage_taken"], 25)
        self.assertEqual(metrics["player_skill"]["kills"], 0)
        self.assertEqual(metrics["player_skill"]["deaths"], 1)
        self.assertEqual(metrics["player_skill"]["recent_trend"], "struggling")

    def test_summarize_events_computes_player_skill_metrics(self) -> None:
        events = [
            {
                "event_type": "round_started",
                "unix_time_msec": 1_000.0,
                "payload": {"round_index": 1},
            },
            {
                "event_type": "shot_hit",
                "unix_time_msec": 1_500.0,
                "payload": {"shooter_id": "player"},
            },
            {
                "event_type": "shot_missed",
                "unix_time_msec": 1_700.0,
                "payload": {"shooter_id": "player"},
            },
            {
                "event_type": "actor_damaged",
                "unix_time_msec": 1_800.0,
                "payload": {
                    "actor_id": "enemy",
                    "source_id": "player",
                    "amount": 80,
                },
            },
            {
                "event_type": "actor_damaged",
                "unix_time_msec": 2_000.0,
                "payload": {
                    "actor_id": "player",
                    "source_id": "enemy",
                    "amount": 10,
                },
            },
            {
                "event_type": "round_ended",
                "unix_time_msec": 5_000.0,
                "payload": {
                    "round_index": 1,
                    "killer_id": "player",
                    "dead_actor_id": "enemy",
                },
            },
        ]

        player_skill = summarize_events(events)["player_skill"]

        self.assertEqual(player_skill["hits"], 1)
        self.assertEqual(player_skill["misses"], 1)
        self.assertEqual(player_skill["total_shots"], 2)
        self.assertAlmostEqual(player_skill["hit_rate"], 0.5)
        self.assertEqual(player_skill["damage_dealt"], 80)
        self.assertEqual(player_skill["damage_taken"], 10)
        self.assertEqual(player_skill["damage_diff"], 70)
        self.assertEqual(player_skill["kills"], 1)
        self.assertEqual(player_skill["deaths"], 0)
        self.assertAlmostEqual(player_skill["survival_time_mean_sec"], 4.0)
        self.assertEqual(player_skill["recent_trend"], "improving")

    def test_summarize_events_computes_human_likeness_metrics(self) -> None:
        events = [
            {
                "event_type": "enemy_behavior_sample",
                "unix_time_msec": 1_000.0,
                "payload": {
                    "sample_reason": "periodic",
                    "line_of_sight": True,
                    "weapon_ready": True,
                    "aim_alignment": 0.2,
                    "enemy_velocity": {"x": 1.0, "y": 0.0, "z": 0.0},
                },
            },
            {
                "event_type": "shot_fired",
                "unix_time_msec": 1_350.0,
                "payload": {"shooter_id": "enemy"},
            },
            {
                "event_type": "enemy_behavior_sample",
                "unix_time_msec": 1_350.0,
                "payload": {
                    "sample_reason": "shot_fired",
                    "line_of_sight": True,
                    "weapon_ready": True,
                    "aim_alignment": 0.95,
                    "enemy_velocity": {"x": -1.0, "y": 0.0, "z": 0.0},
                },
            },
            {
                "event_type": "shot_fired",
                "unix_time_msec": 2_000.0,
                "payload": {"shooter_id": "enemy"},
            },
            {
                "event_type": "enemy_behavior_sample",
                "unix_time_msec": 2_000.0,
                "payload": {
                    "sample_reason": "shot_fired",
                    "line_of_sight": False,
                    "weapon_ready": True,
                    "aim_alignment": 0.1,
                    "enemy_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
            },
            {
                "event_type": "enemy_shot_blocked",
                "unix_time_msec": 2_200.0,
                "payload": {"reason": "reaction_delay"},
            },
            {
                "event_type": "enemy_shot_blocked",
                "unix_time_msec": 2_300.0,
                "payload": {"reason": "low_alignment"},
            },
            {
                "event_type": "enemy_shot_blocked",
                "unix_time_msec": 2_400.0,
                "payload": {"reason": "reaction_delay"},
            },
        ]

        metrics = summarize_events(events)
        human_likeness = metrics["human_likeness"]

        self.assertAlmostEqual(human_likeness["reaction_delay_mean_sec"], 0.35)
        self.assertAlmostEqual(human_likeness["reaction_delay_min_sec"], 0.35)
        self.assertEqual(human_likeness["snap_to_target_count"], 1)
        self.assertAlmostEqual(human_likeness["movement_direction_changes_per_sec"], 1 / 1.4)
        self.assertAlmostEqual(human_likeness["stationary_fraction"], 1 / 3)
        self.assertEqual(human_likeness["shots_without_line_of_sight"], 1)
        self.assertEqual(human_likeness["behavior_sample_count"], 3)
        self.assertEqual(human_likeness["blocked_shot_count"], 3)
        self.assertEqual(human_likeness["blocked_shots_by_reason"]["reaction_delay"], 2)
        self.assertEqual(human_likeness["blocked_shots_by_reason"]["low_alignment"], 1)
        self.assertTrue(human_likeness["flags"]["line_of_sight_violation"])
        self.assertTrue(human_likeness["flags"]["aim_snap_violation"])

    def test_summarize_events_counts_timeout_rounds(self) -> None:
        events = [
            {
                "event_type": "round_started",
                "unix_time_msec": 1_000.0,
                "payload": {"round_index": 1},
            },
            {
                "event_type": "round_ended",
                "unix_time_msec": 3_500.0,
                "payload": {
                    "round_index": 1,
                    "killer_id": "",
                    "dead_actor_id": "",
                    "end_reason": "timeout",
                },
            },
        ]

        metrics = summarize_events(events)

        self.assertEqual(metrics["round_count"], 1)
        self.assertEqual(metrics["timeout_rounds"], 1)
        self.assertEqual(metrics["enemy_wins"], 0)
        self.assertEqual(metrics["enemy_losses"], 0)
        self.assertEqual(metrics["enemy_win_rate"], 0.0)
        self.assertAlmostEqual(metrics["mean_round_duration_sec"], 2.5)

    def test_summarize_events_tracks_adaptive_ppo_difficulty(self) -> None:
        events = [
            {
                "event_type": "round_started",
                "unix_time_msec": 1_000.0,
                "payload": {
                    "round_index": 1,
                    "active_enemy_ppo_difficulty": "medium",
                    "active_scripted_player_profile": "hard",
                },
            },
            {
                "event_type": "enemy_ppo_difficulty_changed",
                "unix_time_msec": 2_000.0,
                "payload": {
                    "round_index": 1,
                    "previous_profile": "medium",
                    "active_profile": "hard",
                    "player_trend": "improving",
                    "difficulty_config": {"reaction_delay": 0.175},
                },
            },
        ]

        adaptive = summarize_events(events)["adaptive_difficulty"]

        self.assertEqual(adaptive["ppo_difficulty_rounds"], {"medium": 1})
        self.assertEqual(adaptive["scripted_player_rounds"], {"hard": 1})
        self.assertEqual(adaptive["transition_count"], 1)
        self.assertEqual(adaptive["transitions"][0]["active_profile"], "hard")
        self.assertEqual(adaptive["transitions"][0]["difficulty_config"]["reaction_delay"], 0.175)

    def test_shot_sample_uses_controller_reaction_elapsed_when_available(self) -> None:
        events = [
            {
                "event_type": "enemy_behavior_sample",
                "unix_time_msec": 1_000.0,
                "payload": {
                    "sample_reason": "periodic",
                    "line_of_sight": True,
                    "weapon_ready": True,
                    "aim_alignment": 0.5,
                    "enemy_velocity": {"x": 1.0, "y": 0.0, "z": 0.0},
                },
            },
            {
                "event_type": "shot_fired",
                "unix_time_msec": 1_020.0,
                "payload": {"shooter_id": "enemy"},
            },
            {
                "event_type": "enemy_behavior_sample",
                "unix_time_msec": 1_020.0,
                "payload": {
                    "sample_reason": "shot_fired",
                    "line_of_sight": True,
                    "line_of_sight_elapsed": 0.22,
                    "weapon_ready": True,
                    "aim_alignment": 0.6,
                    "enemy_velocity": {"x": 1.0, "y": 0.0, "z": 0.0},
                },
            },
        ]

        human_likeness = summarize_events(events)["human_likeness"]

        self.assertAlmostEqual(human_likeness["reaction_delay_min_sec"], 0.22)


if __name__ == "__main__":
    unittest.main()
