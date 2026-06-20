from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from training.benchmark_enemy_policies import build_godot_command, build_policy_command, flatten_result, parse_policies


class BenchmarkEnemyPoliciesTests(unittest.TestCase):
    def test_parse_policies_requires_existing_labelled_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "policy.zip"
            path.write_text("placeholder", encoding="utf-8")

            policies = parse_policies([f"3k={path}"])

            self.assertEqual(policies[0][0], "3k")
            self.assertEqual(policies[0][1], path.resolve())

    def test_commands_fix_opponent_and_execution_tier(self) -> None:
        args = Namespace(
            scripted_player_profile="easy",
            ppo_difficulty_profile="medium",
            reward_profile="timeout_aim_assist",
            steps=1200,
            stochastic_actions=True,
            seed=7,
            speedup=4,
            godot_path="/Godot",
            project_path="/project",
        )

        policy = build_policy_command(args, Path("/model.zip"), 11600, Path("/logs"))
        godot = build_godot_command(args, 11600, Path("/logs"))

        self.assertIn("easy", policy)
        self.assertIn("medium", policy)
        self.assertIn("--stochastic-actions", policy)
        self.assertIn("7", policy)
        self.assertIn("--scripted-player-profile=easy", godot)
        self.assertIn("--ppo-difficulty-profile=medium", godot)
        self.assertIn("--reward-profile=timeout_aim_assist", godot)
        self.assertIn("--gameplay-log-dir=/logs", godot)

    def test_flatten_result_exposes_presentation_metrics(self) -> None:
        metrics = {
            "round_count": 4,
            "enemy_wins": 2,
            "enemy_losses": 1,
            "timeout_rounds": 1,
            "enemy_hits": 8,
            "enemy_total_shots": 20,
            "enemy_hit_rate": 0.4,
            "enemy_damage_dealt": 96,
            "enemy_damage_diff": 12,
            "mean_round_duration_sec": 8.0,
            "human_likeness": {
                "reaction_delay_min_sec": 0.25,
                "shots_without_line_of_sight": 0,
                "flags": {},
            },
        }

        row = flatten_result({"label": "test", "metrics": metrics})

        self.assertEqual(row["policy"], "test")
        self.assertEqual(row["hits"], 8)
        self.assertEqual(row["rounds"], 4)


if __name__ == "__main__":
    unittest.main()
