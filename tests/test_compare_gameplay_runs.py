from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.compare_gameplay_runs import build_markdown, build_rows


class CompareGameplayRunsTests(unittest.TestCase):
    def test_build_rows_extracts_human_likeness_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "enemy_ppo_test_gameplay_metrics.json"
            path.write_text(
                json.dumps(
                    {
                        "per_run": [
                            {
                                "metrics": {
                                    "round_count": 3,
                                    "timeout_rounds": 2,
                                    "enemy_hit_rate": 0.25,
                                    "enemy_damage_diff": 12,
                                    "player_skill": {
                                        "hit_rate": 0.5,
                                        "damage_diff": -12,
                                        "recent_trend": "stable",
                                    },
                                    "human_likeness": {
                                        "reaction_delay_min_sec": 0.2,
                                        "reaction_delay_mean_sec": 0.4,
                                        "shots_per_second": 1.2,
                                        "blocked_shot_count": 5,
                                        "movement_direction_changes_per_sec": 2.0,
                                        "snap_to_target_count": 1,
                                        "flags": {
                                            "aim_snap_violation": True,
                                            "line_of_sight_violation": False,
                                        },
                                    },
                                }
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rows = build_rows([path])

        self.assertEqual(rows[0]["run"], "enemy_ppo_test")
        self.assertEqual(rows[0]["rounds"], 3)
        self.assertEqual(rows[0]["player_hit_rate"], 0.5)
        self.assertEqual(rows[0]["player_damage_diff"], -12)
        self.assertEqual(rows[0]["player_trend"], "stable")
        self.assertEqual(rows[0]["flags"], "aim_snap_violation")

    def test_build_markdown_formats_table(self) -> None:
        markdown = build_markdown(
            [
                {
                    "run": "a",
                    "rounds": 1,
                    "timeouts": 1,
                    "hit_rate": 0.1,
                    "damage_diff": 0,
                    "player_hit_rate": 0.2,
                    "player_damage_diff": -20,
                    "player_trend": "struggling",
                    "reaction_min": 0.2,
                    "reaction_mean": 0.3,
                    "shots_sec": 0.4,
                    "blocked": 2,
                    "move_changes_sec": 1.2,
                    "snap_count": 0,
                    "flags": "none",
                }
            ]
        )

        self.assertIn("| run | rounds |", markdown)
        self.assertIn("| a | 1 | 1 | 0.100 | 0 | 0.200 | -20 | struggling |", markdown)


if __name__ == "__main__":
    unittest.main()
