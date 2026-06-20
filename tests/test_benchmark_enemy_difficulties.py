from __future__ import annotations

import unittest
from argparse import Namespace

from training.benchmark_enemy_difficulties import build_profile_args, parse_profiles


class BenchmarkEnemyDifficultiesTests(unittest.TestCase):
    def test_parse_profiles_rejects_adaptive_profile(self) -> None:
        with self.assertRaises(ValueError):
            parse_profiles("easy,adaptive")

    def test_build_profile_args_fixes_requested_tier(self) -> None:
        args = Namespace(scripted_player_profile="easy", steps=2400)

        profile_args = build_profile_args(args, "hard")

        self.assertEqual(profile_args.scripted_player_profile, "easy")
        self.assertEqual(profile_args.ppo_difficulty_profile, "hard")


if __name__ == "__main__":
    unittest.main()
