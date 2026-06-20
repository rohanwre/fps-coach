from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

from training.run_adaptive_curriculum import build_stage_command, parse_profiles


class AdaptiveCurriculumTests(unittest.TestCase):
    def test_parse_profiles_rejects_unknown_profile(self) -> None:
        with self.assertRaises(ValueError):
            parse_profiles("easy,impossible")

    def test_stage_command_continues_previous_checkpoint(self) -> None:
        args = Namespace(
            name_prefix="adaptive_test",
            timesteps_per_stage=1000,
            eval_episodes=2,
            speedup=4,
            start_port=12000,
            godot_path="/Applications/Godot.app/Contents/MacOS/Godot",
            models_dir="/models",
            log_root="/logs",
        )

        command, save_path = build_stage_command(args, "hard", 2, Path("/models/previous.zip"))

        self.assertIn("--scripted-player-profile", command)
        self.assertIn("hard", command)
        self.assertIn("--load-model-path", command)
        self.assertIn("/models/previous.zip", command)
        self.assertEqual(save_path, Path("/models/adaptive_test_03_hard.zip"))


if __name__ == "__main__":
    unittest.main()
