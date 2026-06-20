from __future__ import annotations

import os
import tempfile
import time
import unittest
from argparse import Namespace
from pathlib import Path

from training.run_training_ablation import build_godot_command, find_newest_log


class TrainingAblationRunnerTests(unittest.TestCase):
    def test_find_newest_log_ignores_logs_before_start_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            old_log = log_dir / "old.jsonl"
            old_log.write_text("old", encoding="utf-8")
            old_time = time.time() - 10
            os.utime(old_log, (old_time, old_time))

            start_time = time.time()
            time.sleep(0.01)
            new_log = log_dir / "new.jsonl"
            new_log.write_text("new", encoding="utf-8")

            self.assertEqual(find_newest_log(log_dir, start_time), new_log)

    def test_build_godot_command_passes_port_and_reward_profile(self) -> None:
        args = Namespace(
            godot_path="/Applications/Godot.app/Contents/MacOS/Godot",
            project_path="/project",
            port=11012,
            speedup=8,
            reward_profile="timeout_centered",
            log_dir="/logs/run",
        )

        command = build_godot_command(args)

        self.assertIn("--port=11012", command)
        self.assertIn("--speedup=8", command)
        self.assertIn("--reward-profile=timeout_centered", command)
        self.assertIn("--gameplay-log-dir=/logs/run", command)


if __name__ == "__main__":
    unittest.main()
