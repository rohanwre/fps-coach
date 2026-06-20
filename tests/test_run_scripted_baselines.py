from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from training.run_scripted_baselines import find_newest_log, parse_profiles


class ScriptedBaselineRunnerTests(unittest.TestCase):
    def test_parse_profiles_accepts_known_profiles(self) -> None:
        self.assertEqual(parse_profiles("easy, medium,hard"), ["easy", "medium", "hard"])

    def test_parse_profiles_rejects_unknown_profiles(self) -> None:
        with self.assertRaises(ValueError):
            parse_profiles("easy,nightmare")

    def test_find_newest_log_uses_start_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            old_log = log_dir / "old.jsonl"
            old_log.write_text("old", encoding="utf-8")
            old_time = time.time() - 10
            old_log.touch()
            start_time = time.time()
            time.sleep(0.01)
            new_log = log_dir / "new.jsonl"
            new_log.write_text("new", encoding="utf-8")
            old_log.touch()
            old_log_mtime = old_time
            import os

            os.utime(old_log, (old_log_mtime, old_log_mtime))

            self.assertEqual(find_newest_log(log_dir, start_time), new_log)


if __name__ == "__main__":
    unittest.main()
