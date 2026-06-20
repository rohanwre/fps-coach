from __future__ import annotations

import unittest
from argparse import Namespace

from training.run_parallel_ablations import build_jobs, parse_csv


class ParallelAblationRunnerTests(unittest.TestCase):
    def test_parse_csv_ignores_empty_items(self) -> None:
        self.assertEqual(parse_csv("timeout, timeout_centered,,"), ["timeout", "timeout_centered"])

    def test_build_jobs_assigns_unique_ports(self) -> None:
        args = Namespace(
            name_prefix="enemy_ppo_test",
            reward_profiles="timeout,timeout_centered",
            start_port=12000,
        )

        jobs = build_jobs(args)

        self.assertEqual([job.name for job in jobs], ["enemy_ppo_test_timeout", "enemy_ppo_test_timeout_centered"])
        self.assertEqual([job.reward_profile for job in jobs], ["timeout", "timeout_centered"])
        self.assertEqual([job.port for job in jobs], [12000, 12001])


if __name__ == "__main__":
    unittest.main()
