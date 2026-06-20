from __future__ import annotations

import unittest
from argparse import Namespace

from training.train_enemy import build_godot_env_kwargs


class TrainEnemyTests(unittest.TestCase):
    def test_build_godot_env_kwargs_omits_defaults(self) -> None:
        args = Namespace(
            reward_profile="default",
            gameplay_log_dir=None,
            capture_coaching_frames=False,
            coaching_frame_dir=None,
            agent_role="enemy",
            scripted_enemy_profile="hard",
        )

        self.assertEqual(
            build_godot_env_kwargs(args),
            {
                "disable-coaching-frame-capture": "true",
                "agent-role": "enemy",
                "scripted-player-profile": "medium",
                "ppo-difficulty-profile": "adaptive",
            },
        )

    def test_build_godot_env_kwargs_uses_godot_arg_names(self) -> None:
        args = Namespace(
            reward_profile="timeout_aim_assist",
            gameplay_log_dir="/tmp/gameplay",
            capture_coaching_frames=True,
            coaching_frame_dir="/tmp/frames",
            agent_role="player",
            scripted_enemy_profile="hard",
            scripted_player_profile="medium",
            ppo_difficulty_profile="adaptive",
        )

        self.assertEqual(
            build_godot_env_kwargs(args),
            {
                "reward-profile": "timeout_aim_assist",
                "gameplay-log-dir": "/tmp/gameplay",
                "disable-coaching-frame-capture": "false",
                "coaching-frame-dir": "/tmp/frames",
                "agent-role": "player",
                "baseline-profile": "hard",
            },
        )

    def test_enemy_role_uses_scripted_player_and_ppo_difficulty(self) -> None:
        args = Namespace(
            reward_profile="timeout_aim_assist",
            gameplay_log_dir=None,
            capture_coaching_frames=False,
            coaching_frame_dir=None,
            agent_role="enemy",
            scripted_enemy_profile="hard",
            scripted_player_profile="hard",
            ppo_difficulty_profile="easy",
        )

        kwargs = build_godot_env_kwargs(args)

        self.assertEqual(kwargs["scripted-player-profile"], "hard")
        self.assertEqual(kwargs["ppo-difficulty-profile"], "easy")
        self.assertNotIn("baseline-profile", kwargs)


if __name__ == "__main__":
    unittest.main()
