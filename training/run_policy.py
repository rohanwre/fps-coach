"""Run a saved Stable-Baselines3 policy against a connected rendered Godot game."""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.train_enemy import LOCAL_GODOT_RL


if LOCAL_GODOT_RL.exists():
    sys.path.insert(0, str(LOCAL_GODOT_RL))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a saved PPO policy in Godot.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--agent-role", choices=("enemy", "player"), default="player")
    parser.add_argument("--scripted-enemy-profile", default="hard")
    parser.add_argument("--scripted-player-profile", default="medium")
    parser.add_argument("--ppo-difficulty-profile", default="adaptive")
    parser.add_argument("--reward-profile", default="default")
    parser.add_argument("--port", type=int, default=11400)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--stochastic-actions", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coaching-frame-dir", default=None)
    parser.add_argument("--gameplay-log-dir", default=None)
    return parser.parse_args()


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

    args = parse_args()
    set_random_seed(args.seed)
    kwargs = {
        "agent-role": args.agent_role,
        "disable-coaching-frame-capture": "false",
    }
    if args.agent_role == "player":
        kwargs["baseline-profile"] = args.scripted_enemy_profile
    else:
        kwargs["scripted-player-profile"] = args.scripted_player_profile
        kwargs["ppo-difficulty-profile"] = args.ppo_difficulty_profile
    if args.reward_profile != "default":
        kwargs["reward-profile"] = args.reward_profile
    if args.coaching_frame_dir:
        pathlib.Path(args.coaching_frame_dir).mkdir(parents=True, exist_ok=True)
        kwargs["coaching-frame-dir"] = args.coaching_frame_dir
    if args.gameplay_log_dir:
        pathlib.Path(args.gameplay_log_dir).mkdir(parents=True, exist_ok=True)
        kwargs["gameplay-log-dir"] = args.gameplay_log_dir

    env = StableBaselinesGodotEnv(env_path=None, port=args.port, action_repeat=4, **kwargs)
    model = PPO.load(args.model_path)
    obs = env.reset()
    try:
        for _ in range(max(args.steps, 1)):
            action, _ = model.predict(obs, deterministic=not args.stochastic_actions)
            obs, _, _, _ = env.step(action)
    finally:
        env.close()


if __name__ == "__main__":
    main()
