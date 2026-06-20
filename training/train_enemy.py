"""Minimal PPO training entry point for a Godot RL gameplay agent.

Verify the Godot RL Agents loop: Python opens the server, Godot connects through the Sync node, and PPO
receives observations/actions/rewards from the EnemyAIController3D node and records gameplay logs.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
LOCAL_GODOT_RL = WORKSPACE_ROOT / "godot_rl_agents"

if LOCAL_GODOT_RL.exists():
    sys.path.insert(0, str(LOCAL_GODOT_RL))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an adaptive-fps-coach PPO gameplay agent.")
    parser.add_argument(
        "--env-path",
        default=None,
        help=(
            "Optional exported Godot executable/app path without platform suffix. "
            "Omit this for in-editor training."
        ),
    )
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=11008)
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=1,
        help="Number of parallel exported Godot environments. Requires --env-path when greater than 1.",
    )
    parser.add_argument("--speedup", type=int, default=1)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--experiment-dir", default=str(REPO_ROOT / "training" / "logs"))
    parser.add_argument("--experiment-name", default="enemy_ppo")
    parser.add_argument("--save-model-path", default=str(REPO_ROOT / "training" / "models" / "enemy_ppo.zip"))
    parser.add_argument(
        "--load-model-path",
        default=None,
        help="Optional PPO checkpoint to continue training instead of initializing a new policy.",
    )
    parser.add_argument("--save-metadata-path", default=None)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-frequency", type=int, default=0)
    parser.add_argument("--agent-role", choices=("enemy", "player"), default="enemy")
    parser.add_argument(
        "--scripted-enemy-profile",
        choices=("easy", "medium", "hard", "adaptive"),
        default="hard",
        help="Scripted opponent profile used when training the player PPO agent.",
    )
    parser.add_argument(
        "--scripted-player-profile",
        choices=("human", "easy", "medium", "hard", "adaptive"),
        default="medium",
        help="Scripted opponent profile used when training the enemy PPO agent.",
    )
    parser.add_argument(
        "--ppo-difficulty-profile",
        choices=("easy", "medium", "hard", "adaptive"),
        default="adaptive",
        help="Human-like execution tier used by the enemy PPO.",
    )
    parser.add_argument(
        "--reward-profile",
        default="default",
        help="Reward/control profile to pass to exported Godot environments.",
    )
    parser.add_argument(
        "--gameplay-log-dir",
        default=None,
        help="Optional gameplay log directory passed to exported Godot environments.",
    )
    parser.add_argument(
        "--capture-coaching-frames",
        action="store_true",
        help="Capture player-perspective coaching frames during training. Disabled by default for speed.",
    )
    parser.add_argument(
        "--coaching-frame-dir",
        default=None,
        help="Optional coaching frame directory. Implies --capture-coaching-frames.",
    )
    parser.add_argument("--viz", action="store_true", help="Show exported Godot window when using --env-path.")
    return parser.parse_args()


def build_godot_env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    agent_role = getattr(args, "agent_role", "enemy")
    kwargs: dict[str, Any] = {
        "disable-coaching-frame-capture": "false"
        if args.capture_coaching_frames or args.coaching_frame_dir is not None
        else "true"
    }
    kwargs["agent-role"] = agent_role
    if agent_role == "player":
        kwargs["baseline-profile"] = getattr(args, "scripted_enemy_profile", "hard")
    else:
        kwargs["scripted-player-profile"] = getattr(args, "scripted_player_profile", "medium")
        kwargs["ppo-difficulty-profile"] = getattr(args, "ppo_difficulty_profile", "adaptive")
    if args.reward_profile != "default":
        kwargs["reward-profile"] = args.reward_profile
    if args.gameplay_log_dir is not None:
        kwargs["gameplay-log-dir"] = args.gameplay_log_dir
    if args.coaching_frame_dir is not None:
        kwargs["coaching-frame-dir"] = args.coaching_frame_dir
    return kwargs


def train_once(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
        from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing training dependency. From adaptive-fps-coach, run:\n"
            "  python -m pip install -e ../godot_rl_agents\n"
            "Then retry this command."
        ) from exc

    pathlib.Path(args.experiment_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.save_model_path).parent.mkdir(parents=True, exist_ok=True)

    env = StableBaselinesGodotEnv(
        env_path=args.env_path,
        n_parallel=args.n_parallel,
        show_window=args.viz,
        seed=args.seed,
        port=args.port,
        speedup=args.speedup,
        action_repeat=args.action_repeat,
        **build_godot_env_kwargs(args),
    )
    env = VecMonitor(env)

    if args.load_model_path is not None:
        model = PPO.load(args.load_model_path, env=env, tensorboard_log=args.experiment_dir)
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=2,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            tensorboard_log=args.experiment_dir,
        )

    callbacks = []
    if args.checkpoint_frequency > 0:
        checkpoint_dir = pathlib.Path(args.experiment_dir) / f"{args.experiment_name}_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_frequency // env.num_envs, 1),
                save_path=str(checkpoint_dir),
                name_prefix=args.experiment_name,
            )
        )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            tb_log_name=args.experiment_name,
            callback=callbacks or None,
            reset_num_timesteps=args.load_model_path is None,
        )
        mean_reward, std_reward = evaluate_policy(
            model,
            env,
            n_eval_episodes=max(args.eval_episodes, 1),
            deterministic=True,
        )
        model.save(args.save_model_path)
        print(f"Saved model to {args.save_model_path}")
        return {
            "experiment_name": args.experiment_name,
            "save_model_path": args.save_model_path,
            "load_model_path": args.load_model_path,
            "timesteps": args.timesteps,
            "seed": args.seed,
            "port": args.port,
            "n_parallel": args.n_parallel,
            "speedup": args.speedup,
            "action_repeat": args.action_repeat,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "ent_coef": args.ent_coef,
            "clip_range": args.clip_range,
            "eval_episodes": max(args.eval_episodes, 1),
            "reward_profile": args.reward_profile,
            "agent_role": args.agent_role,
            "scripted_enemy_profile": args.scripted_enemy_profile,
            "scripted_player_profile": args.scripted_player_profile,
            "ppo_difficulty_profile": args.ppo_difficulty_profile,
            "gameplay_log_dir": args.gameplay_log_dir,
            "capture_coaching_frames": args.capture_coaching_frames,
            "coaching_frame_dir": args.coaching_frame_dir,
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
        }
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    metadata = train_once(args)
    if args.save_metadata_path is not None:
        metadata_path = pathlib.Path(args.save_metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
