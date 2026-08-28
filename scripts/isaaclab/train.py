"""Train RoboTamer's PPO policy in Isaac Lab."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from collections import deque
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="qmini_isaaclab")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--steps_per_env", type=int, default=24)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Fail immediately if actions, observations, critic states, or rewards become non-finite.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


ARGS = build_parser().parse_args()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "isaaclab"))

import torch
from torch.utils.tensorboard import SummaryWriter

from model import load_actor, load_critic
from rl.alg import PPO
from robo_tamer_isaaclab.tasks.direct.qmini import QminiEnv, QminiEnvCfg


def main():
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed_all(ARGS.seed)

    cfg = QminiEnvCfg()
    cfg.scene.num_envs = ARGS.num_envs
    cfg.sim.device = ARGS.device
    cfg.seed = ARGS.seed
    # Train on a flat plane; keep the reusable rough-terrain preset in the task config.
    cfg.terrain.terrain_type = "plane"
    cfg.terrain.terrain_generator = None
    env = QminiEnv(cfg)

    run_dir = REPOSITORY_ROOT / "experiments" / ARGS.name / "isaaclab"
    model_dir = run_dir / "model"
    checkpoint_dir = model_dir / "all"
    log_dir = run_dir / "log"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "backend": "isaaclab",
        "task": "RoboTamer-Qmini-Direct-v0",
        "num_envs": ARGS.num_envs,
        "policy_observations": 129,
        "critic_observations": 381,
        "actions": 12,
        "seed": ARGS.seed,
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    policy_cfg = {
        "name": "simple_policy",
        "num_observations": 129,
        "num_actions": 12,
        "hidden_layers": (512, 256),
        "activation": "relu",
    }
    critic_cfg = {
        "name": "simple_policy",
        "num_critic_obs": 381,
        "hidden_layers": (512, 256),
        "activation": "relu",
    }
    actor = load_actor(policy_cfg, env.device)
    critic = load_critic(critic_cfg, env.device)
    algorithm = PPO(
        actor,
        critic,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        discount_factor=0.995,
        gae_lambda=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0005,
        max_grad_norm=1.0,
        desired_kl=0.01,
        eps_clip=0.2,
        use_clipped_value_loss=True,
        schedule="adaptive",
        device=env.device,
    )
    algorithm.init_storage(
        ARGS.num_envs,
        ARGS.steps_per_env,
        [381],
        [129],
        [12],
    )

    start_iteration = 0
    if ARGS.resume:
        checkpoint = torch.load(ARGS.resume, map_location=env.device, weights_only=False)
        actor.load_state_dict(checkpoint["actor"])
        critic.load_state_dict(checkpoint["critic"])
        algorithm.optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = int(checkpoint["iteration"]) + 1

    reward_history = deque(maxlen=100)
    length_history = deque(maxlen=100)
    episode_rewards = torch.zeros(ARGS.num_envs, device=env.device)
    episode_lengths = torch.zeros(ARGS.num_envs, device=env.device)
    writer = SummaryWriter(str(log_dir))

    obs_dict, _ = env.reset(seed=ARGS.seed)
    obs = obs_dict["policy"]
    critic_obs = obs_dict["critic"]
    total_time = 0.0
    print(
        f"Starting PPO at iteration {start_iteration}; target={ARGS.max_iterations}, "
        f"steps_per_env={ARGS.steps_per_env}, policy_obs={tuple(obs.shape)}, "
        f"critic_obs={tuple(critic_obs.shape)}",
        flush=True,
    )
    last_env_log: dict[str, torch.Tensor] = {}

    try:
        for iteration in range(start_iteration, ARGS.max_iterations):
            collection_start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(ARGS.steps_per_env):
                    actions = algorithm.act(obs, critic_obs)
                    next_obs_dict, rewards, terminated, truncated, info = env.step(actions)
                    dones = terminated | truncated
                    if ARGS.validate:
                        tensors = {
                            "actions": actions,
                            "policy observations": next_obs_dict["policy"],
                            "critic observations": next_obs_dict["critic"],
                            "rewards": rewards,
                        }
                        for name, value in tensors.items():
                            if not torch.isfinite(value).all():
                                raise FloatingPointError(f"Non-finite {name} at iteration {iteration}")
                    algorithm.process_env_step(
                        rewards,
                        dones,
                        {"timeouts": truncated.float()},
                    )

                    episode_rewards += rewards
                    episode_lengths += 1
                    reset_ids = torch.nonzero(dones, as_tuple=False).flatten()
                    if len(reset_ids):
                        reward_history.extend(episode_rewards[reset_ids].cpu().tolist())
                        length_history.extend(episode_lengths[reset_ids].cpu().tolist())
                        episode_rewards[reset_ids] = 0.0
                        episode_lengths[reset_ids] = 0.0
                    obs = next_obs_dict["policy"]
                    critic_obs = next_obs_dict["critic"]
                    last_env_log = info.get("log", {})

            algorithm.compute_returns(critic_obs)
            collection_time = time.perf_counter() - collection_start
            learn_start = time.perf_counter()
            value_loss, surrogate_loss, mean_kl = algorithm.update()
            learning_time = time.perf_counter() - learn_start
            iteration_time = collection_time + learning_time
            total_time += iteration_time

            checkpoint = {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "optimizer": algorithm.optimizer.state_dict(),
                "iteration": iteration,
                "backend": "isaaclab",
            }
            torch.save(checkpoint, model_dir / "policy.pt")
            if iteration % ARGS.save_interval == 0:
                torch.save(checkpoint, checkpoint_dir / f"policy_{iteration}.pt")

            mean_reward = statistics.mean(reward_history) if reward_history else 0.0
            mean_length = statistics.mean(length_history) if length_history else 0.0
            fps = int(ARGS.steps_per_env * ARGS.num_envs / max(iteration_time, 1.0e-6))
            writer.add_scalar("Train/mean_reward", mean_reward, iteration)
            writer.add_scalar("Train/mean_episode_length", mean_length, iteration)
            writer.add_scalar("Loss/value", value_loss, iteration)
            writer.add_scalar("Loss/surrogate", surrogate_loss, iteration)
            writer.add_scalar("Loss/mean_kl", mean_kl, iteration)
            writer.add_scalar("Perf/fps", fps, iteration)
            for name, value in last_env_log.items():
                writer.add_scalar(name, value, iteration)

            print(
                f"{ARGS.name}#{iteration}  t={total_time / 60:.1f}m  fps={fps}  "
                f"reward={mean_reward:.3f}  length={mean_length:.0f}  "
                f"value={value_loss:.4f}  policy={surrogate_loss:.4f}  kl={mean_kl:.4f}",
                flush=True,
            )
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        SIMULATION_APP.close()
