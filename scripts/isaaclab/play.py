"""Evaluate a legacy or Isaac Lab RoboTamer checkpoint in Isaac Lab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--time", type=float, default=20.0)
    parser.add_argument("--command_x", type=float, default=0.3)
    parser.add_argument("--command_yaw", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--free_camera",
        action="store_true",
        help="Center on env 0 initially, then allow manual viewport navigation.",
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

from isaaclab.utils.math import quat_apply_inverse
from model import load_actor
from robo_tamer_isaaclab.tasks.direct.qmini import QminiEnv, QminiEnvCfg


def main():
    cfg = QminiEnvCfg()
    cfg.scene.num_envs = ARGS.num_envs
    cfg.sim.device = ARGS.device
    # Match training playback to the flat-plane training environment.
    cfg.terrain.terrain_type = "plane"
    cfg.terrain.terrain_generator = None
    cfg.domain_randomization.enabled = False
    cfg.domain_randomization.observation_noise = False
    cfg.events = None
    cfg.episode_length_s = max(ARGS.time, cfg.episode_length_s)
    cfg.viewer.eye = (2.5, 2.5, 1.6)
    cfg.viewer.lookat = (0.0, 0.0, 0.45)
    cfg.viewer.env_index = 0
    if ARGS.free_camera:
        cfg.viewer.origin_type = "env"
    else:
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
    env = QminiEnv(cfg)
    print("[play] Environment constructed.", flush=True)
    env.set_command(ARGS.command_x, ARGS.command_yaw)

    actor = load_actor(
        {
            "name": "simple_policy",
            "num_observations": 129,
            "num_actions": 12,
            "hidden_layers": (512, 256),
            "activation": "relu",
        },
        env.device,
    ).eval()
    checkpoint = torch.load(ARGS.checkpoint, map_location=env.device, weights_only=False)
    actor.load_state_dict(checkpoint["actor"])
    print(
        f"[play] Loaded {ARGS.checkpoint} (iteration={checkpoint.get('iteration', 'unknown')}).",
        flush=True,
    )

    print("[play] Resetting environment...", flush=True)
    obs_dict, _ = env.reset(seed=ARGS.seed)
    obs = obs_dict["policy"]
    initial_pos_w = env._robot.data.root_pos_w[0].clone()
    initial_quat_w = env._robot.data.root_quat_w[0].clone()
    steps = int(ARGS.time / (cfg.sim.dt * cfg.decimation))
    report_every = max(1, round(1.0 / (cfg.sim.dt * cfg.decimation)))
    reset_count = 0
    print(
        f"[play] Running {steps} control steps: command_x={ARGS.command_x}, "
        f"command_yaw={ARGS.command_yaw}.",
        flush=True,
    )
    try:
        for step in range(steps):
            if not SIMULATION_APP.is_running():
                break
            with torch.inference_mode():
                actions = actor(obs)["act"]
            obs_dict, _, terminated, truncated, _ = env.step(actions)
            obs = obs_dict["policy"]
            reset_count += int((terminated | truncated)[0].item())
            if step % report_every == 0 or step == steps - 1:
                delta_w = env._robot.data.root_pos_w[0] - initial_pos_w
                delta_b = quat_apply_inverse(
                    initial_quat_w.unsqueeze(0), delta_w.unsqueeze(0)
                )[0]
                print(
                    f"[play] t={step * cfg.sim.dt * cfg.decimation:5.1f}s "
                    f"base_delta=(forward={delta_b[0].item():+.3f}, "
                    f"left={delta_b[1].item():+.3f}, up={delta_b[2].item():+.3f}) "
                    f"mean_abs_action={actions[0].abs().mean().item():.3f} "
                    f"resets={reset_count}",
                    flush=True,
                )
        print("[play] Playback finished.", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        SIMULATION_APP.close()
