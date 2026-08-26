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
    AppLauncher.add_app_launcher_args(parser)
    return parser


ARGS = build_parser().parse_args()
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "isaaclab"))

import torch

from model import load_actor
from robo_tamer_isaaclab.tasks.direct.qmini import QminiEnv, QminiEnvCfg


def main():
    cfg = QminiEnvCfg()
    cfg.scene.num_envs = ARGS.num_envs
    cfg.sim.device = ARGS.device
    cfg.domain_randomization.enabled = False
    cfg.domain_randomization.observation_noise = False
    cfg.events = None
    cfg.episode_length_s = max(ARGS.time, cfg.episode_length_s)
    env = QminiEnv(cfg)
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

    obs_dict, _ = env.reset(seed=ARGS.seed)
    obs = obs_dict["policy"]
    steps = int(ARGS.time / (cfg.sim.dt * cfg.decimation))
    try:
        for _ in range(steps):
            if not SIMULATION_APP.is_running():
                break
            with torch.inference_mode():
                actions = actor(obs)["act"]
            obs_dict, _, _, _, _ = env.step(actions)
            obs = obs_dict["policy"]
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        SIMULATION_APP.close()
