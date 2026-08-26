"""Gym registration for the Qmini locomotion task."""

import gymnasium as gym

from . import agents

gym.register(
    id="RoboTamer-Qmini-Direct-v0",
    entry_point=f"{__name__}.qmini_env:QminiEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_env_cfg:QminiEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QminiPPORunnerCfg",
    },
)

from .qmini_env import QminiEnv
from .qmini_env_cfg import QminiEnvCfg

__all__ = ["QminiEnv", "QminiEnvCfg"]
