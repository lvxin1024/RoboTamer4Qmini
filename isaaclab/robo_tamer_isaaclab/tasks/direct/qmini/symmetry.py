"""Sagittal-plane symmetry transforms for the Qmini policy interface."""

from __future__ import annotations

import torch


POLICY_FRAME_SIZE = 43
ACTION_SIZE = 12


def mirror_policy_observation(observations: torch.Tensor) -> torch.Tensor:
    """Mirror one or more stacked 43-feature policy observations left-to-right."""
    if observations.shape[-1] % POLICY_FRAME_SIZE != 0:
        raise ValueError(
            f"Expected stacked {POLICY_FRAME_SIZE}-feature observations, got "
            f"{observations.shape[-1]} features"
        )

    source = observations.reshape(*observations.shape[:-1], -1, POLICY_FRAME_SIZE)
    mirrored = source.clone()

    # Command [forward, yaw], base [roll, pitch], and angular velocity [x, y, z].
    mirrored[..., 1] = -source[..., 1]
    mirrored[..., 2] = -source[..., 2]
    mirrored[..., 4] = -source[..., 4]
    mirrored[..., 6] = -source[..., 6]

    # Joint position, velocity, and target-error blocks. All Qmini joint axes
    # are mirrored with the opposite generalized-coordinate sign in the URDF.
    for start in (7, 17, 27):
        mirrored[..., start : start + 5] = -source[..., start + 5 : start + 10]
        mirrored[..., start + 5 : start + 10] = -source[..., start : start + 5]

    # Left/right phase sine, phase cosine, and frequency features.
    mirrored[..., 37] = source[..., 38]
    mirrored[..., 38] = source[..., 37]
    mirrored[..., 39] = source[..., 40]
    mirrored[..., 40] = source[..., 39]
    mirrored[..., 41] = source[..., 42]
    mirrored[..., 42] = source[..., 41]

    return mirrored.reshape_as(observations)


def mirror_policy_action(actions: torch.Tensor) -> torch.Tensor:
    """Mirror normalized 12-dimensional actor actions left-to-right."""
    if actions.shape[-1] != ACTION_SIZE:
        raise ValueError(f"Expected {ACTION_SIZE} policy actions, got {actions.shape[-1]}")

    mirrored = actions.clone()
    mirrored[..., 0] = actions[..., 1]
    mirrored[..., 1] = actions[..., 0]
    mirrored[..., 2:7] = -actions[..., 7:12]
    mirrored[..., 7:12] = -actions[..., 2:7]
    return mirrored
