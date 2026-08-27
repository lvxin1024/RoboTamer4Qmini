# Isaac Lab backend

This repository keeps the original Isaac Gym Preview 3 backend and adds an
independent Isaac Lab backend. The port targets the public Isaac Lab API at
release `v2.3.2`, commit `37ddf626871758333d6ed89cf64ad702aef127d0`.

## What is implemented

- automatic Qmini URDF-to-USD conversion through `UrdfFileCfg`
- a vectorized `DirectRLEnv` task registered as `RoboTamer-Qmini-Direct-v0`
- the original 12-dimensional action contract:
  two phase frequencies and ten incremental joint targets
- the original 43-dimensional policy feature layout with three-frame stacking
  (`129` actor observations)
- explicit legacy-style PD torque control and URDF effort limits
- foot contact sensing, fall/contact termination, gait phase features, command
  sampling, random pushes, gain/torque randomization, and observation noise
- PPO training with the repository's existing actor, critic, rollout storage,
  and optimizer
- task registration includes an RSL-RL PPO configuration for standard Isaac
  Lab tooling
- evaluation of both newly trained checkpoints and legacy Actor checkpoints
  with the same `129 -> 12` network shape

The simulator dynamics are different, so a legacy checkpoint is useful as a
diagnostic or fine-tuning start; it is not expected to reproduce the Isaac Gym
gait without retraining.

## Install

Install Isaac Sim and Isaac Lab first, following the Isaac Lab release that
matches your Isaac Sim version. From the Isaac Lab repository root, install
this backend into Isaac Lab's Python environment:

```bash
./isaaclab.sh -p -m pip install -e /absolute/path/to/RoboTamer4Qmini/isaaclab
```

The Qmini meshes remain in this repository. Keep the checkout in place while
running the task because the URDF converter resolves them from
`assets/q1/meshes`.

## Smoke test

Run the first test with few environments. Isaac Lab performs the URDF-to-USD
conversion during the first launch, so that launch is slower.

```bash
./isaaclab.sh -p /absolute/path/to/RoboTamer4Qmini/scripts/isaaclab/train.py \
  --headless \
  --num_envs 64 \
  --max_iterations 2 \
  --name smoke_isaaclab
```

Then increase `--num_envs` through `512`, `2048`, and `4096` while monitoring
GPU memory and simulation FPS.

## Train

```bash
./isaaclab.sh -p /absolute/path/to/RoboTamer4Qmini/scripts/isaaclab/train.py \
  --headless \
  --num_envs 4096 \
  --max_iterations 5000 \
  --name qmini_isaaclab
```

Outputs are written below `experiments/<name>/isaaclab/`. Resume with:

```bash
./isaaclab.sh -p /absolute/path/to/RoboTamer4Qmini/scripts/isaaclab/train.py \
  --headless \
  --resume /absolute/path/to/policy.pt \
  --name qmini_isaaclab
```

## Play

Run without `--headless` to use the Isaac Sim viewer:

```bash
./isaaclab.sh -p /absolute/path/to/RoboTamer4Qmini/scripts/isaaclab/play.py \
  --checkpoint /absolute/path/to/policy.pt \
  --command_x 0.3 \
  --command_yaw 0.0
```

## Parity status

| Area | Status | Notes |
| --- | --- | --- |
| URDF and joint order | Implemented | Ten actuated joints are asserted at startup. |
| Actor action/observation shape | Implemented | `12` actions and `129` stacked observations. |
| PPO training and checkpointing | Implemented | Uses the existing RoboTamer PPO implementation. |
| Contact and gait rewards | Implemented | Consolidated for the initial Isaac Lab baseline. |
| Push, gain, torque, observation randomization | Implemented | Per-environment tensor randomization. |
| Friction, mass, inertia randomization | Implemented | Material and mass events; inertia is recomputed from mass. |
| Delayed observations | Pending parity | The actor shape is preserved, but delay queues are not enabled yet. |
| Rough terrain curriculum | Pending parity | The initial task uses `TerrainImporterCfg` with a plane. |
| IMU fixed-link state | Implemented | A native Isaac Lab IMU uses the `imu_in_torso_joint` URDF origin as an offset from `base_link`; the massless fixed visual is merged into the base during conversion. |
| Exact legacy reward curve | Pending parity | Reward terms were reduced to a stable training baseline. |

Do not compare final rewards between backends until the pending parity items are
implemented. First compare startup pose, joint order, contact labels, command
tracking, base height, termination rate, and per-term reward magnitudes.

## Cloud instance guidance

For this task, an RTX 3090 24 GB or A10 24 GB is a practical baseline. Use an
Ubuntu image supported by the selected Isaac Sim/Isaac Lab release. A cloud
image that only contains legacy Isaac Gym is not sufficient for this backend.
