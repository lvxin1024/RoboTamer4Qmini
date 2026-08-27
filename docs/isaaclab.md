# Isaac Lab backend

This repository keeps the original Isaac Gym Preview 3 backend and adds an
independent Isaac Lab backend. The parity port is tested with Isaac Lab `2.2.1`
and Isaac Sim `5.0`.

## What is implemented

- automatic Qmini URDF-to-USD conversion through `UrdfFileCfg`
- a vectorized `DirectRLEnv` task registered as `RoboTamer-Qmini-Direct-v0`
- the original 12-dimensional action contract:
  two phase frequencies and ten incremental joint targets
- the original 43-dimensional policy feature layout with three-frame stacking
  (`129` actor observations)
- the original asymmetric critic layout with three-frame stacking
  (`381` privileged critic observations)
- explicit legacy-style PD torque control and URDF effort limits
- foot contact sensing, fall/contact termination, gait phase features, command
  sampling, random pushes, gain/torque randomization, observation noise, and
  1 ms observation-delay queues
- the complete BIRL reward composition, including per-term clipping, and the
  rough random-height terrain distribution used by the legacy configuration
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
  --validate \
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

Parity checkpoints use a `381`-input critic. Older Isaac Lab baseline
checkpoints used a `129`-input critic and cannot be resumed as full PPO
checkpoints; their actor weights can still be loaded separately for evaluation.

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
| Contact and gait rewards | Implemented | All weighted BIRL terms and legacy per-term clipping are present. |
| Asymmetric critic | Implemented | `127` privileged features are stacked to `381`; the actor remains `129`. |
| Push, gain, torque, observation randomization | Implemented | Legacy ranges and update intervals are preserved. |
| Friction, mass, inertia randomization | Implemented | Material and mass events; inertia is recomputed from mass. |
| Delayed observations | Implemented | Joint, angle, and rate queues run at the 1 ms physics rate. |
| Rough terrain curriculum | Implemented | Random-height terrain matches the legacy 0-4 cm distribution; curriculum remains disabled by default, as in `main`. |
| IMU fixed-link state | Implemented | A native Isaac Lab IMU uses the `imu_in_torso_joint` URDF origin as an offset from `base_link`; the massless fixed visual is merged into the base during conversion. |
| Legacy reward curve | Implemented | Formula and weight parity; raw contact magnitudes still differ between physics engines. |

Isaac Gym and Isaac Lab use different PhysX integrations, so identical reward
numbers are not expected. Compare command tracking, base height, termination
rate, foot behavior, and per-term reward trends instead of raw totals alone.

## Cloud instance guidance

For this task, an RTX 3090 24 GB or A10 24 GB is a practical baseline. Use an
Ubuntu image supported by the selected Isaac Sim/Isaac Lab release. A cloud
image that only contains legacy Isaac Gym is not sufficient for this backend.
