"""Isaac Lab implementation of the Qmini locomotion environment."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from .qmini_env_cfg import QminiEnvCfg


class QminiEnv(DirectRLEnv):
    """Vectorized Qmini task with the legacy 12-action/129-observation contract."""

    JOINT_NAMES = (
        "hip_yaw_l",
        "hip_roll_l",
        "hip_pitch_l",
        "knee_pitch_l",
        "ankle_pitch_l",
        "hip_yaw_r",
        "hip_roll_r",
        "hip_pitch_r",
        "knee_pitch_r",
        "ankle_pitch_r",
    )

    def __init__(self, cfg: QminiEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, joint_names = self._robot.find_joints(list(self.JOINT_NAMES), preserve_order=True)
        if tuple(joint_names) != self.JOINT_NAMES:
            raise RuntimeError(f"Unexpected Qmini joint order: {joint_names}")

        self._foot_body_ids, _ = self._robot.find_bodies(
            ["ankle_pitch_l", "ankle_pitch_r"], preserve_order=True
        )
        self._foot_sensor_ids, _ = self._contact_sensor.find_bodies(
            ["ankle_pitch_l", "ankle_pitch_r"], preserve_order=True
        )
        self._termination_sensor_ids, _ = self._contact_sensor.find_bodies(
            ["base_link", "hip_.*", "knee_.*"], preserve_order=True
        )

        self._control_dt = self.cfg.sim.dt * self.cfg.decimation
        self._command_resample_steps = max(
            1, round(self.cfg.command.resampling_time_s / self._control_dt)
        )
        self._push_interval_steps = max(
            1, round(self.cfg.domain_randomization.push_interval_s / self._control_dt)
        )

        self._default_joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids].clone()
        self._joint_pos_limits = self._robot.data.soft_joint_pos_limits[:, self._joint_ids].clone()

        self._kp = self._tensor([55.0, 105.0, 75.0, 45.0, 30.0] * 2)
        self._kd = self._tensor([0.3, 2.5, 0.3, 0.5, 0.25] * 2)
        self._torque_limits = self._tensor([20.0, 60.0, 20.0, 20.0, 20.0] * 2)
        self._torque_offset = self._tensor(
            [0.6, 1.0, 0.0, 0.7, 0.0, -0.6, -1.0, 0.0, -0.7, 0.0]
        )
        self._velocity_sign = self._tensor([0.0, 1.0, 0.0, 0.0, 0.0] * 2)

        self._action_low = self._tensor([0.5, 0.5] + [-15.0] * 10)
        self._action_high = self._tensor([3.5, 3.5] + [15.0] * 10)
        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._previous_joint_targets = self._default_joint_pos.clone()
        self._joint_targets = self._default_joint_pos.clone()

        self._commands = torch.zeros(self.num_envs, 2, device=self.device)
        self._command_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._fixed_commands: torch.Tensor | None = None

        self._phase = torch.zeros(self.num_envs, 2, device=self.device)
        self._frequency = torch.full_like(self._phase, 0.5)
        self._obs_history = torch.zeros(self.num_envs, 3, 43, device=self.device)

        self._kp_scale = torch.ones(self.num_envs, 10, device=self.device)
        self._kd_scale = torch.ones(self.num_envs, 10, device=self.device)
        self._torque_scale = torch.ones(self.num_envs, 10, device=self.device)

    def _tensor(self, values: Sequence[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._imu_sensor = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self._imu_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def set_command(self, linear_x: float, yaw: float):
        """Hold one command for evaluation instead of randomly resampling it."""
        command = self._tensor([linear_x, yaw]).repeat(self.num_envs, 1)
        self._fixed_commands = command
        self._commands.copy_(command)

    def clear_fixed_command(self):
        """Return to random command sampling."""
        self._fixed_commands = None
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = torch.clamp(actions, -1.0, 1.0)
        self._previous_actions.copy_(self._actions)
        self._actions = self._action_low + 0.5 * (actions + 1.0) * (
            self._action_high - self._action_low
        )

        self._frequency.copy_(self._actions[:, :2])
        self._phase.add_(2.0 * math.pi * self._frequency * self._control_dt)
        self._phase.remainder_(2.0 * math.pi)

        self._previous_joint_targets.copy_(self._joint_targets)
        self._joint_targets.add_(self._actions[:, 2:] * self._control_dt)
        self._joint_targets = torch.clamp(
            self._joint_targets,
            self._joint_pos_limits[..., 0],
            self._joint_pos_limits[..., 1],
        )

        self._command_age.add_(1)
        expired = torch.nonzero(
            self._command_age >= self._command_resample_steps, as_tuple=False
        ).flatten()
        if len(expired) and self._fixed_commands is None:
            self._resample_commands(expired)

        if self.cfg.domain_randomization.enabled:
            push_ids = torch.nonzero(
                (self.episode_length_buf > 0)
                & (self.episode_length_buf % self._push_interval_steps == 0),
                as_tuple=False,
            ).flatten()
            if len(push_ids):
                self._push_robots(push_ids)

    def _apply_action(self):
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
        position_error = self._joint_targets - joint_pos
        torque = (
            self._kp * self._kp_scale * position_error
            - self._kd * self._kd_scale * joint_vel
            + self._torque_offset
            - 3.5 * torch.sign(joint_vel) * self._velocity_sign
        )
        torque = torch.clamp(torque, -self._torque_limits, self._torque_limits)
        torque.mul_(self._torque_scale)
        torque = torch.clamp(torque, -self._torque_limits, self._torque_limits)
        self._robot.set_joint_effort_target(torque, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        policy_obs = self._compute_policy_observation()
        self._obs_history = torch.roll(self._obs_history, shifts=-1, dims=1)
        self._obs_history[:, -1].copy_(policy_obs)
        stacked_obs = self._obs_history.flatten(start_dim=1)
        return {"policy": stacked_obs}

    def _compute_policy_observation(self) -> torch.Tensor:
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
        # Read the native IMU using the fixed-joint offset parsed from the URDF.
        # The visual-only mesh transform does not alter the sensor axes.
        imu_pos_w = self._imu_sensor.data.pos_w
        imu_quat_w = self._imu_sensor.data.quat_w
        base_ang_vel = self._imu_sensor.data.ang_vel_b
        self._imu_pos_w = imu_pos_w
        roll, pitch, _ = euler_xyz_from_quat(imu_quat_w)
        base_rp = torch.stack((self._wrap_angle(roll), self._wrap_angle(pitch)), dim=-1)
        joint_error = self._joint_targets - joint_pos

        static = (torch.linalg.vector_norm(self._commands, dim=1, keepdim=True) >= 0.15).float()
        phase_features = torch.cat((torch.sin(self._phase), torch.cos(self._phase)), dim=-1)
        frequency_features = (self._frequency * 0.3 - 1.0) * static

        if (
            self.cfg.domain_randomization.enabled
            and self.cfg.domain_randomization.observation_noise
        ):
            base_rp = base_rp + 0.03 * (2.0 * torch.rand_like(base_rp) - 1.0)
            base_ang_vel = base_ang_vel + 0.3 * (2.0 * torch.rand_like(base_ang_vel) - 1.0)
            joint_pos = joint_pos + 0.1 * (2.0 * torch.rand_like(joint_pos) - 1.0)
            joint_vel = joint_vel + 1.2 * (2.0 * torch.rand_like(joint_vel) - 1.0)

        obs = torch.cat(
            (
                self._commands,
                base_rp,
                base_ang_vel * 0.5,
                joint_pos - self._default_joint_pos,
                joint_vel * 0.1,
                joint_error,
                phase_features * static,
                frequency_features,
            ),
            dim=-1,
        )
        if obs.shape[-1] != 43:
            raise RuntimeError(f"Qmini policy observation changed from 43 to {obs.shape[-1]}")
        return torch.clamp(obs, -3.0, 3.0)

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg.rewards
        root_lin_vel = self._robot.data.root_lin_vel_b
        root_ang_vel = self._robot.data.root_ang_vel_b
        root_pos = self._robot.data.root_pos_w
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)

        linear_error = torch.square(self._commands[:, 0] - root_lin_vel[:, 0])
        yaw_error = torch.square(self._commands[:, 1] - root_ang_vel[:, 2])
        upright_error = torch.square(self._wrap_angle(roll)) + torch.square(
            self._wrap_angle(pitch)
        )
        height_error = torch.square(root_pos[:, 2] - 0.45)

        foot_forces = self._foot_contact_forces()
        foot_contact = foot_forces >= 10.0
        support = self._support_mask()
        gait_contact = torch.logical_and(foot_contact, support).float().mean(dim=1)

        foot_pos = self._robot.data.body_pos_w[:, self._foot_body_ids]
        foot_height = foot_pos[..., 2] - self._terrain.env_origins[:, None, 2]
        swing = ~support
        clearance = torch.clamp(foot_height, 0.0, 0.06) / 0.06
        clearance_reward = (clearance * swing.float()).mean(dim=1)

        foot_velocity = self._robot.data.body_lin_vel_w[:, self._foot_body_ids]
        foot_slip = (
            torch.square(foot_velocity[..., :2]).sum(dim=-1) * foot_contact.float()
        ).mean(dim=1)

        joint_velocity = self._robot.data.joint_vel[:, self._joint_ids]
        applied_torque = self._robot.data.applied_torque[:, self._joint_ids]
        action_rate = torch.square(self._actions - self._previous_actions).sum(dim=1)

        terms = {
            "alive": torch.ones(self.num_envs, device=self.device) * cfg.alive,
            "track_linear_velocity": torch.exp(-5.0 * linear_error)
            * cfg.track_linear_velocity,
            "track_yaw_velocity": torch.exp(-2.0 * yaw_error) * cfg.track_yaw_velocity,
            "upright": torch.exp(-5.0 * upright_error) * cfg.upright,
            "base_height": torch.exp(-70.0 * height_error) * cfg.base_height,
            "vertical_velocity": torch.exp(-5.0 * torch.square(root_lin_vel[:, 2]))
            * cfg.vertical_velocity,
            "gait_contact": gait_contact * cfg.gait_contact,
            "foot_clearance": clearance_reward * cfg.foot_clearance,
            "foot_slip": foot_slip * cfg.foot_slip,
            "action_rate": action_rate * cfg.action_rate,
            "joint_velocity": torch.square(joint_velocity).sum(dim=1) * cfg.joint_velocity,
            "torque": torch.square(applied_torque).sum(dim=1) * cfg.torque,
        }
        reward = torch.stack(tuple(terms.values()), dim=0).sum(dim=0) * self._control_dt
        self.extras["log"] = {
            f"Reward/{name}": value.mean() * self._control_dt for name, value in terms.items()
        }
        return torch.clamp(reward, min=0.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos = self._robot.data.root_pos_w
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        tipped = (torch.abs(self._wrap_angle(roll)) > 0.7) | (
            torch.abs(self._wrap_angle(pitch)) > 0.7
        )
        too_low = root_pos[:, 2] < 0.2

        contact_forces = self._contact_sensor.data.net_forces_w_history[
            :, :, self._termination_sensor_ids
        ]
        illegal_contact = (
            torch.linalg.vector_norm(contact_forces, dim=-1).amax(dim=1).amax(dim=1)
            > 1.0
        )
        terminated = tipped | too_low | illegal_contact
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]

        randomize = self.cfg.domain_randomization.enabled
        if randomize:
            root_state[:, :2] += self._uniform(-0.25, 0.25, (len(env_ids), 2))
            rpy = self._uniform(-0.2, 0.2, (len(env_ids), 3))
            root_state[:, 3:7] = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
            root_state[:, 7:10] = self._uniform(-0.5, 0.5, (len(env_ids), 3))
            root_state[:, 10:13] = self._uniform(-0.5, 0.5, (len(env_ids), 3))

        joint_pos = self._default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        if randomize:
            joint_pos += self._uniform(-0.1, 0.1, joint_pos.shape)
            joint_vel += self._uniform(-2.0, 2.0, joint_vel.shape)
        joint_pos = torch.clamp(
            joint_pos,
            self._joint_pos_limits[env_ids, :, 0],
            self._joint_pos_limits[env_ids, :, 1],
        )

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, self._joint_ids, env_ids)

        self._joint_targets[env_ids] = joint_pos
        self._previous_joint_targets[env_ids] = joint_pos
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        if randomize:
            self._phase[env_ids] = self._uniform(0.0, 2.0 * math.pi, (len(env_ids), 2))
        else:
            self._phase[env_ids] = 0.0
        self._frequency[env_ids] = 0.5
        self._obs_history[env_ids] = 0.0
        if self._fixed_commands is None:
            self._resample_commands(env_ids)
        else:
            self._commands[env_ids] = self._fixed_commands[env_ids]
            self._command_age[env_ids] = 0

        if randomize:
            gain_low, gain_high = self.cfg.domain_randomization.gain_scale
            torque_low, torque_high = self.cfg.domain_randomization.torque_scale
            self._kp_scale[env_ids] = self._uniform(gain_low, gain_high, (len(env_ids), 10))
            self._kd_scale[env_ids] = self._uniform(gain_low, gain_high, (len(env_ids), 10))
            self._torque_scale[env_ids] = self._uniform(
                torque_low, torque_high, (len(env_ids), 10)
            )
        else:
            self._kp_scale[env_ids] = 1.0
            self._kd_scale[env_ids] = 1.0
            self._torque_scale[env_ids] = 1.0

    def _resample_commands(self, env_ids: torch.Tensor):
        self._commands[env_ids, 0] = self._uniform(
            self.cfg.command.lin_vel_x[0], self.cfg.command.lin_vel_x[1], (len(env_ids),)
        )
        self._commands[env_ids, 1] = self._uniform(
            self.cfg.command.yaw_vel[0], self.cfg.command.yaw_vel[1], (len(env_ids),)
        )
        self._commands[env_ids] *= (
            torch.linalg.vector_norm(self._commands[env_ids], dim=1, keepdim=True) >= 0.15
        )
        self._command_age[env_ids] = 0

    def _push_robots(self, env_ids: torch.Tensor):
        root_velocity = self._robot.data.root_vel_w[env_ids].clone()
        linear_limit = self.cfg.domain_randomization.max_push_linear_velocity
        angular_limit = self.cfg.domain_randomization.max_push_angular_velocity
        root_velocity[:, :3] = self._uniform(-linear_limit, linear_limit, (len(env_ids), 3))
        root_velocity[:, 3:] = self._uniform(-angular_limit, angular_limit, (len(env_ids), 3))
        self._robot.write_root_velocity_to_sim(root_velocity, env_ids)

    def _foot_contact_forces(self) -> torch.Tensor:
        forces = self._contact_sensor.data.net_forces_w[:, self._foot_sensor_ids]
        return torch.linalg.vector_norm(forces, dim=-1).clamp(max=1000.0)

    def _support_mask(self) -> torch.Tensor:
        return self._phase < 1.2 * math.pi

    def _uniform(self, low: float, high: float, shape: Sequence[int]) -> torch.Tensor:
        return low + (high - low) * torch.rand(*shape, device=self.device)

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))
