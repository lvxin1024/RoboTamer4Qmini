"""Isaac Lab implementation of the Qmini locomotion environment."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu, RayCaster
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
)

from .qmini_env_cfg import IMU_ROT, QminiEnvCfg


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

        self._q_base_sensor = self._tensor(IMU_ROT).unsqueeze(0).repeat(self.num_envs, 1)
        self._q_sensor_base = quat_conjugate(self._q_base_sensor)

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
        self._joint_targets = self._default_joint_pos.clone()

        self._commands = torch.zeros(self.num_envs, 2, device=self.device)
        self._command_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._fixed_commands: torch.Tensor | None = None

        self._phase = torch.zeros(self.num_envs, 2, device=self.device)
        self._frequency = torch.full_like(self._phase, 0.5)
        self._obs_history = torch.zeros(self.num_envs, 3, 43, device=self.device)
        self._critic_obs_history = torch.zeros(self.num_envs, 3, 127, device=self.device)
        self._joint_target_history = self._default_joint_pos[:, None, :].repeat(1, 3, 1)
        self._network_action_history = torch.zeros(self.num_envs, 3, 12, device=self.device)

        self._kp_scale = torch.ones(self.num_envs, 10, device=self.device)
        self._kd_scale = torch.ones(self.num_envs, 10, device=self.device)
        self._torque_scale = torch.ones(self.num_envs, 10, device=self.device)

        max_delay = max(
            self.cfg.domain_randomization.joint_delay_steps[1],
            self.cfg.domain_randomization.rate_delay_steps[1],
            self.cfg.domain_randomization.angle_delay_steps[1],
        )
        self._delay_history_length = max_delay
        self._delay_history_index = 0
        self._joint_pos_history = torch.zeros(self.num_envs, max_delay, 10, device=self.device)
        self._joint_vel_history = torch.zeros_like(self._joint_pos_history)
        self._base_euler_history = torch.zeros(self.num_envs, max_delay, 3, device=self.device)
        self._base_ang_vel_history = torch.zeros_like(self._base_euler_history)
        self._base_acc_history = torch.zeros_like(self._base_euler_history)
        self._delay_joint_steps = 1
        self._delay_rate_steps = 1
        self._delay_angle_steps = 1
        self._sample_observation_delays()

        self._physics_step_count = 0
        self._joint_pos_noise = torch.zeros(self.num_envs, 10, device=self.device)
        self._joint_vel_noise = torch.zeros_like(self._joint_pos_noise)
        self._base_euler_noise = torch.zeros(self.num_envs, 3, device=self.device)
        self._base_ang_vel_noise = torch.zeros_like(self._base_euler_noise)
        self._base_acc_noise = torch.zeros_like(self._base_euler_noise)
        self._last_foot_force = torch.zeros(self.num_envs, 2, device=self.device)

    def _tensor(self, values: Sequence[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _imu_state_in_base(self):
        quat_world_sensor = self._imu_sensor.data.quat_w

        # q_world_base = q_world_sensor * inverse(q_base_sensor)
        quat_world_base = quat_mul(quat_world_sensor, self._q_sensor_base)

        # Sensor-frame vectors -> base_link-frame vectors.
        lin_vel_base = quat_apply(
            self._q_base_sensor, self._imu_sensor.data.lin_vel_b
        )
        ang_vel_base = quat_apply(
            self._q_base_sensor, self._imu_sensor.data.ang_vel_b
        )
        lin_acc_base = quat_apply(
            self._q_base_sensor, self._imu_sensor.data.lin_acc_b
        )

        return quat_world_base, lin_vel_base, ang_vel_base, lin_acc_base

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._imu_sensor = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self._imu_sensor
        self._left_foot_height_sensor = RayCaster(self.cfg.left_foot_height)
        self.scene.sensors["left_foot_height"] = self._left_foot_height_sensor
        self._right_foot_height_sensor = RayCaster(self.cfg.right_foot_height)
        self.scene.sensors["right_foot_height"] = self._right_foot_height_sensor
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
        self._actions = self._action_low + 0.5 * (actions + 1.0) * (
            self._action_high - self._action_low
        )
        self._network_action_history = torch.roll(
            self._network_action_history, shifts=-1, dims=1
        )
        self._network_action_history[:, -1].copy_(self._actions)

        self._frequency.copy_(self._actions[:, :2])
        self._phase.add_(2.0 * math.pi * self._frequency * self._control_dt)
        self._phase.remainder_(2.0 * math.pi)

        self._joint_targets.add_(self._actions[:, 2:] * self._control_dt)
        self._joint_targets = torch.clamp(
            self._joint_targets,
            self._joint_pos_limits[..., 0],
            self._joint_pos_limits[..., 1],
        )
        self._joint_target_history = torch.roll(
            self._joint_target_history, shifts=-1, dims=1
        )
        self._joint_target_history[:, -1].copy_(self._joint_targets)

        delay_interval = self.cfg.domain_randomization.delay_resample_control_steps
        if (
            self.cfg.domain_randomization.enabled
            and self.cfg.domain_randomization.delay_observation
            and self.common_step_counter > 0
            and self.common_step_counter % delay_interval == 0
        ):
            self._sample_observation_delays()

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
        self._record_physics_state()
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
        critic_obs = self._compute_critic_observation()
        self._obs_history = torch.roll(self._obs_history, shifts=-1, dims=1)
        self._obs_history[:, -1].copy_(policy_obs)
        self._critic_obs_history = torch.roll(
            self._critic_obs_history, shifts=-1, dims=1
        )
        self._critic_obs_history[:, -1].copy_(critic_obs)
        return {
            "policy": self._obs_history.flatten(start_dim=1),
            "critic": self._critic_obs_history.flatten(start_dim=1),
        }

    def _compute_policy_observation(self) -> torch.Tensor:
        joint_pos = self._delayed(self._joint_pos_history, self._delay_joint_steps)
        joint_vel = self._delayed(self._joint_vel_history, self._delay_joint_steps)
        base_euler = self._delayed(self._base_euler_history, self._delay_angle_steps)
        base_ang_vel = self._delayed(self._base_ang_vel_history, self._delay_rate_steps)
        base_rp = base_euler[:, :2]
        joint_error = self._joint_targets - joint_pos

        static = (torch.linalg.vector_norm(self._commands, dim=1, keepdim=True) >= 0.15).float()
        phase_features = torch.cat((torch.sin(self._phase), torch.cos(self._phase)), dim=-1)
        frequency_features = (self._frequency * 0.3 - 1.0) * static

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

    def _compute_critic_observation(self) -> torch.Tensor:
        state = self._legacy_state()
        delayed_joint_pos = self._delayed(self._joint_pos_history, self._delay_joint_steps)
        delayed_joint_vel = self._delayed(self._joint_vel_history, self._delay_joint_steps)
        delayed_euler = self._delayed(self._base_euler_history, self._delay_angle_steps)
        delayed_ang_vel = self._delayed(self._base_ang_vel_history, self._delay_rate_steps)
        joint_error = self._joint_targets - delayed_joint_pos
        static = (torch.linalg.vector_norm(self._commands, dim=1, keepdim=True) >= 0.15).float()
        phase_features = torch.cat((torch.sin(self._phase), torch.cos(self._phase)), dim=-1)
        joint_network_output = self._network_action_history[:, -1, 2:] / 15.0

        obs = torch.cat(
            (
                self._commands,
                self._commands[:, 0:1] - state["base_lin_vel"][:, 0:1],
                self._commands[:, 1:2] - state["base_ang_vel"][:, 2:3],
                state["base_lin_vel"],
                state["base_euler"][:, :2],
                state["base_ang_vel"] * 0.5,
                state["joint_pos"] - self._default_joint_pos,
                state["joint_vel"] * 0.1,
                self._joint_targets - self._default_joint_pos,
                joint_error,
                phase_features * static,
                (self._frequency * 0.3 - 1.0) * static,
                joint_network_output,
                torch.clamp(state["foot_height"], -0.5, 0.5) * 10.0,
                (state["base_pos_hd"][:, 2:3] - 0.4) * 10.0,
                torch.clamp(state["foot_vel"], -8.0, 8.0) * 0.5,
                torch.clamp(state["base_acc"], -20.0, 20.0) * 0.2,
                torch.clamp(state["foot_force"], 0.0, 200.0) * 0.01,
                joint_network_output,
                delayed_euler[:, :2],
                delayed_ang_vel * 0.5,
                delayed_joint_pos - self._default_joint_pos,
                delayed_joint_vel * 0.1,
                joint_error,
            ),
            dim=-1,
        )
        if obs.shape[-1] != 127:
            raise RuntimeError(f"Qmini critic observation changed from 127 to {obs.shape[-1]}")
        return obs

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg.rewards
        state = self._legacy_state()
        command_x = self._commands[:, 0:1]
        command_yaw = self._commands[:, 1:2]
        moving = (torch.linalg.vector_norm(self._commands, dim=1, keepdim=True) >= 0.15).float()
        lin_vel_x_norm = torch.clamp(torch.abs(command_x), 0.3, 2.0) + 0.2
        yaw_rate_norm = torch.clamp(torch.abs(command_yaw), 0.3, 1.5) + 0.2

        base_height_reward = torch.exp(-70.0 * (state["base_pos"][:, 2:3] - 0.45) ** 2)
        balance_reward = 0.5 * (
            base_height_reward
            * torch.exp(
                -torch.clamp(5.0 / lin_vel_x_norm, 2.0, 8.0)
                * torch.linalg.vector_norm(state["base_euler"][:, :2], dim=1, keepdim=True)
            )
            + 1.0
        )
        forward_velocity_reward = torch.exp(
            -torch.clamp(5.0 / lin_vel_x_norm, 2.0, 10.0)
            * (command_x - state["base_lin_vel"][:, 0:1]) ** 2
        )
        lateral_velocity_reward = torch.exp(
            -torch.clamp(5.0 / lin_vel_x_norm, 3.0, 15.0)
            * state["base_lin_vel"][:, 1:2] ** 2
        )
        lateral_velocity_reward += (
            -0.6
            / lin_vel_x_norm
            * torch.abs(state["base_lin_vel"][:, 1:2])
            * moving
        )
        yaw_rate_reward = torch.exp(
            -torch.clamp(2.0 / yaw_rate_norm, 2.0, 6.0)
            * (command_yaw - state["base_ang_vel"][:, 2:3]) ** 2
        )
        angular_velocity_reward = torch.exp(
            -torch.clamp(2.0 / lin_vel_x_norm, 0.7, 6.0)
            * torch.linalg.vector_norm(state["base_ang_vel"][:, :2], dim=1, keepdim=True) ** 2
        )
        delayed_acc = self._delayed(self._base_acc_history, self._delay_rate_steps)
        gravity = self._tensor([0.0, 0.0, 9.81])
        base_acceleration_reward = (
            -0.4
            / lin_vel_x_norm
            * torch.linalg.vector_norm((delayed_acc - gravity) * 0.1, dim=1, keepdim=True)
            * moving
        )
        vertical_velocity_reward = torch.exp(
            -torch.clamp(5.0 / lin_vel_x_norm, 2.0, 10.0)
            * state["base_lin_vel"][:, 2:3] ** 2
        )
        vertical_velocity_reward -= (
            0.2
            / lin_vel_x_norm
            * torch.linalg.vector_norm(state["base_lin_vel"][:, 1:], dim=1, keepdim=True)
            * moving
        )

        support_contact = state["foot_force"] >= cfg.contact_force_threshold
        clear_contact = state["foot_force"] < 1.0
        support_phase = self._support_mask()
        swing_phase = ~support_phase
        foot_clearance_reward = (
            torch.logical_and(clear_contact, swing_phase).float().sum(dim=1, keepdim=True) / 2.0
        ) * moving
        foot_support_reward = (
            torch.logical_and(support_contact, support_phase).float().sum(dim=1, keepdim=True) / 2.0
        ) * moving
        airborne_reward = -torch.logical_not(
            support_contact.any(dim=1, keepdim=True)
        ).float()
        swapped_support = torch.logical_xor(support_contact, support_phase).all(
            dim=1, keepdim=True
        )
        contact_phase_reward = -swapped_support.float() * moving

        foot_height_score = 40.0 * torch.clamp(state["foot_height"], 0.0, 0.05)
        foot_height_reward = torch.clamp(
            (swing_phase.float() * foot_height_score).sum(dim=1, keepdim=True), max=2.0
        ) * moving
        foot_height_reward += -20.0 * torch.clamp(
            state["foot_height"] - 0.06, min=0.0
        ).sum(dim=1, keepdim=True)
        foot_height_reward += (
            -0.2 * (support_phase.float() * foot_height_score).sum(dim=1, keepdim=True) * moving
        )
        foot_height_reward += (
            -0.2 * (support_contact.float() * foot_height_score).sum(dim=1, keepdim=True) * moving
        )

        twist_reward = -torch.linalg.vector_norm(
            state["base_euler"][:, :2], dim=1, keepdim=True
        )
        foot_force_acceleration = state["foot_force"] - self._last_foot_force
        foot_soft_contact_reward = (
            -0.1
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 1.5)
            * torch.linalg.vector_norm(foot_force_acceleration, dim=1, keepdim=True)
            / 100.0
        )
        foot_contact_force_reward = (
            -torch.linalg.vector_norm(
                state["foot_force"] * swing_phase.float(), dim=1, keepdim=True
            )
            * moving
        )
        foot_contact_force_reward += -torch.linalg.vector_norm(
            torch.clamp(
                torch.abs(state["foot_force"] - 55.0) * support_contact.float(), min=0.0
            ),
            dim=1,
            keepdim=True,
        )

        foot_velocity = state["foot_vel"].view(self.num_envs, 2, 3)
        clipped_foot_height = torch.abs(state["foot_height"]) + 0.03
        foot_slip_reward = 2.0 * torch.clamp(
            lin_vel_x_norm
            * (
                foot_velocity[:, :, 0]
                * torch.sign(command_x)
                * swing_phase.float()
            ).sum(dim=1, keepdim=True),
            0.0,
            1.0,
        ) * moving
        foot_slip_reward += (
            -0.5
            * torch.linalg.vector_norm(
                torch.linalg.vector_norm(foot_velocity[:, :, 1:2], dim=-1),
                dim=1,
                keepdim=True,
            )
            * moving
        )
        foot_slip_reward += (
            0.3
            * torch.linalg.vector_norm(
                torch.linalg.vector_norm(foot_velocity[:, :, :2], dim=-1),
                dim=1,
                keepdim=True,
            )
            * (moving - 1.0)
        )
        foot_slip_reward += (
            -0.3
            / lin_vel_x_norm
            * torch.linalg.vector_norm(
                0.1
                * torch.linalg.vector_norm(foot_velocity[:, :, :2], dim=-1)
                / clipped_foot_height
                * support_phase.float(),
                dim=1,
                keepdim=True,
            )
            * moving
        )
        foot_vertical_velocity_reward = (
            -0.1
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 1.0)
            * torch.linalg.vector_norm(
                torch.linalg.vector_norm(torch.clamp(foot_velocity[:, :, 2:3], max=0.0), dim=-1)
                / clipped_foot_height,
                dim=1,
                keepdim=True,
            )
            * moving
        )
        foot_vertical_velocity_reward += (
            0.8
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 1.0)
            * torch.linalg.vector_norm(
                torch.linalg.vector_norm(torch.clamp(foot_velocity[:, :, 2:3], max=0.0), dim=-1),
                dim=1,
                keepdim=True,
            )
            * (moving - 1.0)
        )
        foot_acceleration_reward = (
            -0.4
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 2.0)
            * torch.linalg.vector_norm(state["foot_vel"][:, [2, 5]], dim=1, keepdim=True)
        )

        action_smoothness_reward = (
            -0.3
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 2.0)
            * torch.linalg.vector_norm(
                self._joint_target_history[:, 0]
                - 2.0 * self._joint_target_history[:, 1]
                + self._joint_target_history[:, 2],
                dim=1,
                keepdim=True,
            )
        )
        network_smoothness_reward = (
            -0.2
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 2.0)
            * torch.linalg.vector_norm(
                self._network_action_history[:, 0, 2:]
                - 2.0 * self._network_action_history[:, 1, 2:]
                + self._network_action_history[:, 2, 2:],
                dim=1,
                keepdim=True,
            )
            ** 2
        )
        action_offset = self._joint_targets - self._default_joint_pos
        action_constraint_reward = (
            -0.1
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 1.0)
            * torch.linalg.vector_norm(action_offset, dim=1, keepdim=True)
        )
        action_constraint_reward += (
            -3.0
            * torch.linalg.vector_norm(action_offset[:, [0, 1, 5, 6]], dim=1, keepdim=True)
            * moving
        )
        stand_action_constraint_reward = (
            -0.1
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 1.0)
            * torch.linalg.vector_norm(action_offset, dim=1, keepdim=True) ** 2
            * moving
        )
        stand_action_constraint_reward += (
            -moving
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 1.0)
            * torch.linalg.vector_norm(
                (state["joint_pos"][:, :5] - self._default_joint_pos[:, :5])
                * support_contact[:, 0:1].float(),
                dim=1,
                keepdim=True,
            )
            ** 2
        )
        stand_action_constraint_reward += (
            -moving
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 1.0)
            * torch.linalg.vector_norm(
                (state["joint_pos"][:, 5:] - self._default_joint_pos[:, 5:])
                * support_contact[:, 1:2].float(),
                dim=1,
                keepdim=True,
            )
            ** 2
        )
        joint_position_error_reward = (
            -0.4
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 1.0)
            * torch.linalg.vector_norm(
                self._joint_targets - state["joint_pos"], dim=1, keepdim=True
            )
            ** 2
        )
        joint_velocity_reward = (
            -0.4
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 1.0)
            * torch.linalg.vector_norm(state["joint_vel"], dim=1, keepdim=True) ** 2
        )
        joint_velocity_reward += (
            -torch.clamp(1.0 / lin_vel_x_norm, 0.0, 1.0)
            * torch.linalg.vector_norm(
                state["joint_vel"][:, [0, 1, 5, 6]], dim=1, keepdim=True
            )
            ** 2
        )
        joint_torque_reward = (
            -0.4
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 2.0)
            * torch.clamp(
                torch.abs(state["joint_torque"]) - self._torque_limits, min=0.0
            ).sum(dim=1, keepdim=True)
            * moving
        )
        phase_modulator_reward = (
            -0.02
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 1.0)
            * torch.linalg.vector_norm(
                self._network_action_history[:, 0, :2]
                - 2.0 * self._network_action_history[:, 1, :2]
                + self._network_action_history[:, 2, :2],
                dim=1,
                keepdim=True,
            )
        )
        phase_modulator_reward += (
            -0.5
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 1.0)
            * torch.linalg.vector_norm(
                self._network_action_history[:, -1, :2] * support_phase.float(),
                dim=1,
                keepdim=True,
            )
            ** 2
        )
        phase_modulator_reward *= moving
        network_output_reward = (
            -0.4
            * torch.clamp(1.0 / lin_vel_x_norm, 0.0, 1.0)
            * torch.linalg.vector_norm(
                self._network_action_history[:, -1, 2:], dim=1, keepdim=True
            )
            ** 2
        )
        foot_orientation_reward = -0.5 * torch.linalg.vector_norm(
            state["foot_euler"][:, [1, 4]], dim=1, keepdim=True
        )
        leg_width_reward = -torch.linalg.vector_norm(
            torch.abs(
                state["foot_pos_hd"][:, [1, 4]] - state["base_pos_hd"][:, 1:2]
            )
            - 0.14,
            dim=1,
            keepdim=True,
        )
        phase_sin = torch.sin(self._phase)
        phase_cos = torch.cos(self._phase)
        foot_phase_reward = -torch.linalg.vector_norm(
            phase_sin[:, 0:1] + phase_sin[:, 1:2], dim=1, keepdim=True
        ) ** 2
        foot_phase_reward += -torch.linalg.vector_norm(
            phase_cos[:, 0:1] + phase_cos[:, 1:2], dim=1, keepdim=True
        ) ** 2
        foot_phase_reward *= moving

        terms = {
            "constant": torch.ones_like(command_x) * cfg.constant,
            "base_height": base_height_reward * cfg.base_height,
            "balance": balance_reward * cfg.balance,
            "forward_velocity": forward_velocity_reward * cfg.forward_velocity,
            "yaw_rate": yaw_rate_reward * cfg.yaw_rate,
            "lateral_velocity": lateral_velocity_reward * cfg.lateral_velocity,
            "vertical_velocity": vertical_velocity_reward * cfg.vertical_velocity,
            "angular_velocity": angular_velocity_reward * cfg.angular_velocity,
            "twist": twist_reward * cfg.twist,
            "base_acceleration": base_acceleration_reward * balance_reward * cfg.base_acceleration,
            "airborne": airborne_reward * cfg.airborne,
            "contact_phase": contact_phase_reward * cfg.contact_phase,
            "foot_clearance": foot_clearance_reward * cfg.foot_clearance,
            "foot_support": foot_support_reward * cfg.foot_support,
            "foot_height": foot_height_reward * cfg.foot_height,
            "leg_width": leg_width_reward * balance_reward * cfg.leg_width,
            "action_constraint": action_constraint_reward * balance_reward * cfg.action_constraint,
            "stand_action_constraint": stand_action_constraint_reward
            * balance_reward
            * cfg.stand_action_constraint,
            "foot_phase": foot_phase_reward * balance_reward * cfg.foot_phase,
            "joint_position_error": joint_position_error_reward
            * balance_reward
            * cfg.joint_position_error,
            "action_smoothness": action_smoothness_reward
            * balance_reward
            * cfg.action_smoothness,
            "network_smoothness": network_smoothness_reward
            * balance_reward
            * cfg.network_smoothness,
            "network_output": network_output_reward * balance_reward * cfg.network_output,
            "foot_slip": foot_slip_reward * balance_reward * cfg.foot_slip,
            "foot_vertical_velocity": foot_vertical_velocity_reward
            * balance_reward
            * cfg.foot_vertical_velocity,
            "foot_acceleration": foot_acceleration_reward
            * balance_reward
            * cfg.foot_acceleration,
            "foot_soft_contact": foot_soft_contact_reward
            * balance_reward
            * cfg.foot_soft_contact,
            "joint_velocity": joint_velocity_reward * balance_reward * cfg.joint_velocity,
            "foot_orientation": foot_orientation_reward
            * balance_reward
            * cfg.foot_orientation,
            "foot_contact_force": foot_contact_force_reward * cfg.foot_contact_force,
            "joint_torque": joint_torque_reward * cfg.joint_torque,
            "phase_modulator": phase_modulator_reward
            * balance_reward
            * cfg.phase_modulator,
        }
        clip_low, clip_high = cfg.term_clip
        clipped_terms = {
            name: torch.clamp(value, clip_low, clip_high) * self._control_dt
            for name, value in terms.items()
        }
        reward = torch.cat(tuple(clipped_terms.values()), dim=1).sum(dim=1)
        self.extras["log"] = {
            f"Reward/{name}": value.mean() for name, value in clipped_terms.items()
        }
        self._last_foot_force.copy_(state["foot_force"])
        return torch.clamp(reward, min=0.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._legacy_state()
        tipped = (torch.abs(state["base_euler"][:, 0]) > 0.7) | (
            torch.abs(state["base_euler"][:, 1]) > 0.7
        )
        too_low = state["base_pos_hd"][:, 2] < 0.2

        contact_forces = self._contact_sensor.data.net_forces_w_history[
            :, :, self._termination_sensor_ids
        ]
        illegal_contact = (
            torch.linalg.vector_norm(contact_forces, dim=-1).amax(dim=1).amax(dim=1)
            > 1.0
        )
        delayed_joint_pos = self._delayed(self._joint_pos_history, self._delay_joint_steps)
        at_low_limit = (
            (torch.abs(self._joint_targets - self._joint_pos_limits[..., 0]) < 0.02)
            & (torch.abs(delayed_joint_pos - self._joint_pos_limits[..., 0]) < 0.02)
        ).any(dim=1)
        at_high_limit = (
            (torch.abs(self._joint_targets - self._joint_pos_limits[..., 1]) < 0.02)
            & (torch.abs(delayed_joint_pos - self._joint_pos_limits[..., 1]) < 0.02)
        ).any(dim=1)
        terminated = tipped | too_low | illegal_contact | at_low_limit | at_high_limit
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self._terrain_curriculum_enabled() and self.common_step_counter > 0:
            self._update_terrain_curriculum(env_ids)

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]

        randomize = self.cfg.domain_randomization.enabled
        if randomize:
            root_state[:, :2] += self._uniform(-1.0, 1.0, (len(env_ids), 2))
            rpy = self._uniform(-0.2, 0.2, (len(env_ids), 3))
            root_state[:, 3:7] = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
            root_state[:, 7:9] = self._uniform(-0.5, 0.5, (len(env_ids), 2))
            root_state[:, 9:10] = self._uniform(-0.2, 0.2, (len(env_ids), 1))
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
        self._actions[env_ids] = 0.0
        self._joint_target_history[env_ids] = joint_pos[:, None, :]
        self._network_action_history[env_ids] = 0.0
        if randomize:
            self._phase[env_ids] = self._uniform(0.0, 2.0 * math.pi, (len(env_ids), 2))
        else:
            self._phase[env_ids] = 0.0
        self._frequency[env_ids] = 0.5
        self._obs_history[env_ids] = 0.0
        self._critic_obs_history[env_ids] = 0.0
        self._last_foot_force[env_ids] = 0.0
        self._fill_delay_history(env_ids, joint_pos, joint_vel)
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
        static_ids = env_ids[env_ids < min(self.cfg.command.static_envs, self.num_envs)]
        if len(static_ids):
            self._commands[static_ids] = 0.0
        self._commands[env_ids, 0:1] *= (
            torch.abs(self._commands[env_ids, 0:1]) >= 0.15
        )
        self._commands[env_ids, 1:2] *= (
            torch.abs(self._commands[env_ids, 1:2]) >= 0.15
        )
        self._command_age[env_ids] = 0

    def _push_robots(self, env_ids: torch.Tensor):
        root_velocity = self._robot.data.root_vel_w[env_ids].clone()
        curriculum_steps = self.cfg.domain_randomization.push_curriculum_control_steps
        curriculum_ratio = 1.0 + min(self.common_step_counter / curriculum_steps, 0.5)
        linear_limit = (
            self.cfg.domain_randomization.max_push_linear_velocity * curriculum_ratio
        )
        angular_limit = (
            self.cfg.domain_randomization.max_push_angular_velocity * curriculum_ratio
        )
        root_velocity[:, :3] = self._uniform(-linear_limit, linear_limit, (len(env_ids), 3))
        root_velocity[:, 3:] = self._uniform(-angular_limit, angular_limit, (len(env_ids), 3))
        self._robot.write_root_velocity_to_sim(root_velocity, env_ids)

    def _foot_contact_forces(self) -> torch.Tensor:
        forces = self._contact_sensor.data.net_forces_w[:, self._foot_sensor_ids]
        return torch.linalg.vector_norm(forces, dim=-1).clamp(max=1000.0)

    def _legacy_state(self) -> dict[str, torch.Tensor]:
        imu_quat_w, base_lin_vel, base_ang_vel, base_acc = self._imu_state_in_base()
        roll, pitch, yaw = euler_xyz_from_quat(imu_quat_w)
        base_euler = self._wrap_angle(torch.stack((roll, pitch, yaw), dim=-1))
        heading_quat = quat_from_euler_xyz(
            torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
        )

        foot_pos_w = self._robot.data.body_pos_w[:, self._foot_body_ids].clone()
        foot_pos_w[..., 2] -= 0.1
        foot_vel_w = self._robot.data.body_lin_vel_w[:, self._foot_body_ids]
        heading_foot = heading_quat[:, None, :].expand(-1, 2, -1).reshape(-1, 4)
        foot_pos_hd = quat_apply_inverse(
            heading_foot, foot_pos_w.reshape(-1, 3)
        ).reshape(self.num_envs, 6)
        foot_vel_hd = quat_apply_inverse(
            heading_foot, foot_vel_w.reshape(-1, 3)
        ).reshape(self.num_envs, 6)
        base_pos_hd = quat_apply_inverse(heading_quat, self._imu_sensor.data.pos_w)

        foot_quat_w = self._robot.data.body_quat_w[:, self._foot_body_ids]
        foot_roll, foot_pitch, foot_yaw = euler_xyz_from_quat(foot_quat_w.reshape(-1, 4))
        foot_euler = self._wrap_angle(
            torch.stack((foot_roll, foot_pitch, foot_yaw), dim=-1)
        ).reshape(self.num_envs, 6)

        ground_z = torch.stack(
            (
                self._left_foot_height_sensor.data.ray_hits_w[..., 2].mean(dim=1),
                self._right_foot_height_sensor.data.ray_hits_w[..., 2].mean(dim=1),
            ),
            dim=1,
        )
        fallback_ground_z = self._terrain.env_origins[:, 2:3].expand(-1, 2)
        ground_z = torch.where(torch.isfinite(ground_z), ground_z, fallback_ground_z)

        return {
            "base_pos": self._imu_sensor.data.pos_w,
            "base_pos_hd": base_pos_hd,
            "base_euler": base_euler,
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "base_acc": base_acc,
            "joint_pos": self._robot.data.joint_pos[:, self._joint_ids],
            "joint_vel": self._robot.data.joint_vel[:, self._joint_ids],
            "joint_torque": self._robot.data.applied_torque[:, self._joint_ids],
            "foot_pos_hd": foot_pos_hd,
            "foot_vel": foot_vel_hd,
            "foot_euler": foot_euler,
            "foot_height": foot_pos_w[..., 2] - ground_z,
            "foot_force": self._foot_contact_forces(),
        }

    def _record_physics_state(self):
        noise_interval = self.cfg.domain_randomization.noise_resample_physics_steps
        if self._physics_step_count % noise_interval == 0:
            self._sample_observation_noise()

        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids]
        imu_quat_w, _, base_ang_vel, base_acc = self._imu_state_in_base()
        roll, pitch, yaw = euler_xyz_from_quat(imu_quat_w)
        base_euler = self._wrap_angle(torch.stack((roll, pitch, yaw), dim=-1))
        index = self._delay_history_index
        self._joint_pos_history[:, index] = joint_pos + self._joint_pos_noise
        self._joint_vel_history[:, index] = joint_vel + self._joint_vel_noise
        self._base_euler_history[:, index] = base_euler + self._base_euler_noise
        self._base_ang_vel_history[:, index] = (
            base_ang_vel + self._base_ang_vel_noise
        )
        self._base_acc_history[:, index] = (
            base_acc + self._base_acc_noise
        )
        self._delay_history_index = (index + 1) % self._delay_history_length
        self._physics_step_count += 1

    def _fill_delay_history(
        self, env_ids: torch.Tensor, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ):
        state = self._legacy_state()
        self._joint_pos_history[env_ids] = joint_pos[:, None, :]
        self._joint_vel_history[env_ids] = joint_vel[:, None, :]
        self._base_euler_history[env_ids] = state["base_euler"][env_ids, None, :]
        self._base_ang_vel_history[env_ids] = state["base_ang_vel"][env_ids, None, :]
        self._base_acc_history[env_ids] = state["base_acc"][env_ids, None, :]

    def _sample_observation_noise(self):
        cfg = self.cfg.domain_randomization
        if not (cfg.enabled and cfg.observation_noise):
            self._joint_pos_noise.zero_()
            self._joint_vel_noise.zero_()
            self._base_euler_noise.zero_()
            self._base_ang_vel_noise.zero_()
            self._base_acc_noise.zero_()
            return
        self._joint_pos_noise.copy_(
            self._uniform(-cfg.joint_pos_noise, cfg.joint_pos_noise, self._joint_pos_noise.shape)
        )
        self._joint_vel_noise.copy_(
            self._uniform(-cfg.joint_vel_noise, cfg.joint_vel_noise, self._joint_vel_noise.shape)
        )
        self._base_euler_noise.copy_(
            self._uniform(-cfg.angle_noise, cfg.angle_noise, self._base_euler_noise.shape)
        )
        self._base_ang_vel_noise.copy_(
            self._uniform(
                -cfg.angular_velocity_noise,
                cfg.angular_velocity_noise,
                self._base_ang_vel_noise.shape,
            )
        )
        self._base_acc_noise.copy_(
            self._uniform(
                -cfg.linear_acceleration_noise,
                cfg.linear_acceleration_noise,
                self._base_acc_noise.shape,
            )
        )

    def _sample_observation_delays(self):
        cfg = self.cfg.domain_randomization
        if not (cfg.enabled and cfg.delay_observation):
            self._delay_joint_steps = 1
            self._delay_rate_steps = 1
            self._delay_angle_steps = 1
            return
        self._delay_joint_steps = self._sample_integer_range(cfg.joint_delay_steps)
        self._delay_rate_steps = self._sample_integer_range(cfg.rate_delay_steps)
        self._delay_angle_steps = self._sample_integer_range(cfg.angle_delay_steps)

    def _sample_integer_range(self, bounds: tuple[int, int]) -> int:
        return int(torch.randint(bounds[0], bounds[1] + 1, (1,), device=self.device).item())

    def _delayed(self, history: torch.Tensor, delay_steps: int) -> torch.Tensor:
        index = (self._delay_history_index - delay_steps) % self._delay_history_length
        return history[:, index]

    def _terrain_curriculum_enabled(self) -> bool:
        generator = self.cfg.terrain.terrain_generator
        return generator is not None and generator.curriculum

    def _update_terrain_curriculum(self, env_ids: torch.Tensor):
        distance = torch.linalg.vector_norm(
            self._robot.data.root_pos_w[env_ids, :2]
            - self._terrain.env_origins[env_ids, :2],
            dim=1,
        )
        terrain_length = self.cfg.terrain.terrain_generator.size[0]
        move_up = distance > terrain_length / 2.0
        expected_distance = (
            torch.abs(self._commands[env_ids, 0]) * self.max_episode_length_s * 0.5
        )
        move_down = (distance < expected_distance) & ~move_up
        self._terrain.update_env_origins(env_ids, move_up, move_down)
        self.extras.setdefault("log", {})["Curriculum/terrain_level"] = (
            self._terrain.terrain_levels.float().mean()
        )

    def _support_mask(self) -> torch.Tensor:
        return self._phase < 1.2 * math.pi

    def _uniform(self, low: float, high: float, shape: Sequence[int]) -> torch.Tensor:
        return low + (high - low) * torch.rand(*shape, device=self.device)

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))
