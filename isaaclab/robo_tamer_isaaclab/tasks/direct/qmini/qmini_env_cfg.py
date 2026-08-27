"""Configuration for the Isaac Lab Qmini environment."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.envs.mdp as mdp
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
QMINI_URDF = str(REPOSITORY_ROOT / "assets" / "q1" / "urdf" / "q1.urdf")


def _imu_offset_from_urdf() -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Read the fixed GY-91 frame from the URDF as position and wxyz quaternion."""
    root = ET.parse(QMINI_URDF).getroot()
    joint = root.find("./joint[@name='imu_in_torso_joint']")
    if joint is None or joint.get("type") != "fixed":
        raise ValueError("q1.urdf must define imu_in_torso_joint as a fixed joint")
    origin = joint.find("origin")
    if origin is None:
        raise ValueError("imu_in_torso_joint must define an origin")

    pos = tuple(float(value) for value in origin.get("xyz", "0 0 0").split())
    roll, pitch, yaw = (float(value) for value in origin.get("rpy", "0 0 0").split())
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    rot = (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )
    return pos, rot


IMU_POS, IMU_ROT = _imu_offset_from_urdf()


@configclass
class CommandCfg:
    """Velocity command sampling."""

    resampling_time_s: float = 5.0
    lin_vel_x: tuple[float, float] = (-0.3, 0.7)
    yaw_vel: tuple[float, float] = (-1.0, 1.0)


@configclass
class DomainRandomizationCfg:
    """Randomization implemented directly in the environment."""

    enabled: bool = True
    gain_scale: tuple[float, float] = (0.8, 1.2)
    torque_scale: tuple[float, float] = (0.8, 1.2)
    push_interval_s: float = 3.0
    max_push_linear_velocity: float = 0.5
    max_push_angular_velocity: float = 0.5
    observation_noise: bool = True


@configclass
class RewardCfg:
    """Weights for the first Isaac Lab parity target."""

    alive: float = 0.3
    track_linear_velocity: float = 2.3
    track_yaw_velocity: float = 2.5
    upright: float = 1.5
    base_height: float = 1.0
    vertical_velocity: float = 0.6
    gait_contact: float = 0.7
    foot_clearance: float = 0.5
    foot_slip: float = -0.15
    action_rate: float = -0.02
    joint_velocity: float = -0.0005
    torque: float = -0.00002


@configclass
class EventCfg:
    """Isaac Lab startup randomization for physical properties."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.2, 1.5),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    body_mass_and_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )


@configclass
class QminiEnvCfg(DirectRLEnvCfg):
    """DirectRLEnv configuration preserving the legacy policy interface."""

    decimation = 15
    episode_length_s = 10.0

    # 12 actions: two gait frequencies followed by ten joint increments.
    action_space = 12
    # 43 policy features stacked across three control steps.
    observation_space = 129
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=0.001,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        replicate_physics=True,
    )
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.05,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=QMINI_URDF,
            fix_base=False,
            # The GY-91 is massless and rigidly mounted. Its visual is merged
            # into base_link while the native IMU uses the URDF-derived offset.
            merge_fixed_joints=True,
            self_collision=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                ),
            ),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.45),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "hip_yaw_l": 0.4,
                "hip_roll_l": -0.1,
                "hip_pitch_l": -1.5,
                "knee_pitch_l": 1.0,
                "ankle_pitch_l": -1.3,
                "hip_yaw_r": -0.4,
                "hip_roll_r": 0.1,
                "hip_pitch_r": 1.5,
                "knee_pitch_r": -1.0,
                "ankle_pitch_r": 1.3,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=60.0,
                velocity_limit_sim=30.0,
                stiffness=0.0,
                damping=0.0,
            )
        },
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
    )
    # Isaac Lab recommends an offset from an existing rigid body instead of a
    # small-mass fixed body. The offset is loaded from the URDF fixed joint.
    imu: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=ImuCfg.OffsetCfg(pos=IMU_POS, rot=IMU_ROT),
        update_period=0.0,
        history_length=0,
    )

    events: EventCfg = EventCfg()
    command: CommandCfg = CommandCfg()
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()
    rewards: RewardCfg = RewardCfg()
