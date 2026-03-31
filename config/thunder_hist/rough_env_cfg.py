# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Thunder wheeled robot with 5-frame history observations in rough terrain."""

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from robot_lab.assets.thunder import THUNDER_V3_CFG  # isort: skip


@configclass
class ThunderHistRoughCommandParams:
    """Command parameters for Thunder Hist Rough environment."""

    # Velocity command ranges
    lin_vel_x: tuple = (-2.5, 2.5)
    lin_vel_y: tuple = (-1.0, 1.0)
    ang_vel_z: tuple = (-1.0, 1.0)


@configclass
class ThunderHistRoughEventParams:
    """Event randomization parameters for Thunder Hist Rough environment."""

    # Reset base randomization
    reset_base_pose_range_x: tuple = (-0.5, 0.5)
    reset_base_pose_range_y: tuple = (-0.5, 0.5)
    reset_base_pose_range_z: tuple = (0.0, 0.2)
    reset_base_pose_range_roll: tuple = (-3.14, 3.14)
    reset_base_pose_range_pitch: tuple = (-3.14, 3.14)
    reset_base_pose_range_yaw: tuple = (-3.14, 3.14)

    reset_base_velocity_range_x: tuple = (-0.5, 0.5)
    reset_base_velocity_range_y: tuple = (-0.5, 0.5)
    reset_base_velocity_range_z: tuple = (-0.5, 0.5)
    reset_base_velocity_range_roll: tuple = (-0.5, 0.5)
    reset_base_velocity_range_pitch: tuple = (-0.5, 0.5)
    reset_base_velocity_range_yaw: tuple = (-0.5, 0.5)

    # External force/torque randomization — disabled for Phase 1 baseline
    external_force_range: tuple = (0.0, 0.0)
    external_torque_range: tuple = (0.0, 0.0)

    # Mass randomization (optional, can be added if needed)
    mass_base_distribution_params: tuple = None  # Not used in rough_env_cfg
    mass_others_distribution_params: tuple = None  # Not used in rough_env_cfg


@configclass
class ThunderHistRoughRewardWeights:
    """Reward weights configuration for Thunder Hist Rough environment."""

    # General
    is_terminated: float = 0.0

    # Tracking rewards
    track_lin_vel_xy_exp: float = 6.0
    track_ang_vel_z_exp: float = 3.0
    upward: float = 2.0

    # Root penalties
    lin_vel_z_l2: float = -2.0
    ang_vel_xy_l2: float = -0.05
    flat_orientation_l2: float = -0.1
    base_height_l2: float = 0.0
    body_lin_acc_l2: float = 0.0

    # Joint penalties
    joint_torques_l2: float = -1e-5
    joint_torques_wheel_l2: float = 0.0
    joint_vel_l2: float = 0.0
    joint_vel_wheel_l2: float = 0.0
    joint_acc_l2: float = -1e-7
    joint_acc_wheel_l2: float = 0
    joint_pos_limits: float = -2.0
    joint_vel_limits: float = 0.0
    joint_power: float = -2e-5
    stand_still: float = -2.0
    joint_pos_penalty: float = -1.0
    wheel_vel_penalty: float = 0.0
    joint_mirror: float = -0.05

    # Action penalties
    action_rate_l2: float = -0.01
    action_smoothness_l2: float = 0.0

    # Contact penalties
    undesired_contacts: float = -1.0
    contact_forces: float = -5e-4

    # Other rewards
    feet_air_time: float = 0.0
    feet_contact: float = 0.0
    feet_contact_without_cmd: float = 0.1
    feet_stumble: float = -5.0
    feet_slide: float = 0.0
    feet_height: float = 0.0
    feet_height_body: float = 0.0
    feet_gait: float = 0.0


@configclass
class ThunderHistObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy观测配置 - 带5帧历史信息"""

        # 基座角速度 - 5帧历史 (3维 * 5帧 = 15维)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )

        # 投影重力向量 - 5帧历史 (3维 * 5帧 = 15维)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 速度命令 - 5帧历史 (3维 * 5帧 = 15维)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 关节位置观测 - 5帧历史 (16关节 * 5帧 = 80维)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 关节速度观测 - 5帧历史 (16关节 * 5帧 = 80维)
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )

        # 上一步动作 - 5帧历史 (16动作 * 5帧 = 80维)
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5  # 为整个Policy观测组设置5帧历史

    @configclass
    class CriticCfg(ObsGroup):
        """Critic观测配置 - 带历史信息"""

        # 基座线速度 - 5帧历史 (3维 * 5帧 = 15维)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 基座角速度 - 5帧历史 (3维 * 5帧 = 15维)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 投影重力向量 - 5帧历史 (3维 * 5帧 = 15维)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 速度命令 - 5帧历史 (3维 * 5帧 = 15维)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 关节位置观测 - 5帧历史 (16关节 * 5帧 = 80维)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 关节速度观测 - 5帧历史 (16关节 * 5帧 = 80维)
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # 上一步动作 - 5帧历史 (16动作 * 5帧 = 80维)
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class HeightScanCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.05},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

    height_scan_group: HeightScanCfg = HeightScanCfg()


@configclass
class ThunderHistActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "fr_hip_joint",
            "fr_thigh_joint",
            "fr_calf_joint",
            "fl_hip_joint",
            "fl_thigh_joint",
            "fl_calf_joint",
            "rr_hip_joint",
            "rr_thigh_joint",
            "rr_calf_joint",
            "rl_hip_joint",
            "rl_thigh_joint",
            "rl_calf_joint",
        ],
        scale={
            ".*_hip_joint": 0.125,
            "^(?!.*_hip_joint).*": 0.25,
        },
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[
            "fr_foot_joint",
            "fl_foot_joint",
            "rr_foot_joint",
            "rl_foot_joint",
        ],
        scale=5.0,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )


@configclass
class ThunderHistRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )


@configclass
class ThunderHistRoughActuatorGains:
    """Actuator PD gains configuration."""

    hip_stiffness: float = 70.0
    hip_damping: float = 15.0
    thigh_stiffness: float = 100.0
    thigh_damping: float = 15.0
    calf_stiffness: float = 120.0
    calf_damping: float = 20.0
    wheel_stiffness: float = 0.0
    wheel_damping: float = 1.0


@configclass
class ThunderHistRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Thunder Hist (NoHead) Rough Environment Configuration"""

    observations: ThunderHistObservationsCfg = ThunderHistObservationsCfg()
    actions: ThunderHistActionsCfg = ThunderHistActionsCfg()
    rewards: ThunderHistRewardsCfg = ThunderHistRewardsCfg()
    reward_weights: ThunderHistRoughRewardWeights = ThunderHistRoughRewardWeights()
    event_params: ThunderHistRoughEventParams = ThunderHistRoughEventParams()
    command_params: ThunderHistRoughCommandParams = ThunderHistRoughCommandParams()
    actuator_gains: ThunderHistRoughActuatorGains = ThunderHistRoughActuatorGains()

    base_link_name = "base_link"
    foot_link_name = ".*_foot_Link"

    # fmt: off
    leg_joint_names = [
        "fr_hip_joint", "fr_thigh_joint", "fr_calf_joint",
        "fl_hip_joint", "fl_thigh_joint", "fl_calf_joint",
        "rr_hip_joint", "rr_thigh_joint", "rr_calf_joint",
        "rl_hip_joint", "rl_thigh_joint", "rl_calf_joint",
    ]
    wheel_joint_names = [
        "fr_foot_joint", "fl_foot_joint", "rr_foot_joint", "rl_foot_joint",
    ]
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = THUNDER_V3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # pylint: disable=no-member
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # Initial state configuration
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.55)
        self.scene.robot.init_state.joint_pos = {
            "fr_hip_joint": -0.1,
            "fr_thigh_joint": -0.8,
            "fr_calf_joint": 1.8,
            "fl_hip_joint": 0.1,
            "fl_thigh_joint": 0.8,
            "fl_calf_joint": -1.8,
            "rr_hip_joint": 0.1,
            "rr_thigh_joint": 0.8,
            "rr_calf_joint": -1.8,
            "rl_hip_joint": -0.1,
            "rl_thigh_joint": -0.8,
            "rl_calf_joint": 1.8,
            ".*_foot_joint": 0.0,
        }
        self.scene.robot.init_state.joint_vel = {".*": 0.0}

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        if "wheel_asset_cfg" in self.observations.policy.joint_pos.params:
            self.observations.policy.joint_pos.params["wheel_asset_cfg"].joint_names = self.wheel_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        if "wheel_asset_cfg" in self.observations.critic.joint_pos.params:
            self.observations.critic.joint_pos.params["wheel_asset_cfg"].joint_names = self.wheel_joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125,
            "^(?!.*_hip_joint).*": 0.25,
        }
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        # Apply event randomization parameters
        e = self.event_params

        # Reset base randomization
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": e.reset_base_pose_range_x,
                "y": e.reset_base_pose_range_y,
                "z": e.reset_base_pose_range_z,
                "roll": e.reset_base_pose_range_roll,
                "pitch": e.reset_base_pose_range_pitch,
                "yaw": e.reset_base_pose_range_yaw,
            },
            "velocity_range": {
                "x": e.reset_base_velocity_range_x,
                "y": e.reset_base_velocity_range_y,
                "z": e.reset_base_velocity_range_z,
                "roll": e.reset_base_velocity_range_roll,
                "pitch": e.reset_base_velocity_range_pitch,
                "yaw": e.reset_base_velocity_range_yaw,
            },
        }

        # ── Phase 1: ALL DR disabled (add later after baseline) ──
        self.events.randomize_rigid_body_material = None       # friction
        self.events.randomize_rigid_body_mass_base = None      # mass base
        self.events.randomize_rigid_body_mass_others = None    # mass others
        self.events.randomize_com_positions = None             # COM
        self.events.randomize_apply_external_force_torque = None  # external force
        self.events.randomize_actuator_gains = None            # actuator gains
        self.events.randomize_push_robot = None                # push



        # ------------------------------Rewards------------------------------
        # Apply reward weights
        w = self.reward_weights

        # General
        self.rewards.is_terminated.weight = w.is_terminated

        # Tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = w.track_lin_vel_xy_exp  # 使用Body Frame跟踪
        self.rewards.track_ang_vel_z_exp.weight = w.track_ang_vel_z_exp
        self.rewards.upward.weight = w.upward

        # Root penalties (调整为与B2W一致)
        self.rewards.lin_vel_z_l2.weight = w.lin_vel_z_l2
        self.rewards.ang_vel_xy_l2.weight = w.ang_vel_xy_l2
        self.rewards.flat_orientation_l2.weight = w.flat_orientation_l2
        self.rewards.base_height_l2.weight = w.base_height_l2
        self.rewards.base_height_l2.params["target_height"] = 0.60
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = w.body_lin_acc_l2
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = w.joint_torques_l2
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = w.joint_torques_wheel_l2
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = w.joint_vel_l2
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = w.joint_vel_wheel_l2
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = w.joint_acc_l2
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = w.joint_acc_wheel_l2
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_pos_limits.weight = w.joint_pos_limits
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = w.joint_vel_limits
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = w.joint_power  # 调整为与B2W一致
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still.weight = w.stand_still
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_pos_penalty.weight = w.joint_pos_penalty
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.wheel_vel_penalty.weight = w.wheel_vel_penalty
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_mirror.weight = w.joint_mirror
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["fr_(hip|thigh|calf).*", "rl_(hip|thigh|calf).*"],
            ["fl_(hip|thigh|calf).*", "rr_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = w.action_rate_l2

        # Contact penalties
        self.rewards.undesired_contacts.weight = w.undesired_contacts
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]

        self.rewards.contact_forces.weight = w.contact_forces
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Other rewards
        self.rewards.feet_air_time.weight = w.feet_air_time
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = w.feet_contact
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = w.feet_contact_without_cmd
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = w.feet_stumble
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = w.feet_slide
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = w.feet_height
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = w.feet_height_body
        self.rewards.feet_height_body.params["target_height"] = -0.4
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = w.feet_gait
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (
            ("fl_foot_Link", "rr_foot_Link"),
            ("fr_foot_Link", "rl_foot_Link"),
        )

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "ThunderHistRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None

        # 禁用其他课程学习，专注于基本运动训练
        # self.curriculum.command_levels — enabled (velocity curriculum)
        self.curriculum.disturbance_levels = None
        self.curriculum.mass_randomization_levels = None
        self.curriculum.com_randomization_levels = None
        # Unlock full terrain difficulty (default max=5 in curriculums.py)
        self.curriculum.terrain_levels.params["max_terrain_level"] = 10


        # ------------------------------Commands------------------------------
        # Apply command parameters
        c: ThunderHistRoughCommandParams = self.command_params

        # Velocity command ranges
        self.commands.base_velocity.ranges.lin_vel_x = c.lin_vel_x
        self.commands.base_velocity.ranges.lin_vel_y = c.lin_vel_y
        self.commands.base_velocity.ranges.ang_vel_z = c.ang_vel_z
