# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Thunder recovery environment configuration.

Based on 'Learning to Recover' (arXiv:2506.05516):
- Random fallen initialization (drop from 1.1m, 2s free-fall)
- Fixed 5s episode, no early termination
- Episode-based Dynamic Reward Shaping (ED)
- Flat terrain for Phase 0
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.thunder_hist.rough_env_cfg import (
    ThunderHistRoughEnvCfg,
    ThunderHistRoughRewardWeights,
)


@configclass
class RecoveryRewardWeights:
    """Reward weights for recovery task (Table I in paper)."""

    # Task rewards (multiplied by ED factor)
    stand_joint_pos: float = 42.0
    base_height: float = 120.0
    base_orientation: float = 50.0

    # Behavior rewards (collision + action_rate multiplied by CW)
    body_collision: float = -5e-2
    action_rate: float = -1e-2   # legs only, wheels excluded

    # Constant penalties
    joint_velocity: float = -2e-2
    torques: float = -2.5e-5
    joint_acceleration: float = -2.5e-7
    wheel_velocity: float = -2e-2


@configclass
class ThunderRecoveryEnvCfg(ThunderHistRoughEnvCfg):
    """Thunder recovery environment — Phase 0 (flat ground).

    Key differences from locomotion env:
    - No velocity tracking, no terrain curriculum
    - ED reward shaping: free exploration early, precise standing late
    - Random fallen initialization with 2s free-fall
    - Fixed 5s episode, no early termination
    """

    recovery_weights: RecoveryRewardWeights = RecoveryRewardWeights()

    def __post_init__(self):
        super().__post_init__()

        # ── Scene: flat terrain ──
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None

        # ── Episode: 5 seconds fixed ──
        self.episode_length_s = 5.0
        # Control at 50Hz (paper uses 50Hz)
        self.sim.dt = 0.005
        self.decimation = 4  # 50Hz control = 200Hz sim / 4

        # ── Initialization: random fallen state ──
        # Random orientation (full range including upside-down)
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (1.0, 1.2),  # drop from ~1.1m
            "roll": (-math.pi, math.pi),
            "pitch": (-math.pi, math.pi),
            "yaw": (-math.pi, math.pi),
        }
        # Zero velocity at spawn (gravity does the rest)
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        # ── Disable ALL locomotion rewards ──
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.upward.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.body_lin_acc_l2.weight = 0.0
        self.rewards.joint_torques_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = 0.0
        self.rewards.joint_pos_limits.weight = 0.0
        self.rewards.joint_power.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.joint_pos_penalty.weight = 0.0
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.action_rate_l2.weight = 0.0
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.contact_forces.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_slide.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_gait.weight = 0.0
        self.rewards.is_terminated.weight = 0.0

        # ── Disable ALL terminations except timeout ──
        self.terminations.illegal_contact = None

        # ── Disable ALL curriculum ──
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None
        self.curriculum.disturbance_levels = None
        self.curriculum.mass_randomization_levels = None
        self.curriculum.com_randomization_levels = None

        # ── Disable velocity commands (not needed for recovery) ──
        # Keep commands manager but set ranges to zero
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # ── Disable ALL DR for Phase 0 ──
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_push_robot = None

        # Remove height scan observations (no terrain)
        if hasattr(self.observations, 'height_scan_group'):
            self.observations.height_scan_group = None

        # Zero-weight all rewards, then re-enable with disable_zero_weight_rewards
        if self.__class__.__name__ == "ThunderRecoveryEnvCfg":
            self.disable_zero_weight_rewards()
