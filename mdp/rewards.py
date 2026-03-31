# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_heading_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking the target heading angle using exponential kernel.

    Only active for heading-mode envs (is_heading_env=True).
    heading_error = wrap_to_pi(heading_target - robot.heading_w).
    Same upright gate as track_ang_vel_z_world_exp.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd_term = env.command_manager.get_term(command_name)
    heading_error = math_utils.wrap_to_pi(cmd_term.heading_target - asset.data.heading_w)
    reward = torch.exp(-torch.square(heading_error) / std**2)
    # only reward heading-mode envs
    reward = torch.where(cmd_term.is_heading_env, reward, torch.zeros_like(reward))
    # upright gate
    reward *= torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_body_height_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of body height command using exponential kernel.

    This reward encourages the robot to match the commanded body height,
    enabling behaviors like crouching and standing tall.
    """
    # extract the used quantities
    asset: RigidObject = env.scene[asset_cfg.name]
    # get the commanded height (shape: num_envs,)
    target_height = env.command_manager.get_command(command_name)[:, 0]
    # get current body height
    current_height = asset.data.root_link_pos_w[:, 2]
    # compute height error
    height_error = torch.square(current_height - target_height)
    # exponential reward
    reward = torch.exp(-height_error / std**2)
    # scale by upright penalty (only give reward when robot is upright)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_standing_posture_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of standing posture command using exponential kernel.

    Posture commands (one-hot encoded, shape: num_envs x 5):
    - [1,0,0,0,0]: Normal standing (gravity points down: [0, 0, -1])
    - [0,1,0,0,0]: Handstand (inverted, gravity points up: [0, 0, 1])
    - [0,0,1,0,0]: Left side standing (gravity points left: [0, -1, 0])
    - [0,0,0,1,0]: Right side standing (gravity points right: [0, 1, 0])
    - [0,0,0,0,1]: Front two legs standing (pitch up ~45deg, gravity: [0.7, 0, -0.7])
    """
    # extract the used quantities
    asset: RigidObject = env.scene[asset_cfg.name]
    # get the commanded posture (one-hot encoded, shape: num_envs x 5)
    posture_cmd_onehot = env.command_manager.get_command(command_name)
    # convert one-hot to indices (shape: num_envs,)
    posture_cmd = torch.argmax(posture_cmd_onehot, dim=1)
    # get current gravity direction in body frame (shape: num_envs, 3)
    gravity_b = asset.data.projected_gravity_b

    # Define target gravity directions for each posture
    target_gravity = torch.zeros(env.num_envs, 3, device=env.device)

    # Posture 0: Normal standing (gravity down)
    mask_0 = posture_cmd == 0
    target_gravity[mask_0] = torch.tensor([0.0, 0.0, -1.0], device=env.device)

    # Posture 1: Handstand (gravity up)
    mask_1 = posture_cmd == 1
    target_gravity[mask_1] = torch.tensor([0.0, 0.0, 1.0], device=env.device)

    # Posture 2: Left side standing
    mask_2 = posture_cmd == 2
    target_gravity[mask_2] = torch.tensor([0.0, -1.0, 0.0], device=env.device)

    # Posture 3: Right side standing
    mask_3 = posture_cmd == 3
    target_gravity[mask_3] = torch.tensor([0.0, 1.0, 0.0], device=env.device)

    # Posture 4: Front two legs standing (pitched up ~45deg)
    mask_4 = posture_cmd == 4
    target_gravity[mask_4] = torch.tensor([0.7071, 0.0, -0.7071], device=env.device)

    # Compute error as L2 distance between current and target gravity direction
    gravity_error = torch.sum(torch.square(gravity_b - target_gravity), dim=1)

    # Exponential reward
    reward = torch.exp(-gravity_error / (2 * std**2))

    return reward


def posture_stability_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize angular velocity to encourage stable postures."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # Penalize all angular velocities (roll, pitch, yaw)
    ang_vel = asset.data.root_ang_vel_b
    penalty = torch.sum(torch.square(ang_vel), dim=1)
    return penalty


def posture_feet_height_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking specific foot heights for each posture.

    This ensures feet are at appropriate heights for each posture:
    - Normal standing: all feet at ground (~0.0m)
    - Handstand: all feet elevated (~0.5m)
    - Side standing: half feet at ground, half elevated (~0.4m)
    - Front standing: front feet elevated (~0.5m), back at ground
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    posture_cmd_onehot = env.command_manager.get_command(command_name)
    posture_cmd = torch.argmax(posture_cmd_onehot, dim=1)

    # Get feet heights in world frame
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (num_envs, num_feet)

    # Define target heights for each foot in each posture
    # Assuming body_ids order: [FR, FL, RR, RL]
    target_heights = torch.zeros(env.num_envs, len(asset_cfg.body_ids), device=env.device)

    # Posture 0: Normal standing - all feet on ground
    mask_0 = posture_cmd == 0
    target_heights[mask_0] = 0.05  # slightly above ground

    # Posture 1: Handstand - all feet in air
    mask_1 = posture_cmd == 1
    target_heights[mask_1] = 0.5

    # Posture 2: Left side standing - right feet on ground, left feet in air
    mask_2 = posture_cmd == 2
    target_heights[mask_2, 0] = 0.05  # FR on ground
    target_heights[mask_2, 2] = 0.05  # RR on ground
    target_heights[mask_2, 1] = 0.4  # FL in air
    target_heights[mask_2, 3] = 0.4  # RL in air

    # Posture 3: Right side standing - left feet on ground, right feet in air
    mask_3 = posture_cmd == 3
    target_heights[mask_3, 1] = 0.05  # FL on ground
    target_heights[mask_3, 3] = 0.05  # RL on ground
    target_heights[mask_3, 0] = 0.4  # FR in air
    target_heights[mask_3, 2] = 0.4  # RR in air

    # Posture 4: Front two legs standing - back feet on ground, front feet in air
    mask_4 = posture_cmd == 4
    target_heights[mask_4, 2] = 0.05  # RR on ground
    target_heights[mask_4, 3] = 0.05  # RL on ground
    target_heights[mask_4, 0] = 0.5  # FR in air
    target_heights[mask_4, 1] = 0.5  # FL in air

    # Compute height error
    height_error = torch.sum(torch.square(feet_height - target_heights), dim=1)

    # Exponential reward
    reward = torch.exp(-height_error / (2 * std**2))

    return reward


def posture_feet_contact_consistency(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward feet contact states matching the commanded posture.

    Ensures correct feet are in contact with ground for each posture.
    """
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    posture_cmd_onehot = env.command_manager.get_command(command_name)
    posture_cmd = torch.argmax(posture_cmd_onehot, dim=1)

    # Get contact forces (num_envs, num_feet)
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2]
    contact_forces = net_forces.max(dim=1)[0]
    is_contact = (contact_forces > threshold).float()  # (num_envs, num_feet)

    # Define target contact patterns for each posture
    # Assuming body_ids order: [FR, FL, RR, RL]
    target_contact = torch.zeros(env.num_envs, len(sensor_cfg.body_ids), device=env.device)

    # Posture 0: Normal standing - all feet in contact
    mask_0 = posture_cmd == 0
    target_contact[mask_0] = 1.0

    # Posture 1: Handstand - no feet in contact
    mask_1 = posture_cmd == 1
    target_contact[mask_1] = 0.0

    # Posture 2: Left side - right feet in contact
    mask_2 = posture_cmd == 2
    target_contact[mask_2, 0] = 1.0  # FR
    target_contact[mask_2, 2] = 1.0  # RR

    # Posture 3: Right side - left feet in contact
    mask_3 = posture_cmd == 3
    target_contact[mask_3, 1] = 1.0  # FL
    target_contact[mask_3, 3] = 1.0  # RL

    # Posture 4: Front standing - back feet in contact
    mask_4 = posture_cmd == 4
    target_contact[mask_4, 2] = 1.0  # RR
    target_contact[mask_4, 3] = 1.0  # RL

    # Compute contact consistency (binary accuracy)
    contact_match = (is_contact == target_contact).float()
    reward = contact_match.mean(dim=1)  # Average over all feet

    return reward


def posture_joint_symmetry(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint asymmetry for postures that should be symmetric.

    Normal standing, handstand should have left-right symmetry.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    posture_cmd_onehot = env.command_manager.get_command(command_name)
    posture_cmd = torch.argmax(posture_cmd_onehot, dim=1)

    # Get joint positions (assuming standard quadruped joint order)
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # Assuming joint order: [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
    #                         RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
    if joint_pos.shape[1] >= 12:
        # Front-left vs Front-right symmetry
        fr_joints = joint_pos[:, 0:3]
        fl_joints = joint_pos[:, 3:6]
        # Rear-left vs Rear-right symmetry
        rr_joints = joint_pos[:, 6:9]
        rl_joints = joint_pos[:, 9:12]

        # Compute symmetry error (mirror joints should have opposite hip angles)
        front_hip_sym = torch.sum(torch.square(fr_joints[:, 0] + fl_joints[:, 0]), dim=0)
        front_other_sym = torch.sum(torch.square(fr_joints[:, 1:] - fl_joints[:, 1:]), dim=1)
        rear_hip_sym = torch.sum(torch.square(rr_joints[:, 0] + rl_joints[:, 0]), dim=0)
        rear_other_sym = torch.sum(torch.square(rr_joints[:, 1:] - rl_joints[:, 1:]), dim=1)

        total_asymmetry = front_hip_sym + front_other_sym + rear_hip_sym + rear_other_sym

        # Only penalize for symmetric postures (0: normal, 1: handstand)
        mask_symmetric = (posture_cmd == 0) | (posture_cmd == 1)
        penalty = torch.where(mask_symmetric, total_asymmetry, torch.zeros_like(total_asymmetry))

        return penalty
    else:
        return torch.zeros(env.num_envs, device=env.device)


def track_ground_clearance_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    """Reward tracking of ground clearance command using exponential kernel.
    
    This reward encourages the robot to lift its feet to the commanded clearance height
    during the swing phase. It measures the maximum foot lift during swing.
    
    Note: This is a soft reward - exact clearance tracking is difficult, so we reward
    feet that lift above a threshold proportional to the commanded clearance.
    """
    # Get commanded clearance (shape: num_envs,)
    target_clearance = env.command_manager.get_command(command_name)[:, 0]
    
    # Get contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Check which feet are in air (not in contact)
    contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # z-component
    feet_in_air = contact_forces < 1.0  # Threshold for contact detection
    
    # Get foot positions in world frame
    asset: RigidObject = env.scene[asset_cfg.name]
    # We use the asset's foot body positions
    foot_pos_w = asset.data.body_pos_w[:, sensor_cfg.body_ids, :]
    
    # Get terrain height (assume flat terrain at z=0)
    terrain_height = 0.0
    
    # Compute foot heights above ground
    foot_heights = foot_pos_w[:, :, 2] - terrain_height  # (num_envs, num_feet)
    
    # Only consider feet in swing phase (in air)
    # Reward if any foot achieves good clearance
    max_clearance = torch.max(foot_heights * feet_in_air.float(), dim=1)[0]
    
    # Compute error
    clearance_error = torch.square(max_clearance - target_clearance)
    
    # Exponential reward
    reward = torch.exp(-clearance_error / std**2)
    
    return reward


def track_gait_command_categorical(
    env: ManagerBasedRLEnv,
    command_name: str,
    weight_match: float = 1.0,
) -> torch.Tensor:
    """Reward for matching the commanded gait pattern.
    
    This is a placeholder reward - the actual gait tracking happens through
    the CPG dynamics in the action term. This reward can be used to verify
    the robot is following the intended gait pattern.
    """
    # Get commanded gait ID
    gait_cmd = env.command_manager.get_command(command_name)[:, 0]
    
    # Since gait is already enforced by CPG, just return a constant reward
    # when a valid gait command is given
    reward = torch.ones(env.num_envs, device=env.device) * weight_match
    
    return reward



# ── Foot Impact Penalties (ETH, Arm et al. 2025 style) ──

def foot_impact_velocity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot velocity at the moment of ground contact.

    Only activates when a foot JUST made contact (transition from air to ground).
    Encourages soft landings and smooth wheel-ground interaction.

    r = sum_feet( ||v_foot||^2 * just_contacted )
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    # Contact just started: current contact AND previous no contact
    # contact_sensor.data.net_forces_w_history: (N, history, num_bodies, 3)
    current_contact = torch.norm(contact_sensor.data.net_forces_w_history[:, 0, :, :], dim=-1) > 1.0  # (N, num_bodies)
    if contact_sensor.data.net_forces_w_history.shape[1] > 1:
        prev_contact = torch.norm(contact_sensor.data.net_forces_w_history[:, 1, :, :], dim=-1) > 1.0
    else:
        prev_contact = torch.zeros_like(current_contact)

    # Just contacted = current AND NOT previous
    just_contacted = current_contact & ~prev_contact  # (N, num_bodies)

    # Get foot body velocities
    # body_vel_w shape: (N, num_bodies, 6) — [lin_vel(3), ang_vel(3)]
    foot_vel = asset.data.body_lin_vel_w  # (N, num_bodies, 3)

    # Velocity magnitude squared for each body
    vel_sq = torch.sum(torch.square(foot_vel), dim=-1)  # (N, num_bodies)

    # Only penalize at contact moment
    impact = vel_sq * just_contacted.float()  # (N, num_bodies)

    # Sum across all bodies (contact sensor tracks the configured bodies)
    return torch.sum(impact, dim=1)  # (N,)


def contact_force_threshold(
    env: ManagerBasedRLEnv,
    threshold: float = 100.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Penalize contact forces exceeding a threshold.

    r = sum_bodies( max(||F|| - threshold, 0) )

    Different from the existing contact_forces reward which penalizes ALL forces.
    This only penalizes forces ABOVE the threshold — normal walking forces are free.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]  # (N, num_bodies, 3)
    force_mag = torch.norm(forces, dim=-1)  # (N, num_bodies)
    excess = torch.clamp(force_mag - threshold, min=0.0)
    return torch.sum(excess, dim=1)  # (N,)



# ── Base Height with Tolerance (B2W Eq.18) ──

def base_height_tolerance(
    env: ManagerBasedRLEnv,
    target_height: float = 0.426,
    tolerance: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base height deviation beyond tolerance (B2W Eq.18).

    r_h = max(0, |h_base - target| - tolerance)
    No penalty within [target-tol, target+tol] = [0.376, 0.476]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    deviation = torch.abs(h - target_height) - tolerance
    return torch.clamp(deviation, min=0.0)



# ── B2W-style Rewards (Bjelonic et al.) ──

def track_lin_vel_direction(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """B2W Eq.14: Linear velocity tracking with direction reward.

    if |v_des| < 0.05:
        r = 2.0 * exp(-2.0 * ||v_xy_body||^2)     # stop command: penalize any motion
    else:
        r = exp(-2.0 * ||v_xy - v_des||^2) + v_des · v_xy  # track + direction dot product
    """
    asset = env.scene[asset_cfg.name]
    v_xy = asset.data.root_lin_vel_b[:, :2]  # body frame xy velocity
    v_des = env.command_manager.get_command(command_name)[:, :2]  # desired xy velocity

    v_des_norm = torch.norm(v_des, dim=1)
    is_stop = v_des_norm < 0.05

    # Stop: reward stillness
    r_stop = 2.0 * torch.exp(-2.0 * torch.sum(v_xy ** 2, dim=1))

    # Move: exp tracking + direction dot product
    error = torch.sum((v_xy - v_des) ** 2, dim=1)
    dot = torch.sum(v_des * v_xy, dim=1)  # direction alignment
    r_move = torch.exp(-2.0 * error) + dot

    return torch.where(is_stop, r_stop, r_move)


def track_ang_vel_yaw(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """B2W Eq.15: Angular velocity tracking.

    r_av = exp(-2.0 * (omega_z_body - omega_des)^2)
    """
    asset = env.scene[asset_cfg.name]
    omega_z = asset.data.root_ang_vel_b[:, 2]
    omega_des = env.command_manager.get_command(command_name)[:, 2]
    return torch.exp(-2.0 * (omega_z - omega_des) ** 2)


def body_motion_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """B2W Eq.16: Penalize non-commanded body motion.

    r_bm = -1.25 * v_z^2 - 0.4 * |omega_x| - 0.4 * |omega_y|

    Penalizes: vertical bouncing (v_z), roll (omega_x), pitch (omega_y).
    Returns positive value — use negative weight.
    """
    asset = env.scene[asset_cfg.name]
    v_z = asset.data.root_lin_vel_b[:, 2]
    omega_x = asset.data.root_ang_vel_b[:, 0]
    omega_y = asset.data.root_ang_vel_b[:, 1]
    return 1.25 * v_z ** 2 + 0.4 * torch.abs(omega_x) + 0.4 * torch.abs(omega_y)


def body_tilt_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """B2W Eq.17: Body orientation penalty via rotation matrix.

    r_ori = arccos(R_b(3,3))^2

    R_b(3,3) = cos(tilt_angle). Upright → 0, tilted → large.
    More robust than Euler angles (no gimbal lock).
    Returns positive value — use negative weight.
    """
    asset = env.scene[asset_cfg.name]
    # projected_gravity_b = R_b^T @ [0,0,-1]
    # R_b(3,3) = cos(tilt) = -projected_gravity_b[z]
    gz = -asset.data.projected_gravity_b[:, 2]  # = R_b(3,3)
    gz = torch.clamp(gz, -1.0, 1.0)  # numerical safety for arccos
    tilt = torch.arccos(gz)
    return tilt ** 2
