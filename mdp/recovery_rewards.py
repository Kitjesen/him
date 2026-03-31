# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Recovery reward functions based on 'Learning to Recover' (arXiv:2506.05516).

Key mechanism: Episode-based Dynamic Reward Shaping (ED)
- Episode start: ed ≈ 0 → task rewards suppressed → free exploration of recovery strategies
- Episode end: ed → 1 → full task reward → precise standing convergence
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def compute_ed(step: int, total_steps: int, k: int = 3) -> float:
    """Episode-based Dynamic factor (Eq.1 in paper).

    Args:
        step: Current step in the episode.
        total_steps: Total steps per episode (T).
        k: Growth rate exponent. Paper uses k=3.

    Returns:
        ED factor in [0, 1]. Near 0 at start, near 1 at end.
    """
    T = total_steps
    a = T / 2.0
    ed = (a * step / T) ** k
    return ed


def compute_cw(progress: float, beta: float = 0.3, decay: float = 0.968) -> float:
    """Curriculum Weight for behavior rewards (Eq.3 in paper).

    Args:
        progress: Training progress in [0, 1].
        beta: Initial difficulty factor.
        decay: Decay rate per 10000 iterations.

    Returns:
        CW factor that decays over training.
    """
    cw = beta * (decay ** (progress * 10000))
    return cw


# ── Task Rewards (multiplied by ED) ──

def recovery_stand_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.5,
) -> torch.Tensor:
    """Reward for returning joints to default standing position (Table I, scale=42).

    exp(-sum(q_j - q_default)^2 / sigma^2)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    default_pos = asset.data.default_joint_pos
    error = torch.sum(torch.square(joint_pos - default_pos), dim=1)
    return torch.exp(-error / (sigma ** 2))


def recovery_base_height(
    env: ManagerBasedRLEnv,
    target_height: float = 0.55,
    sigma: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for reaching target standing height (Table I, scale=120).

    exp(-max(h_target - h, 0)^2 / sigma^2)
    One-sided: no penalty for being above target.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    height_error = torch.clamp(target_height - base_height, min=0.0)
    return torch.exp(-torch.square(height_error) / (sigma ** 2))


def recovery_base_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for upright orientation (Table I, scale=50).

    (g_body - [0, 0, -1])^2 — penalizes deviation from upright.
    Returns NEGATIVE value (this is a penalty).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_b = asset.data.projected_gravity_b  # (N, 3)
    # Ideal upright: gravity in body frame = [0, 0, -1]
    ideal = torch.tensor([0.0, 0.0, -1.0], device=gravity_b.device)
    error = torch.sum(torch.square(gravity_b - ideal), dim=1)
    return error


# ── Behavior Rewards (some multiplied by CW) ──

def recovery_body_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Penalty for body link collisions (Table I, scale=-5e-2).

    Sum of squared contact forces on thigh/shank/base links.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    # Sum squared forces across all tracked body links
    return torch.sum(torch.square(net_forces[:, :, :2].sum(dim=1)), dim=1)


def recovery_action_rate(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalty for action changes between steps (Table I, scale=-1e-2).

    Only penalizes leg actions (indices 0:12), wheels excluded.
    """
    # action_manager stores last two actions
    if hasattr(env, 'action_manager'):
        current = env.action_manager.action[:, :12]
        prev = env.action_manager.prev_action[:, :12]
        return torch.sum(torch.square(current - prev), dim=1)
    return torch.zeros(env.num_envs, device=env.device)


def recovery_joint_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for joint velocities (Table I, scale=-2e-2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def recovery_torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for joint torques (Table I, scale=-2.5e-5)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)


def recovery_joint_acceleration(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for joint accelerations (Table I, scale=-2.5e-7)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc), dim=1)


def recovery_wheel_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_joint_names: list = None,
) -> torch.Tensor:
    """Penalty for excessive wheel spinning (Table I, scale=-2e-2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if wheel_joint_names and asset_cfg.joint_ids is not None:
        wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    else:
        # Default: last 4 joints are wheels
        wheel_vel = asset.data.joint_vel[:, -4:]
    return torch.sum(torch.square(wheel_vel), dim=1)
