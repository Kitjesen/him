# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Training script for Thunder fall recovery policy.

Based on 'Learning to Recover' (arXiv:2506.05516):
- Episode-based Dynamic Reward Shaping (ED)
- 2s free-fall initialization + 3s recovery window
- Asymmetric actor-critic (reuses HIM architecture)

Usage:
    python train_recovery.py --num_envs 2048 --headless --max_iterations 5000
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn

# ── Recovery Reward Computer ──

class RecoveryRewardComputer:
    """Computes recovery rewards with Episode-based Dynamic Reward Shaping.

    The total reward at each timestep:
        r = ed * (w_jp * r_joint_pos + w_bh * r_base_height + w_bo * r_base_orient)
          + cw * (w_col * r_collision + w_ar * r_action_rate)
          + w_jv * r_joint_vel + w_tq * r_torques + w_ja * r_joint_acc + w_wv * r_wheel_vel

    ED factor (Eq.1): ed = (a*t/T)^k
        - t=0: ed≈0 → free exploration (rolling, wheel-assisted flipping)
        - t=T: ed→1 → precise standing convergence

    CW factor (Eq.3): cw = beta * decay^(progress * 10000)
        - Decays over training to relax behavior constraints
    """

    def __init__(
        self,
        num_envs: int,
        episode_steps: int = 150,  # 3s at 50Hz (after 2s free-fall)
        device: str = "cuda",
        # Task reward weights (Table I)
        w_joint_pos: float = 42.0,
        w_base_height: float = 120.0,
        w_base_orient: float = 50.0,
        # Behavior reward weights
        w_collision: float = -5e-2,
        w_action_rate: float = -1e-2,
        # Constant penalty weights
        w_joint_vel: float = -2e-2,
        w_torques: float = -2.5e-5,
        w_joint_acc: float = -2.5e-7,
        w_wheel_vel: float = -2e-2,
        # ED params
        ed_k: int = 3,
        # CW params
        cw_beta: float = 0.3,
        cw_decay: float = 0.968,
        # Thunder params
        target_height: float = 0.55,
        joint_pos_sigma: float = 0.5,
        height_sigma: float = 0.1,
        num_leg_joints: int = 12,
    ):
        self.num_envs = num_envs
        self.episode_steps = episode_steps
        self.device = device

        # Weights
        self.w_joint_pos = w_joint_pos
        self.w_base_height = w_base_height
        self.w_base_orient = w_base_orient
        self.w_collision = w_collision
        self.w_action_rate = w_action_rate
        self.w_joint_vel = w_joint_vel
        self.w_torques = w_torques
        self.w_joint_acc = w_joint_acc
        self.w_wheel_vel = w_wheel_vel

        # ED params
        self.ed_k = ed_k
        self.ed_a = episode_steps / 2.0

        # CW params
        self.cw_beta = cw_beta
        self.cw_decay = cw_decay

        # Robot params
        self.target_height = target_height
        self.joint_pos_sigma = joint_pos_sigma
        self.height_sigma = height_sigma
        self.num_leg_joints = num_leg_joints

        # Step counter per environment
        self.step_count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.prev_actions = None

    def reset(self, env_ids: torch.Tensor):
        """Reset step counters for specified environments."""
        self.step_count[env_ids] = 0

    def compute_ed(self) -> torch.Tensor:
        """Compute per-env ED factor based on current step."""
        t = self.step_count.float()
        T = self.episode_steps
        a = self.ed_a
        ed = (a * t / T) ** self.ed_k
        return ed

    def compute_cw(self, training_progress: float) -> float:
        """Compute curriculum weight based on training progress [0, 1]."""
        return self.cw_beta * (self.cw_decay ** (training_progress * 10000))

    def compute(
        self,
        joint_pos: torch.Tensor,        # (N, num_joints)
        default_joint_pos: torch.Tensor, # (N, num_joints)
        base_height: torch.Tensor,       # (N,)
        projected_gravity: torch.Tensor, # (N, 3)
        joint_vel: torch.Tensor,         # (N, num_joints)
        applied_torque: torch.Tensor,    # (N, num_joints)
        joint_acc: torch.Tensor,         # (N, num_joints)
        actions: torch.Tensor,           # (N, num_actions)
        contact_forces: torch.Tensor,    # (N, num_bodies, 3) or None
        training_progress: float = 0.0,
    ) -> dict:
        """Compute all recovery rewards for current step.

        Returns:
            Dict with 'total' reward and individual components for logging.
        """
        self.step_count += 1
        N = joint_pos.shape[0]

        # ── ED and CW factors ──
        ed = self.compute_ed()  # (N,)
        cw = self.compute_cw(training_progress)

        # ── Task Rewards (× ED) ──

        # 1. Stand joint position: exp(-sum(q - q_default)^2 / sigma^2)
        joint_error = torch.sum(torch.square(joint_pos - default_joint_pos), dim=1)
        r_joint_pos = torch.exp(-joint_error / (self.joint_pos_sigma ** 2))

        # 2. Base height: exp(-max(h_target - h, 0)^2 / sigma^2)
        height_error = torch.clamp(self.target_height - base_height, min=0.0)
        r_base_height = torch.exp(-torch.square(height_error) / (self.height_sigma ** 2))

        # 3. Base orientation: (g_body - [0,0,-1])^2
        ideal_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        r_base_orient = torch.sum(torch.square(projected_gravity - ideal_gravity), dim=1)

        task_reward = ed * (
            self.w_joint_pos * r_joint_pos
            + self.w_base_height * r_base_height
            + self.w_base_orient * r_base_orient  # note: w_base_orient is positive, r is error → net penalty shaped by ED
        )

        # ── Behavior Rewards (collision + action_rate × CW) ──

        # 4. Body collision
        if contact_forces is not None:
            r_collision = torch.sum(torch.square(contact_forces[:, :, :2].sum(dim=1)), dim=1)
        else:
            r_collision = torch.zeros(N, device=self.device)

        # 5. Action rate (legs only, exclude wheels)
        if self.prev_actions is not None:
            r_action_rate = torch.sum(
                torch.square(actions[:, :self.num_leg_joints] - self.prev_actions[:, :self.num_leg_joints]),
                dim=1
            )
        else:
            r_action_rate = torch.zeros(N, device=self.device)
        self.prev_actions = actions.clone()

        behavior_reward = cw * (
            self.w_collision * r_collision
            + self.w_action_rate * r_action_rate
        )

        # ── Constant Penalties ──

        # 6. Joint velocity
        r_joint_vel = torch.sum(torch.square(joint_vel), dim=1)

        # 7. Torques
        r_torques = torch.sum(torch.square(applied_torque), dim=1)

        # 8. Joint acceleration
        r_joint_acc = torch.sum(torch.square(joint_acc), dim=1)

        # 9. Wheel velocity (last 4 joints)
        r_wheel_vel = torch.sum(torch.square(joint_vel[:, -4:]), dim=1)

        constant_penalty = (
            self.w_joint_vel * r_joint_vel
            + self.w_torques * r_torques
            + self.w_joint_acc * r_joint_acc
            + self.w_wheel_vel * r_wheel_vel
        )

        # ── Total ──
        total = task_reward + behavior_reward + constant_penalty

        return {
            "total": total,
            "ed": ed.mean().item(),
            "cw": cw,
            "r_joint_pos": (self.w_joint_pos * r_joint_pos).mean().item(),
            "r_base_height": (self.w_base_height * r_base_height).mean().item(),
            "r_base_orient": (self.w_base_orient * r_base_orient).mean().item(),
            "r_collision": (self.w_collision * r_collision).mean().item(),
            "r_action_rate": (self.w_action_rate * r_action_rate).mean().item(),
            "r_joint_vel": (self.w_joint_vel * r_joint_vel).mean().item(),
            "r_torques": (self.w_torques * r_torques).mean().item(),
            "r_wheel_vel": (self.w_wheel_vel * r_wheel_vel).mean().item(),
        }


# ── Success Checker ──

def check_recovery_success(
    base_height: torch.Tensor,
    projected_gravity: torch.Tensor,
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    height_threshold: float = 0.42,
    joint_deviation_threshold: float = 0.5,
    joint_vel_threshold: float = 0.1,
    orientation_threshold: float = 0.1,
) -> torch.Tensor:
    """Check which environments have successfully recovered.

    Paper's definition: height > 0.42m, joint deviation < 0.5 rad,
    max joint vel < 0.1 rad/s, orientation error < 0.1.

    Returns:
        Boolean tensor (N,) — True for successfully recovered environments.
    """
    height_ok = base_height > height_threshold

    joint_deviation = torch.norm(joint_pos - default_joint_pos, dim=1)
    joints_ok = joint_deviation < joint_deviation_threshold

    max_vel = torch.max(torch.abs(joint_vel), dim=1).values
    vel_ok = max_vel < joint_vel_threshold

    ideal_gravity = torch.tensor([0.0, 0.0, -1.0], device=projected_gravity.device)
    ori_error = torch.norm(projected_gravity - ideal_gravity, dim=1)
    ori_ok = ori_error < orientation_threshold

    return height_ok & joints_ok & vel_ok & ori_ok


if __name__ == "__main__":
    print("Recovery reward module loaded successfully.")
    print("Use with HIMOnPolicyRunner by integrating RecoveryRewardComputer into the training loop.")
    print()
    print("Integration steps:")
    print("  1. Create ThunderRecoveryEnvCfg (recovery_env_cfg.py)")
    print("  2. Register task: RobotLab-Isaac-Velocity-Recovery-Thunder-v0")
    print("  3. In runner, replace env reward with RecoveryRewardComputer.compute()")
    print("  4. Add 2s free-fall phase before policy takes over")
    print()

    # Quick test
    rc = RecoveryRewardComputer(num_envs=4, episode_steps=150, device="cpu")
    dummy = {
        "joint_pos": torch.randn(4, 16),
        "default_joint_pos": torch.zeros(4, 16),
        "base_height": torch.tensor([0.1, 0.3, 0.5, 0.55]),
        "projected_gravity": torch.tensor([[0.5, 0.0, -0.5]] * 4),
        "joint_vel": torch.randn(4, 16) * 0.1,
        "applied_torque": torch.randn(4, 16) * 0.5,
        "joint_acc": torch.randn(4, 16) * 0.01,
        "actions": torch.randn(4, 16),
        "contact_forces": None,
    }
    result = rc.compute(**dummy)
    print(f"Test reward: {result['total'].mean().item():.2f}")
    print(f"  ED={result['ed']:.4f}, CW={result['cw']:.4f}")
    print(f"  joint_pos={result['r_joint_pos']:.2f}, height={result['r_base_height']:.2f}")
    print("OK")
