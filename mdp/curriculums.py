# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    range_multiplier: tuple[float, float] = (0.1, 1.0),
    performance_threshold: float = 0.8,
    delta_step: float = 0.1,
) -> torch.Tensor:
    """速度命令课程学习：根据跟踪性能自动调整命令速度范围

    Args:
        env: 环境实例
        env_ids: 环境ID序列
        reward_term_name: 用于评估性能的奖励项名称
        range_multiplier: 速度范围倍数 (开始倍数, 结束倍数)
        performance_threshold: 性能阈值，达到此值时增加命令范围
        delta_step: 每次调整的步长

    Returns:
        当前最大线性速度的tensor
    """
    # 获取命令管理器和基础速度配置
    base_velocity_cmd = env.command_manager.get_term("base_velocity")
    base_velocity_ranges = base_velocity_cmd.cfg.ranges

    # 初始化课程学习参数（仅在需要时）
    if not hasattr(env, "_vel_curriculum_initialized"):
        env._vel_curriculum_initialized = True
        # 存储原始速度范围
        env._original_vel_ranges = {
            "lin_vel_x": list(base_velocity_ranges.lin_vel_x),
            "lin_vel_y": list(base_velocity_ranges.lin_vel_y),
        }
        # 计算初始和最终范围
        original_x = torch.tensor(env._original_vel_ranges["lin_vel_x"], device=env.device)
        original_y = torch.tensor(env._original_vel_ranges["lin_vel_y"], device=env.device)

        env._initial_vel_x = original_x * range_multiplier[0]
        env._final_vel_x = original_x * range_multiplier[1]
        env._initial_vel_y = original_y * range_multiplier[0]
        env._final_vel_y = original_y * range_multiplier[1]

        # 设置初始命令范围
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # 每个episode结束时检查是否需要调整命令范围（当有环境重置时）
    if len(env_ids) > 0:
        try:
            episode_sums = env.reward_manager._episode_sums[reward_term_name]
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            # 计算归一化性能
            max_possible_reward = reward_term_cfg.weight * env.max_episode_length_s
            if abs(max_possible_reward) > 1e-6:
                mean_performance = torch.mean(episode_sums[env_ids]) / max_possible_reward

                # 如果性能超过阈值，增加命令范围
                if mean_performance > performance_threshold:
                    current_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
                    current_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)

                    # 计算新的速度范围
                    delta = torch.tensor([-delta_step, delta_step], device=env.device)
                    new_vel_x = current_vel_x + delta
                    new_vel_y = current_vel_y + delta

                    # 限制在最终范围内
                    new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
                    new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

                    # 更新配置
                    base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
                    base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

        except (KeyError, AttributeError) as e:
            # 记录警告但不中断执行
            pass

    # 返回当前最大线性速度
    current_max_vel = torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)
    return current_max_vel


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reward_term_name: str = "track_lin_vel_xy_exp",
    performance_threshold: float = 0.75,
    distance_threshold_factor: float = 0.5,
    max_terrain_level: int = 5,
) -> torch.Tensor:
    """地形课程学习：根据机器人运动距离和性能自动调整地形难度

    Args:
        env: 环境实例
        env_ids: 环境ID序列
        asset_cfg: 机器人资产配置
        reward_term_name: 用于评估性能的奖励项名称
        performance_threshold: 性能阈值
        distance_threshold_factor: 距离阈值因子
        max_terrain_level: 最大地形级别

    Returns:
        平均地形级别的tensor
    """
    # 检查地形是否存在
    if not hasattr(env.scene, "terrain") or env.scene.terrain is None:
        return torch.zeros(1, device=env.device)

    terrain: TerrainImporter = env.scene.terrain
    if not hasattr(terrain, "terrain_levels") or terrain.terrain_levels is None:
        return torch.zeros(1, device=env.device)

    # 获取机器人资产
    try:
        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command("base_velocity")
    except (KeyError, AttributeError):
        return torch.mean(terrain.terrain_levels.float())

    if len(env_ids) > 0:
        # 计算机器人行走距离
        distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

        # 计算期望距离
        expected_distance = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s

        # 决定地形级别调整
        # 如果走得足够远，升级地形
        move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
        # 如果走得不够远，降级地形
        move_down = distance < expected_distance * distance_threshold_factor
        move_down &= ~move_up  # 确保不同时升级和降级

        # 限制升级：不能超过最大地形级别
        current_levels = terrain.terrain_levels[env_ids]
        move_up &= current_levels < max_terrain_level

        # 更新地形级别
        terrain.update_env_origins(env_ids, move_up, move_down)

    # 返回平均地形级别
    return torch.mean(terrain.terrain_levels.float())


def disturbance_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    force_range_start: tuple[float, float] = (5.0, 10.0),
    force_range_end: tuple[float, float] = (20.0, 50.0),
    performance_threshold: float = 0.8,
    curriculum_step: float = 0.1,
) -> torch.Tensor:
    """外力干扰课程学习：根据性能逐渐增加外力扰动强度

    Args:
        env: 环境实例
        env_ids: 环境ID序列
        reward_term_name: 用于评估性能的奖励项名称
        force_range_start: 初始外力范围 (N)
        force_range_end: 最终外力范围 (N)
        performance_threshold: 性能阈值
        curriculum_step: 课程学习步长

    Returns:
        当前扰动级别的tensor
    """
    # 初始化扰动课程学习状态
    if not hasattr(env, "_disturbance_curriculum_initialized"):
        env._disturbance_curriculum_initialized = True
        env._disturbance_level = 0.0  # 课程学习级别 (0.0-1.0)
        env._disturbance_episode_count = 0
        env._disturbance_performance_sum = 0.0

    if len(env_ids) > 0:
        try:
            episode_sums = env.reward_manager._episode_sums[reward_term_name]
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            # 计算平均性能
            max_possible_reward = reward_term_cfg.weight * env.max_episode_length_s
            if abs(max_possible_reward) > 1e-6:
                current_performance = torch.mean(episode_sums[env_ids]) / max_possible_reward

                # 更新性能统计
                env._disturbance_performance_sum += float(current_performance)
                env._disturbance_episode_count += 1

                # 每100个episode检查一次是否需要增加难度
                if env._disturbance_episode_count % 100 == 0:
                    avg_performance = env._disturbance_performance_sum / env._disturbance_episode_count

                    if avg_performance > performance_threshold and env._disturbance_level < 1.0:
                        env._disturbance_level = min(1.0, env._disturbance_level + curriculum_step)

                        # 重置统计
                        env._disturbance_performance_sum = 0.0
                        env._disturbance_episode_count = 0

                        # 更新外力范围
                        level = env._disturbance_level
                        new_force_min = force_range_start[0] + level * (force_range_end[0] - force_range_start[0])
                        new_force_max = force_range_start[1] + level * (force_range_end[1] - force_range_start[1])

                        # 更新事件管理器的外力参数
                        try:
                            if hasattr(env, "event_manager"):
                                # 查找外力相关的事件项
                                for term_name, term in env.event_manager._terms.items():
                                    if "force" in term_name.lower() and hasattr(term, "cfg"):
                                        if hasattr(term.cfg, "params") and "force_range" in term.cfg.params:
                                            term.cfg.params["force_range"] = (new_force_min, new_force_max)
                        except (AttributeError, KeyError):
                            pass

        except (KeyError, AttributeError):
            pass

    # 返回当前扰动级别
    return torch.tensor(getattr(env, "_disturbance_level", 0.0), device=env.device)


def mass_randomization_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    mass_range_start: tuple[float, float] = (0.9, 1.1),
    mass_range_end: tuple[float, float] = (0.7, 1.3),
    performance_threshold: float = 0.8,
    curriculum_step: float = 0.1,
) -> torch.Tensor:
    """质量随机化课程学习：根据性能逐渐增加质量变化范围

    Args:
        env: 环境实例
        env_ids: 环境ID序列
        reward_term_name: 用于评估性能的奖励项名称
        mass_range_start: 初始质量变化范围（倍数）
        mass_range_end: 最终质量变化范围（倍数）
        performance_threshold: 性能阈值
        curriculum_step: 课程学习步长

    Returns:
        当前质量随机化级别的tensor
    """
    # 初始化质量课程学习状态
    if not hasattr(env, "_mass_curriculum_initialized"):
        env._mass_curriculum_initialized = True
        env._mass_curriculum_level = 0.0  # 课程学习级别 (0.0-1.0)
        env._mass_episode_count = 0
        env._mass_performance_sum = 0.0

    if len(env_ids) > 0:
        try:
            episode_sums = env.reward_manager._episode_sums[reward_term_name]
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            # 计算平均性能
            max_possible_reward = reward_term_cfg.weight * env.max_episode_length_s
            if abs(max_possible_reward) > 1e-6:
                current_performance = torch.mean(episode_sums[env_ids]) / max_possible_reward

                # 更新性能统计
                env._mass_performance_sum += float(current_performance)
                env._mass_episode_count += 1

                # 每50个episode检查一次是否需要增加难度
                if env._mass_episode_count % 50 == 0:
                    avg_performance = env._mass_performance_sum / env._mass_episode_count

                    if avg_performance > performance_threshold and env._mass_curriculum_level < 1.0:
                        env._mass_curriculum_level = min(1.0, env._mass_curriculum_level + curriculum_step)

                        # 重置统计
                        env._mass_performance_sum = 0.0
                        env._mass_episode_count = 0

                        # 更新质量范围
                        level = env._mass_curriculum_level
                        new_mass_min = mass_range_start[0] + level * (mass_range_end[0] - mass_range_start[0])
                        new_mass_max = mass_range_start[1] + level * (mass_range_end[1] - mass_range_start[1])

                        # 更新事件管理器的质量参数
                        try:
                            if hasattr(env, "event_manager"):
                                for term_name, term in env.event_manager._terms.items():
                                    if "mass" in term_name.lower() and hasattr(term, "cfg"):
                                        if hasattr(term.cfg, "params") and "mass_range" in term.cfg.params:
                                            term.cfg.params["mass_range"] = (new_mass_min, new_mass_max)
                        except (AttributeError, KeyError):
                            pass

        except (KeyError, AttributeError):
            pass

    # 返回当前质量课程级别
    return torch.tensor(getattr(env, "_mass_curriculum_level", 0.0), device=env.device)


def com_randomization_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    com_range_start: dict[str, tuple[float, float]] = None,
    com_range_end: dict[str, tuple[float, float]] = None,
    performance_threshold: float = 0.8,
    curriculum_step: float = 0.1,
) -> torch.Tensor:
    """质心位置随机化课程学习：根据性能逐渐增加质心偏移范围

    Args:
        env: 环境实例
        env_ids: 环境ID序列
        reward_term_name: 用于评估性能的奖励项名称
        com_range_start: 初始质心偏移范围 (m)
        com_range_end: 最终质心偏移范围 (m)
        performance_threshold: 性能阈值
        curriculum_step: 课程学习步长

    Returns:
        当前质心随机化级别的tensor
    """
    if com_range_start is None:
        com_range_start = {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)}
    if com_range_end is None:
        com_range_end = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}

    # 初始化COM课程学习状态
    if not hasattr(env, "_com_curriculum_initialized"):
        env._com_curriculum_initialized = True
        env._com_curriculum_level = 0.0  # 课程学习级别 (0.0-1.0)
        env._com_episode_count = 0
        env._com_performance_sum = 0.0

    if len(env_ids) > 0:
        try:
            episode_sums = env.reward_manager._episode_sums[reward_term_name]
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            # 计算平均性能
            max_possible_reward = reward_term_cfg.weight * env.max_episode_length_s
            if abs(max_possible_reward) > 1e-6:
                current_performance = torch.mean(episode_sums[env_ids]) / max_possible_reward

                # 更新性能统计
                env._com_performance_sum += float(current_performance)
                env._com_episode_count += 1

                # 每50个episode检查一次是否需要增加难度
                if env._com_episode_count % 50 == 0:
                    avg_performance = env._com_performance_sum / env._com_episode_count

                    if avg_performance > performance_threshold and env._com_curriculum_level < 1.0:
                        env._com_curriculum_level = min(1.0, env._com_curriculum_level + curriculum_step)

                        # 重置统计
                        env._com_performance_sum = 0.0
                        env._com_episode_count = 0

                        # 更新质心范围
                        level = env._com_curriculum_level
                        new_com_ranges = {}
                        for axis in ["x", "y", "z"]:
                            start_range = com_range_start[axis]
                            end_range = com_range_end[axis]
                            new_min = start_range[0] + level * (end_range[0] - start_range[0])
                            new_max = start_range[1] + level * (end_range[1] - start_range[1])
                            new_com_ranges[axis] = (new_min, new_max)

                        # 更新事件管理器的质心参数
                        try:
                            if hasattr(env, "event_manager"):
                                for term_name, term in env.event_manager._terms.items():
                                    if "com" in term_name.lower() and hasattr(term, "cfg"):
                                        if hasattr(term.cfg, "params"):
                                            for axis, range_val in new_com_ranges.items():
                                                param_name = f"com_{axis}_range"
                                                if param_name in term.cfg.params:
                                                    term.cfg.params[param_name] = range_val
                        except (AttributeError, KeyError):
                            pass

        except (KeyError, AttributeError):
            pass

    # 返回当前质心课程级别
    return torch.tensor(getattr(env, "_com_curriculum_level", 0.0), device=env.device)
