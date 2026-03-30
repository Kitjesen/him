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
    """Velocity command curriculum: automatically adjusts command velocity range based on tracking performance.

    Args:
        env: Environment instance
        env_ids: Sequence of environment IDs
        reward_term_name: Name of the reward term used to evaluate performance
        range_multiplier: Velocity range multipliers (start multiplier, end multiplier)
        performance_threshold: Performance threshold; command range expands when this is exceeded
        delta_step: Step size for each adjustment

    Returns:
        Tensor of the current maximum linear velocity
    """
    # Get command manager and base velocity configuration
    base_velocity_cmd = env.command_manager.get_term("base_velocity")
    base_velocity_ranges = base_velocity_cmd.cfg.ranges

    # Initialize curriculum parameters (only when needed)
    if not hasattr(env, "_vel_curriculum_initialized"):
        env._vel_curriculum_initialized = True
        # Store original velocity ranges
        env._original_vel_ranges = {
            "lin_vel_x": list(base_velocity_ranges.lin_vel_x),
            "lin_vel_y": list(base_velocity_ranges.lin_vel_y),
        }
        # Compute initial and final ranges
        original_x = torch.tensor(env._original_vel_ranges["lin_vel_x"], device=env.device)
        original_y = torch.tensor(env._original_vel_ranges["lin_vel_y"], device=env.device)

        env._initial_vel_x = original_x * range_multiplier[0]
        env._final_vel_x = original_x * range_multiplier[1]
        env._initial_vel_y = original_y * range_multiplier[0]
        env._final_vel_y = original_y * range_multiplier[1]

        # Set initial command range
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # At each episode end, check whether command range needs adjustment (when environments reset)
    if len(env_ids) > 0:
        try:
            episode_sums = env.reward_manager._episode_sums[reward_term_name]
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            # Compute normalized performance
            max_possible_reward = reward_term_cfg.weight * env.max_episode_length_s
            if abs(max_possible_reward) > 1e-6:
                mean_performance = torch.mean(episode_sums[env_ids]) / max_possible_reward

                # If performance exceeds threshold, expand command range
                if mean_performance > performance_threshold:
                    current_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
                    current_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)

                    # Compute new velocity range
                    delta = torch.tensor([-delta_step, delta_step], device=env.device)
                    new_vel_x = current_vel_x + delta
                    new_vel_y = current_vel_y + delta

                    # Clamp to final range
                    new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
                    new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

                    # Update configuration
                    base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
                    base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

        except (KeyError, AttributeError) as e:
            # Log warning but do not interrupt execution
            pass

    # Return current maximum linear velocity
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
    """Terrain curriculum: automatically adjusts terrain difficulty based on robot travel distance and performance.

    Args:
        env: Environment instance
        env_ids: Sequence of environment IDs
        asset_cfg: Robot asset configuration
        reward_term_name: Name of the reward term used to evaluate performance
        performance_threshold: Performance threshold
        distance_threshold_factor: Distance threshold factor
        max_terrain_level: Maximum terrain difficulty level

    Returns:
        Tensor of the mean terrain level
    """
    # Check whether terrain exists
    if not hasattr(env.scene, "terrain") or env.scene.terrain is None:
        return torch.zeros(1, device=env.device)

    terrain: TerrainImporter = env.scene.terrain
    if not hasattr(terrain, "terrain_levels") or terrain.terrain_levels is None:
        return torch.zeros(1, device=env.device)

    # Get robot asset
    try:
        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command("base_velocity")
    except (KeyError, AttributeError):
        return torch.mean(terrain.terrain_levels.float())

    if len(env_ids) > 0:
        # Compute robot travel distance
        distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

        # Compute expected distance
        expected_distance = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s

        # Determine terrain level adjustment
        # If the robot traveled far enough, promote terrain level
        move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
        # If the robot did not travel far enough, demote terrain level
        move_down = distance < expected_distance * distance_threshold_factor
        move_down &= ~move_up  # Ensure promotion and demotion do not occur simultaneously

        # Cap promotion: cannot exceed max terrain level
        current_levels = terrain.terrain_levels[env_ids]
        move_up &= current_levels < max_terrain_level

        # Update terrain levels
        terrain.update_env_origins(env_ids, move_up, move_down)

    # Return mean terrain level
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
    """External disturbance curriculum: gradually increases external force perturbation intensity based on performance.

    Args:
        env: Environment instance
        env_ids: Sequence of environment IDs
        reward_term_name: Name of the reward term used to evaluate performance
        force_range_start: Initial external force range (N)
        force_range_end: Final external force range (N)
        performance_threshold: Performance threshold
        curriculum_step: Curriculum step size

    Returns:
        Tensor of the current disturbance level
    """
    # Initialize disturbance curriculum state
    if not hasattr(env, "_disturbance_curriculum_initialized"):
        env._disturbance_curriculum_initialized = True
        env._disturbance_level = 0.0  # Curriculum level (0.0-1.0)
        env._disturbance_episode_count = 0
        env._disturbance_performance_sum = 0.0

    if len(env_ids) > 0:
        try:
            episode_sums = env.reward_manager._episode_sums[reward_term_name]
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            # Compute mean performance
            max_possible_reward = reward_term_cfg.weight * env.max_episode_length_s
            if abs(max_possible_reward) > 1e-6:
                current_performance = torch.mean(episode_sums[env_ids]) / max_possible_reward

                # Update performance statistics
                env._disturbance_performance_sum += float(current_performance)
                env._disturbance_episode_count += 1

                # Check every 100 episodes whether difficulty should increase
                if env._disturbance_episode_count % 100 == 0:
                    avg_performance = env._disturbance_performance_sum / env._disturbance_episode_count

                    if avg_performance > performance_threshold and env._disturbance_level < 1.0:
                        env._disturbance_level = min(1.0, env._disturbance_level + curriculum_step)

                        # Reset statistics
                        env._disturbance_performance_sum = 0.0
                        env._disturbance_episode_count = 0

                        # Update external force range
                        level = env._disturbance_level
                        new_force_min = force_range_start[0] + level * (force_range_end[0] - force_range_start[0])
                        new_force_max = force_range_start[1] + level * (force_range_end[1] - force_range_start[1])

                        # Update event manager external force parameters
                        try:
                            if hasattr(env, "event_manager"):
                                # Find event terms related to external force
                                for term_name, term in env.event_manager._terms.items():
                                    if "force" in term_name.lower() and hasattr(term, "cfg"):
                                        if hasattr(term.cfg, "params") and "force_range" in term.cfg.params:
                                            term.cfg.params["force_range"] = (new_force_min, new_force_max)
                        except (AttributeError, KeyError):
                            pass

        except (KeyError, AttributeError):
            pass

    # Return current disturbance level
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
    """Mass randomization curriculum: gradually increases the mass variation range based on performance.

    Args:
        env: Environment instance
        env_ids: Sequence of environment IDs
        reward_term_name: Name of the reward term used to evaluate performance
        mass_range_start: Initial mass variation range (multiplier)
        mass_range_end: Final mass variation range (multiplier)
        performance_threshold: Performance threshold
        curriculum_step: Curriculum step size

    Returns:
        Tensor of the current mass randomization level
    """
    # Initialize mass curriculum state
    if not hasattr(env, "_mass_curriculum_initialized"):
        env._mass_curriculum_initialized = True
        env._mass_curriculum_level = 0.0  # Curriculum level (0.0-1.0)
        env._mass_episode_count = 0
        env._mass_performance_sum = 0.0

    if len(env_ids) > 0:
        try:
            episode_sums = env.reward_manager._episode_sums[reward_term_name]
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            # Compute mean performance
            max_possible_reward = reward_term_cfg.weight * env.max_episode_length_s
            if abs(max_possible_reward) > 1e-6:
                current_performance = torch.mean(episode_sums[env_ids]) / max_possible_reward

                # Update performance statistics
                env._mass_performance_sum += float(current_performance)
                env._mass_episode_count += 1

                # Check every 50 episodes whether difficulty should increase
                if env._mass_episode_count % 50 == 0:
                    avg_performance = env._mass_performance_sum / env._mass_episode_count

                    if avg_performance > performance_threshold and env._mass_curriculum_level < 1.0:
                        env._mass_curriculum_level = min(1.0, env._mass_curriculum_level + curriculum_step)

                        # Reset statistics
                        env._mass_performance_sum = 0.0
                        env._mass_episode_count = 0

                        # Update mass range
                        level = env._mass_curriculum_level
                        new_mass_min = mass_range_start[0] + level * (mass_range_end[0] - mass_range_start[0])
                        new_mass_max = mass_range_start[1] + level * (mass_range_end[1] - mass_range_start[1])

                        # Update event manager mass parameters
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

    # Return current mass curriculum level
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
    """Center-of-mass randomization curriculum: gradually increases CoM offset range based on performance.

    Args:
        env: Environment instance
        env_ids: Sequence of environment IDs
        reward_term_name: Name of the reward term used to evaluate performance
        com_range_start: Initial CoM offset range (m)
        com_range_end: Final CoM offset range (m)
        performance_threshold: Performance threshold
        curriculum_step: Curriculum step size

    Returns:
        Tensor of the current CoM randomization level
    """
    if com_range_start is None:
        com_range_start = {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)}
    if com_range_end is None:
        com_range_end = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}

    # Initialize CoM curriculum state
    if not hasattr(env, "_com_curriculum_initialized"):
        env._com_curriculum_initialized = True
        env._com_curriculum_level = 0.0  # Curriculum level (0.0-1.0)
        env._com_episode_count = 0
        env._com_performance_sum = 0.0

    if len(env_ids) > 0:
        try:
            episode_sums = env.reward_manager._episode_sums[reward_term_name]
            reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

            # Compute mean performance
            max_possible_reward = reward_term_cfg.weight * env.max_episode_length_s
            if abs(max_possible_reward) > 1e-6:
                current_performance = torch.mean(episode_sums[env_ids]) / max_possible_reward

                # Update performance statistics
                env._com_performance_sum += float(current_performance)
                env._com_episode_count += 1

                # Check every 50 episodes whether difficulty should increase
                if env._com_episode_count % 50 == 0:
                    avg_performance = env._com_performance_sum / env._com_episode_count

                    if avg_performance > performance_threshold and env._com_curriculum_level < 1.0:
                        env._com_curriculum_level = min(1.0, env._com_curriculum_level + curriculum_step)

                        # Reset statistics
                        env._com_performance_sum = 0.0
                        env._com_episode_count = 0

                        # Update CoM range
                        level = env._com_curriculum_level
                        new_com_ranges = {}
                        for axis in ["x", "y", "z"]:
                            start_range = com_range_start[axis]
                            end_range = com_range_end[axis]
                            new_min = start_range[0] + level * (end_range[0] - start_range[0])
                            new_max = start_range[1] + level * (end_range[1] - start_range[1])
                            new_com_ranges[axis] = (new_min, new_max)

                        # Update event manager CoM parameters
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

    # Return current CoM curriculum level
    return torch.tensor(getattr(env, "_com_curriculum_level", 0.0), device=env.device)
