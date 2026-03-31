# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the center of mass (COM) positions for the rigid bodies.

    This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
    randomized by adding, scaling, or setting random values sampled from the specified distribution.

    .. tip::
        This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor | None): Specific environment indices to apply randomization, or None for all environments.
        asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
        com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
        operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
        distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
    """
    # Extract the asset (Articulation or RigidObject)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Resolve environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current COM offsets (num_assets, num_bodies, 3)
    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):  # Randomize x, y, z independently
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # Set the randomized COM offsets into the simulation
    asset.root_physx_view.set_coms(com_offsets, env_ids)


def randomize_terrain_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float] = (0.5, 1.5),
    dynamic_friction_range: tuple[float, float] = (0.5, 1.5),
    restitution_range: tuple[float, float] = (0.0, 0.0),
):
    """随机化地形的物理材料属性（摩擦力和弹性）。

    此函数会为地形设置随机的静摩擦系数、动摩擦系数和弹性恢复系数。
    与 randomize_rigid_body_material 不同，此函数针对地形而非机器人。

    Args:
        env: 仿真环境实例
        env_ids: 需要应用随机化的环境ID，如果为None则应用到所有环境
        static_friction_range: 静摩擦系数范围 (最小值, 最大值)
        dynamic_friction_range: 动摩擦系数范围 (最小值, 最大值)
        restitution_range: 弹性恢复系数范围 (最小值, 最大值)，0表示完全非弹性碰撞
    """
    # 检查场景中是否有地形
    if not hasattr(env.scene, "terrain") or env.scene.terrain is None:
        return

    # 获取地形的prim路径
    terrain_prim_paths = env.scene.terrain.terrain_prim_paths

    if len(terrain_prim_paths) == 0:
        return

    # 使用 IsaacLab 的工具
    import isaacsim.core.utils.prims as prim_utils
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import safe_set_attribute_on_usd_schema

    # 获取USD stage
    stage = get_current_stage()

    # 对每个地形生成随机的材料属性
    for terrain_path in terrain_prim_paths:
        # 采样随机值
        static_friction = math_utils.sample_uniform(
            static_friction_range[0], static_friction_range[1], (1,), device="cpu"
        ).item()
        dynamic_friction = math_utils.sample_uniform(
            dynamic_friction_range[0], dynamic_friction_range[1], (1,), device="cpu"
        ).item()
        restitution = math_utils.sample_uniform(restitution_range[0], restitution_range[1], (1,), device="cpu").item()

        # 查找或创建材料prim路径
        material_path = f"{terrain_path}/physicsMaterial"

        # 如果材料不存在，创建它
        if not prim_utils.is_prim_path_valid(material_path):
            _ = UsdShade.Material.Define(stage, material_path)

        # 获取材料prim
        material_prim = prim_utils.get_prim_at_path(material_path)

        if material_prim and material_prim.IsValid():
            # 应用 UsdPhysics.MaterialAPI
            usd_physics_material_api = UsdPhysics.MaterialAPI(material_prim)
            if not usd_physics_material_api:
                usd_physics_material_api = UsdPhysics.MaterialAPI.Apply(material_prim)

            # 应用 PhysxSchema.PhysxMaterialAPI
            physx_material_api = PhysxSchema.PhysxMaterialAPI(material_prim)
            if not physx_material_api:
                physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)

            # 使用 safe_set_attribute_on_usd_schema 设置摩擦和弹性属性
            safe_set_attribute_on_usd_schema(
                usd_physics_material_api, "static_friction", static_friction, camel_case=True
            )
            safe_set_attribute_on_usd_schema(
                usd_physics_material_api, "dynamic_friction", dynamic_friction, camel_case=True
            )
            safe_set_attribute_on_usd_schema(usd_physics_material_api, "restitution", restitution, camel_case=True)

            # 将材料绑定到地形
            terrain_prim = prim_utils.get_prim_at_path(terrain_path)
            if terrain_prim and terrain_prim.IsValid():
                # 应用材料绑定
                material_binding_api = UsdShade.MaterialBindingAPI.Apply(terrain_prim)
                material_binding_api.Bind(UsdShade.Material(material_prim))


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def randomize_hcpg_ground_penetration(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    gp_range: tuple[float, float],
    action_name: str = "joint_pos",
):
    """Randomize the ground penetration parameter of the H-CPG action term.
    
    This randomizes the g_p (ground penetration depth) parameter used in the
    PF Layer's stance phase trajectory. During stance, foot z-position is:
        z = -h_cmd + g_p * sin(theta)
    
    Larger g_p values push feet slightly into the ground during stance,
    which can help with ground contact stability.
    
    Args:
        env: The environment object.
        env_ids: The environment IDs to randomize. If None, all environments.
        gp_range: (min, max) range for ground penetration in meters.
                 Seto 2025 recommends [0.0, 0.02] (0 to 2cm).
        action_name: Name of the HCPG action term (default: "joint_pos").
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    # Get the HCPG action term
    action_term = env.action_manager.get_term(action_name)
    
    # Sample random ground penetration values
    gp_min, gp_max = gp_range
    new_gp = torch.rand(len(env_ids), device=env.device) * (gp_max - gp_min) + gp_min
    
    # Update the action term's ground penetration parameter
    # Note: The action term stores g_p as a scalar (same for all envs)
    # We'll update it per-env by storing a tensor
    if not hasattr(action_term, '_g_p_per_env'):
        # First time: initialize per-env ground penetration tensor
        action_term._g_p_per_env = torch.ones(env.num_envs, device=env.device) * action_term._g_p
    
    action_term._g_p_per_env[env_ids] = new_gp


# ── Custom DR wrappers (function-based, compatible with EventManager) ──

def randomize_mass_simple(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mass_range: tuple = (-1.0, 3.0),
    operation: str = "add",
):
    """Simple mass randomization — function-based wrapper."""
    import torch
    asset = env.scene[asset_cfg.name]
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    masses = asset.root_physx_view.get_masses()
    masses[env_ids[:, None], body_ids] = asset.data.default_mass[env_ids[:, None], body_ids].clone()
    if operation == "add":
        masses[env_ids[:, None], body_ids] += torch.empty_like(
            masses[env_ids[:, None], body_ids]
        ).uniform_(mass_range[0], mass_range[1])
    elif operation == "scale":
        masses[env_ids[:, None], body_ids] *= torch.empty_like(
            masses[env_ids[:, None], body_ids]
        ).uniform_(mass_range[0], mass_range[1])
    asset.root_physx_view.set_masses(masses, env_ids)


def randomize_actuator_simple(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stiffness_range: tuple = (0.8, 1.2),
    damping_range: tuple = (0.8, 1.2),
):
    """Simple actuator gain randomization — function-based wrapper."""
    import torch
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    for actuator in asset.actuators.values():
        # Get this actuator's joint indices
        joint_idx = actuator.joint_indices
        if isinstance(joint_idx, slice):
            n_joints = actuator.stiffness.shape[1]
            stiffness = asset.data.default_joint_stiffness[env_ids, :n_joints].clone()
            stiffness *= torch.empty_like(stiffness).uniform_(stiffness_range[0], stiffness_range[1])
            actuator.stiffness[env_ids] = stiffness
            damping = asset.data.default_joint_damping[env_ids, :n_joints].clone()
            damping *= torch.empty_like(damping).uniform_(damping_range[0], damping_range[1])
            actuator.damping[env_ids] = damping
        else:
            stiffness = asset.data.default_joint_stiffness[env_ids][:, joint_idx].clone()
            stiffness *= torch.empty_like(stiffness).uniform_(stiffness_range[0], stiffness_range[1])
            actuator.stiffness[env_ids] = stiffness
            damping = asset.data.default_joint_damping[env_ids][:, joint_idx].clone()
            damping *= torch.empty_like(damping).uniform_(damping_range[0], damping_range[1])
            actuator.damping[env_ids] = damping
