# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for OW wheeled leg dog robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# # Configuration
# ##
# OW_WHEELED_LEG_DOG_CFG = ArticulationCfg(
#     spawn=sim_utils.UrdfFileCfg(
#         fix_base=False,
#         merge_fixed_joints=True,
#         replace_cylinders_with_capsules=False,
#         asset_path=(f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/ow/wheeled_leg_dogv2/" "urdf/wheeled_leg_dogv2.urdf"),
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
#         ),
#         joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
#             gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.6),
#         joint_pos={
#             ".*_hip_joint": 0.0,
#             "FL_thigh_joint": 0.0,
#             "FR_thigh_joint": 0.0,
#             "RL_thigh_joint": 0.0,
#             "RR_thigh_joint": 0.0,
#             "FL_calf_joint": -1.2,
#             "FR_calf_joint": 1.2,
#             "RL_calf_joint": -1.2,
#             "RR_calf_joint": 1.2,
#             ".*_foot_joint": 0.0,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "legs": DelayedPDActuatorCfg(
#             joint_names_expr=[
#                 ".*_hip_joint",
#                 ".*_thigh_joint",
#                 ".*_calf_joint",
#             ],
#             effort_limit=120.0,
#             stiffness=200.0,
#             damping=5,
#             friction=0.0,
#             min_delay=0,  # 最小延迟：0个物理步骤
#             max_delay=4,  # 最大延迟：4个物理步骤
#         ),
#         "wheel": ImplicitActuatorCfg(
#             joint_names_expr=[".*_foot_joint"],
#             effort_limit_sim=60.0,
#             velocity_limit_sim=16.965,
#             stiffness=0.0,
#             damping=1.0,
#             friction=0.0,
#         ),
#     },
# )
# """
# Configuration of OW wheeled leg dog using DC motor for legs and implicit
# actuator for wheels.
# """


# OW_WHEEL_CFG = ArticulationCfg(
#     spawn=sim_utils.UrdfFileCfg(
#         fix_base=False,
#         merge_fixed_joints=True,
#         replace_cylinders_with_capsules=False,
#         asset_path=(f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/ow/ow_wheel_description/" "urdf/ow_wheel.urdf"),
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
#         ),
#         joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
#             gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.6),
#         joint_pos={
#             "FL_hip_joint": 0.0,
#             "FL_thigh_joint": 0.7,
#             "FL_calf_joint": -1.5,
#             "FL_foot_joint": 0.0,
#             "FR_hip_joint": 0.0,
#             "FR_thigh_joint": -0.7,
#             "FR_calf_joint": 1.5,
#             "FR_foot_joint": 0.0,
#             "RL_hip_joint": 0.0,
#             "RL_thigh_joint": -0.7,
#             "RL_calf_joint": 1.5,
#             "RL_foot_joint": 0.0,
#             "RR_hip_joint": 0.0,
#             "RR_thigh_joint": 0.7,
#             "RR_calf_joint": -1.5,
#             "RR_foot_joint": 0.0,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "legs": DelayedPDActuatorCfg(
#             joint_names_expr=[
#                 ".*_hip_joint",
#                 ".*_thigh_joint",
#                 ".*_calf_joint",
#             ],
#             effort_limit=120.0,
#             stiffness=160.0,
#             damping=5,
#             friction=0.0,
#             min_delay=0,
#             max_delay=8,
#         ),
#         "wheel": ImplicitActuatorCfg(
#             joint_names_expr=[".*_foot_joint"],
#             effort_limit_sim=60.0,
#             velocity_limit_sim=16.965,
#             stiffness=0.0,
#             damping=1.0,
#             friction=0.0,
#         ),
#     },
# )
# """Configuration of OW wheel robot using new URDF from ow_wheel_description."""


THUNDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/thunder/urdf/thunder.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),  # 适当的初始高度
        joint_pos={
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": -0.64,
            "FR_calf_joint": 1.6,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.64,
            "FL_calf_joint": -1.6,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.64,
            "RR_calf_joint": -1.6,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": -0.64,
            "RL_calf_joint": 1.6,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,  # 167 rpm ≈ 17.48 rad/s (额定负载转速)
            stiffness=160.0,
            damping=5.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,  # 167 rpm ≈ 17.48 rad/s
            stiffness=160.0,
            damping=5.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,  # 167 rpm ≈ 17.48 rad/s
            stiffness=160.0,
            damping=5.0,
            friction=0.0,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=60.0,  # 轮子力矩限制 60Nm
            velocity_limit_sim=16.956,  # 轮子速度限制
            stiffness=0.0,  # 轮子无刚度
            damping=1.0,  # 轻微阻尼
            friction=0.0,
        ),
    },
)
"""Configuration for Thunder wheeled-legged robot with optimized parameters."""


THUNDER_NOHEAD_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/thunder/urdf/thunder_nohead.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),  # 适当的初始高度
        # joint_pos={
        #     "FR_hip_joint": 0.0,
        #     "FR_thigh_joint": -0.64,
        #     "FR_calf_joint": 1.6,
        #     "FL_hip_joint": 0.0,
        #     "FL_thigh_joint": 0.64,
        #     "FL_calf_joint": -1.6,
        #     "RR_hip_joint": 0.0,
        #     "RR_thigh_joint": 0.64,
        #     "RR_calf_joint": -1.6,
        #     "RL_hip_joint": 0.0,
        #     "RL_thigh_joint": -0.64,
        #     "RL_calf_joint": 1.6,
        #     ".*_foot_joint": 0.0,
        # },
        joint_pos={
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": -0.8,
            "FR_calf_joint": 1.7,
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.7,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.8,
            "RR_calf_joint": -1.7,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": -0.8,
            "RL_calf_joint": 1.7,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,  # RS04 额定 167 RPM
            stiffness=160.0,
            damping=20.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=160.0,
            damping=20.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=160.0,
            damping=20.0,
            friction=0.0,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=60.0,  # 轮子力矩限制 60Nm
            velocity_limit_sim=16.956,  # 轮子速度限制
            stiffness=0.0,  # 轮子无刚度
            damping=1.0,  # 0.15→1.0 防止轮子振荡
            friction=0.0,
        ),
    },
)
"""Configuration for Thunder wheeled-legged robot without head (nohead variant)."""


THUNDER_V3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/thunder_v3/urdf/wheeled_legged_dog_v3.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            "fr_hip_joint": 0.0,
            "fr_thigh_joint": -0.8,
            "fr_calf_joint": 1.7,
            "fl_hip_joint": 0.0,
            "fl_thigh_joint": 0.8,
            "fl_calf_joint": -1.7,
            "rr_hip_joint": 0.0,
            "rr_thigh_joint": 0.8,
            "rr_calf_joint": -1.7,
            "rl_hip_joint": 0.0,
            "rl_thigh_joint": -0.8,
            "rl_calf_joint": 1.7,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=70.0,
            damping=15.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=100.0,
            damping=15.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=120.0,
            saturation_effort=120.0,
            velocity_limit=17.48,
            stiffness=120.0,
            damping=20.0,
            friction=0.0,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=16.956,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration for Thunder V3 wheeled-legged robot (wheeled_legged_dog_v3.urdf, lowercase joints)."""
