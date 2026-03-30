# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.thunder_hist.rough_env_cfg import (  # noqa: E501
    ThunderHistRoughEnvCfg,
)


@configclass
class ThunderHistFlatEnvCfg(ThunderHistRoughEnvCfg):
    """Thunder flat terrain env - no height scan, no curriculum."""

    def __post_init__(self):
        super().__post_init__()

        # terrain -> flat plane
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # remove height scanner sensor
        self.scene.height_scanner = None

        # remove all obs groups/terms referencing height_scanner
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.observations.height_scan_group = None  # fix: was missing

        # reward using height_scanner sensor
        self.rewards.base_height_l2.params["sensor_cfg"] = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # zero-weight rewards -> None
        if self.__class__.__name__ == "ThunderHistFlatEnvCfg":
            self.disable_zero_weight_rewards()
