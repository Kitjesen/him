# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ThunderHistRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Thunder历史观测配置的PPO运行器 - 针对历史观测优化网络结构"""

    num_steps_per_env = 200  # 与HIMLoco论文完全一致（论文用4096×200）
    max_iterations = 20000
    save_interval = 200
    experiment_name = "thunder_hist_rough"
    class_name = "OnPolicyRunner"
    obs_groups = {"policy": ["policy"], "critic": ["critic", "height_scan_group"]}

    # 续接训练参数
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # 针对历史观测增大网络容量
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=10.0,  # 与HIMLoco论文一致（论文用10.0，不是1.0）
    )


@configclass
class ThunderHistFlatPPORunnerCfg(ThunderHistRoughPPORunnerCfg):
    """Thunder历史观测平坦地形PPO配置"""

    max_iterations = 20000
    experiment_name = "thunder_hist_flat"
    class_name = "OnPolicyRunner"
