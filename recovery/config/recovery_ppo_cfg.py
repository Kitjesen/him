# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""PPO configuration for fall recovery training."""


RECOVERY_PPO_CFG = {
    "runner": {
        "policy_class_name": "HIMActorCritic",
        "algorithm_class_name": "HIMPPO",
        "num_steps_per_env": 150,    # 3s effective window at 50Hz
        "save_interval": 200,
    },
    "algorithm": {
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.005,       # slightly higher than locomotion for exploration
        "learning_rate": 0.001,
        "max_grad_norm": 10.0,
        "use_clipped_value_loss": True,
        "schedule": "adaptive",
        "desired_kl": 0.01,
    },
    "policy": {
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "init_noise_std": 1.0,
        "estimator_latent_dim": 16,
        "estimator_lr": 0.001,
        "num_prototype": 32,
        "estimation_loss_weight": 1.0,
        "swap_loss_weight": 1.0,
    },
    "history_len": 5,
}
