# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.

    - experiment_name: 实验名，决定日志与 checkpoint 存放的二级目录名，
        例如 logs/rsl_rl/<experiment_name>/<run_name>/...
      如果不指定，则使用 agent 配置里的默认 experiment_name（如 unitree_go2_flat）。

    - run_name: run 的名字，会作为时间戳后的后缀追加到日志目录名中，
        便于在同一 experiment_name 下区分多次不同设置的训练。

    - resume: 是否从已有 checkpoint 恢复训练。
        - 在 train.py 中，当 resume=True 时，会根据 load_run / load_checkpoint 或默认规则找到 checkpoint 并加载。

    - load_run: 要恢复的 run 目录名，例如 "2025-11-28_19-21-18"，
        通常与 logs/rsl_rl/<experiment_name>/<load_run>/ 对应。

    - checkpoint: 要加载的 checkpoint 路径或文件名：
        - 如果是相对路径（如 "model_01000.pt"），会和 log_root_path/load_run 组合；
        - 如果是完整路径，则直接按该路径查找（play.py 中通过 retrieve_file_path 解析）。

    - logger: 选择使用哪种日志后台（wandb / tensorboard / neptune），
        用于在训练时记录曲线、模型等信息。None 表示只用本地日志目录，不接第三方服务。

    - log_project_name: 当 logger 使用 wandb 或 neptune 时，对应远端项目的名称，
        便于在这些平台中把同一项目的多次实验归类。
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment folder where logs will be stored.",
    )
    arg_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name suffix to the log directory.",
    )
    # -- load arguments
    arg_group.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Whether to resume from a checkpoint.",
    )
    arg_group.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Name of the run folder to resume from.",
    )
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger",
        type=str,
        default=None,
        choices={"wandb", "tensorboard", "neptune"},
        help="Logger module to use.",
    )
    arg_group.add_argument(
        "--log_project_name",
        type=str,
        default=None,
        help="Name of the logging project when using wandb or neptune.",
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
