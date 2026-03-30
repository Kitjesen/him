#!/usr/bin/env python3
"""
Train RL agent with HIM (History-based Implicit Model) and SwAV.

This script trains a robot policy using:
1. HIMLoco-style asymmetric actor-critic
2. SwAV (Swapped Assignment between Views) for latent learning
3. Temporal observation encoding for velocity and latent prediction

Features:
- Asymmetric Actor-Critic (Actor uses estimated features, Critic uses privileged info)
- SwAV contrastive learning for robust representation
- Standard PPO for policy optimization
- Automatic history_length detection from environment configuration
- Episode tracking and detailed logging
/home/bsrl/hongsenpang/RLbased/robot_lab/scripts/reinforcement_learning/rsl_rl
Usage:
python /home/bsrl/hongsenpang/RLbased/robot_lab/scripts/reinforcement_learning/rsl_rl/train_him.py \\
    --task RobotLab-Isaac-Velocity-Rough-Thunder-Hist-v0 \\
    --num_envs 1024 \\
    --headless
"""

import argparse
import os
import sys
from datetime import datetime

# Isaac Sim must be launched first
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train RL agent with HIM and SwAV.")

# Standard training arguments
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")

# HIM-specific arguments
parser.add_argument(
    "--latent_dim",
    type=int,
    default=16,
    help="Dimension of latent features (default: 16)",
)
parser.add_argument(
    "--estimator_lr",
    type=float,
    default=1e-3,
    help="Learning rate for HIM estimator",
)
parser.add_argument(
    "--num_prototype",
    type=int,
    default=32,
    help="Number of prototypes for SwAV clustering (default: 32, same as HIMLoco paper)",
)
parser.add_argument(
    "--estimation_loss_weight",
    type=float,
    default=1.0,
    help="Weight for estimation loss (default: 1.0)",
)
parser.add_argument(
    "--swap_loss_weight",
    type=float,
    default=1.0,
    help="Weight for SwAV swap loss (default: 1.0)",
)

# Import cli_args after parser is defined
import cli_args  # isort: skip

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""
import importlib.metadata as metadata
from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    print(
        f"Please install the correct version of RSL-RL.\n"
        f"Existing version is: '{installed_version}' and required version is: '{RSL_RL_VERSION}'."
    )
    exit(1)

"""Rest everything follows."""
import gymnasium as gym
import torch
from tensordict import TensorDict

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401

# Add rsl_rl to path for imports
import sys
from pathlib import Path
rsl_rl_path = Path(__file__).resolve().parent
if str(rsl_rl_path) not in sys.path:
    sys.path.insert(0, str(rsl_rl_path))

# Import HIM modules
from runners.him_on_policy_runner import HIMOnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with HIM (History-based Implicit Model) and SwAV."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if args_cli.run_name:
        agent_cfg.run_name = args_cli.run_name
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        # Note: IO descriptors are only supported for manager based RL environments
        pass

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    resume_path = None
    if agent_cfg.resume:
        # If load_run contains a path separator, it means we're loading from a different experiment
        # In that case, use the parent directory (logs/rsl_rl) as the search path
        if agent_cfg.load_run and "/" in agent_cfg.load_run:
            # Extract experiment name and run dir from load_run (e.g., "thunder_gang_flat/2025-11-03_15-12-29")
            load_path_parts = agent_cfg.load_run.split("/", 1)
            experiment_name_from_load = load_path_parts[0]
            run_dir_from_load = load_path_parts[1]
            # Use parent directory to allow cross-experiment loading
            parent_log_path = os.path.join("logs", "rsl_rl")
            parent_log_path = os.path.abspath(parent_log_path)
            resume_path = get_checkpoint_path(
                parent_log_path, 
                experiment_name_from_load, 
                agent_cfg.load_checkpoint, 
                other_dirs=[run_dir_from_load]
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Extract history_length directly from environment observation configuration
    obs_manager = env.unwrapped.observation_manager
    # Read history_length from policy group's first term
    # Note: All terms in the same group share the same history_length (set in ObsGroup.__post_init__)
    history_len = 1  # Default fallback
    if 'policy' in obs_manager._group_obs_term_cfgs:
        policy_term_cfgs = obs_manager._group_obs_term_cfgs['policy']
        if policy_term_cfgs and len(policy_term_cfgs) > 0:
            # Read from first term (all terms in group have same history_length)
            first_term_cfg = policy_term_cfgs[0]
            history_len = getattr(first_term_cfg, 'history_length', 1)
    
    # Verify with actual observation dimensions
    obs = env.get_observations()
    policy_obs_dim = obs['policy'].shape[1] if isinstance(obs, TensorDict) else obs.shape[1]
    one_step_obs_dim = policy_obs_dim // history_len if history_len > 0 else policy_obs_dim
    print(f"[INFO] History configuration: length={history_len}, policy_obs={policy_obs_dim}, one_step_obs={one_step_obs_dim}")

    # Create HIM Training Configuration
    train_cfg = {
        "runner": {
            "policy_class_name": "HIMActorCritic",
            "algorithm_class_name": "HIMPPO",
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "save_interval": agent_cfg.save_interval,
        },
        "algorithm": {
            "num_learning_epochs": agent_cfg.algorithm.num_learning_epochs,
            "num_mini_batches": agent_cfg.algorithm.num_mini_batches,
            "clip_param": agent_cfg.algorithm.clip_param,
            "gamma": agent_cfg.algorithm.gamma,
            "lam": agent_cfg.algorithm.lam,
            "value_loss_coef": agent_cfg.algorithm.value_loss_coef,
            "entropy_coef": agent_cfg.algorithm.entropy_coef,
            "learning_rate": agent_cfg.algorithm.learning_rate,
            "max_grad_norm": agent_cfg.algorithm.max_grad_norm,
            "use_clipped_value_loss": agent_cfg.algorithm.use_clipped_value_loss,
            "schedule": agent_cfg.algorithm.schedule,
            "desired_kl": agent_cfg.algorithm.desired_kl,
        },
        "policy": {
            "actor_hidden_dims": agent_cfg.policy.actor_hidden_dims,
            "critic_hidden_dims": agent_cfg.policy.critic_hidden_dims,
            "activation": agent_cfg.policy.activation,
            "init_noise_std": agent_cfg.policy.init_noise_std,
            "estimator_latent_dim": args_cli.latent_dim,
            "estimator_lr": args_cli.estimator_lr,
            "num_prototype": args_cli.num_prototype,
            "estimation_loss_weight": args_cli.estimation_loss_weight,
            "swap_loss_weight": args_cli.swap_loss_weight,
        },
        "history_len": history_len,
    }

    # create runner from rsl-rl (HIM-specific)
    runner = HIMOnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=agent_cfg.device,
    )
    
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "train.yaml"), train_cfg)

    # Print training info (consistent with train.py style)
    if agent_cfg.resume:
        print(f"[INFO]: Resuming training from iteration {runner.current_learning_iteration}")
        print(f"[INFO]: Will train for {agent_cfg.max_iterations} more iterations")
        print(f"[INFO]: Target iteration: {runner.current_learning_iteration + agent_cfg.max_iterations}")
    else:
        print(f"[INFO]: Starting new training")
        print(f"[INFO]: Max iterations: {agent_cfg.max_iterations}")
    print(f"[INFO]: History length: {history_len}")
    print(f"[INFO]: Latent dim: {args_cli.latent_dim}")
    print(f"[INFO]: SwAV prototypes: {args_cli.num_prototype}")
    print(f"[INFO]: Estimation loss weight: {args_cli.estimation_loss_weight}")
    print(f"[INFO]: Swap loss weight: {args_cli.swap_loss_weight}")

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

