# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Play and export HIM (History-based Implicit Model) policy.

This script is used to test and export trained HIM policies for deployment.
It exports the policy as TorchScript JIT module and ONNX model, similar to HIMLoco.

Usage:
    python play_him.py \
        --task RobotLab-Isaac-Velocity-Rough-Thunder-Hist-v0 \
        --load_run thunder_hist/2025-01-01_12-00-00 \
        --checkpoint model_10000.pt \
        --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip
from rl_utils import camera_follow

# add argparse arguments
parser = argparse.ArgumentParser(description="Play and export HIM policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--export", action="store_true", default=True, help="Export policy as JIT and ONNX (default: True)."
)
parser.add_argument("--no_export", action="store_true", default=False, help="Disable policy export.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Handle export flag
if args_cli.no_export:
    args_cli.export = False

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

# Add rsl_rl to path for imports
from pathlib import Path
from tensordict import TensorDict

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401

rsl_rl_path = Path(__file__).resolve().parent
if str(rsl_rl_path) not in sys.path:
    sys.path.insert(0, str(rsl_rl_path))

# Import HIM modules
from runners.him_on_policy_runner import HIMOnPolicyRunner
from utils.export_him_policy import export_him_policy_as_jit, export_him_policy_as_onnx


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Play with HIM policy and export for deployment."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Extract history_length from environment observation configuration
    obs_manager = env.unwrapped.observation_manager
    history_len = 1  # Default fallback

    if "policy" in obs_manager._group_obs_term_cfgs:
        policy_term_cfgs = obs_manager._group_obs_term_cfgs["policy"]
        if policy_term_cfgs and len(policy_term_cfgs) > 0:
            first_term_cfg = policy_term_cfgs[0]
            history_len = getattr(first_term_cfg, "history_length", 1)

    # Create HIM training configuration (minimal, just for loading)
    train_cfg = {
        "runner": {
            "policy_class_name": "HIMActorCritic",
            "algorithm_class_name": "HIMPPO",
            "num_steps_per_env": 200,  # consistent with HIMLoco paper
            "save_interval": 200,
        },
        "algorithm": {
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,  # consistent with HIMLoco paper
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,  # consistent with HIMLoco paper
            "learning_rate": 1e-3,
            "max_grad_norm": 10.0,  # consistent with HIMLoco paper
            "use_clipped_value_loss": True,
            "schedule": "fixed",
            "desired_kl": 0.01,
        },
        "policy": {
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "init_noise_std": 1.0,
            "estimator_latent_dim": 16,
            "estimator_lr": 1e-3,
            "num_prototype": 32,  # consistent with HIMLoco paper
        },
        "history_len": history_len,
    }

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # Create HIM runner and load checkpoint
    runner = HIMOnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=None,
        device=agent_cfg.device,
    )
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Extract the actor-critic network
    actor_critic = runner.alg.actor_critic

    # Export policy as JIT and ONNX
    if args_cli.export:
        export_model_dir = os.path.join(log_dir, "exported")
        print(f"\n📦 Exporting HIM policy to: {export_model_dir}")

        try:
            # Export as TorchScript JIT
            export_him_policy_as_jit(actor_critic=actor_critic, path=export_model_dir, filename="policy.pt")

            # Export as ONNX
            obs = env.get_observations()
            if isinstance(obs, TensorDict):
                policy_obs = obs["policy"]
            else:
                policy_obs = obs
            input_shape = (1, policy_obs.shape[1])

            export_him_policy_as_onnx(
                actor_critic=actor_critic, path=export_model_dir, filename="policy.onnx", input_shape=input_shape
            )

            print(f"\n✅ Policy export complete!")
            print(f"   - JIT model: {os.path.join(export_model_dir, 'policy.pt')}")
            print(f"   - ONNX model: {os.path.join(export_model_dir, 'policy.onnx')}")
        except Exception as e:
            print(f"\n⚠️  Failed to export policy: {e}")
            import traceback

            traceback.print_exc()

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    print("\n🎮 Starting simulation...")
    print("   Press Ctrl+C to stop")
    print()

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actor_obs, _ = runner._extract_observations(obs)
            actions = policy(actor_obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
