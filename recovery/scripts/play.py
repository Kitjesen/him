# Copyright (c) 2026 Qiongpei Technology
# SPDX-License-Identifier: Apache-2.0

"""Evaluate / visualize fall recovery policy.

Usage:
    python recovery/scripts/play.py --checkpoint logs/recovery/model_best.pt --num_envs 16
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate recovery policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to recovery policy checkpoint")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
    parser.add_argument("--episode_length", type=float, default=5.0, help="Episode length in seconds")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--headless", action="store_true", help="Run without rendering")
    args = parser.parse_args()

    print(f"Recovery Policy Evaluation")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Environments: {args.num_envs}")
    print(f"  Episodes: {args.num_episodes}")
    print()
    print("TODO: Integrate with Isaac Lab environment for full evaluation.")
    print("Steps:")
    print("  1. Load ThunderRecoveryEnvCfg")
    print("  2. Load HIMActorCritic from checkpoint")
    print("  3. Run episodes with random fallen initialization")
    print("  4. Report success rate, recovery time, mean torque")


if __name__ == "__main__":
    main()
