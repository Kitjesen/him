# Thunder Fall Recovery Policy

> Based on "Learning to Recover: Dynamic Reward Shaping with Wheel-Leg Coordination" (arXiv:2506.05516)

## What is this?

A standalone RL policy that teaches a wheeled-legged robot to **stand up after falling down**. Trained separately from the locomotion policy, then combined at deployment.

## Key Idea: Episode-based Dynamic Reward Shaping (ED)

```
Episode start (t≈0):  ED ≈ 0  →  Robot freely explores recovery strategies
                                   (rolling, wheel-assisted flipping, etc.)

Episode end (t≈T):    ED → 1  →  Full standing reward kicks in
                                   (precise posture convergence)
```

This solves the "stand still and do nothing" local optimum that plagues sparse recovery rewards.

## Training Pipeline

```
1. Random orientation + drop from 1.1m
2. 2s free-fall (torques disabled) → robot collapses on ground
3. 3s recovery window (policy active, 50Hz control)
4. Fixed 5s episode, NO early termination
```

## File Structure

```
recovery/
├── README.md                 ← You are here
├── config/
│   ├── recovery_env_cfg.py   ← Environment: flat terrain, fallen init, 5s episode
│   └── recovery_ppo_cfg.py   ← PPO hyperparameters
├── mdp/
│   └── rewards.py            ← ED/CW reward shaping + 9 reward terms
└── scripts/
    ├── train.py              ← Training entry point
    └── play.py               ← Evaluation / visualization
```

## Quick Start

```bash
# Train (on server with Isaac Lab)
python recovery/scripts/train.py --num_envs 2048 --headless --max_iterations 5000

# Evaluate
python recovery/scripts/play.py --checkpoint recovery_policy.pt
```

## Reward Function (Paper Table I)

| Type | Reward | Scale | Description |
|------|--------|-------|-------------|
| Task (×ED) | stand_joint_pos | 42 | Joints return to default angles |
| Task (×ED) | base_height | 120 | Body height reaches 0.55m |
| Task (×ED) | base_orientation | 50 | Body returns to upright |
| Behavior (×CW) | body_collision | -5e-2 | Avoid damaging ground impacts |
| Behavior (×CW) | action_rate | -1e-2 | Smooth leg actions (wheels free) |
| Constant | joint_velocity | -2e-2 | Joint speed penalty |
| Constant | torques | -2.5e-5 | Torque penalty |
| Constant | acceleration | -2.5e-7 | Acceleration penalty |
| Constant | wheel_velocity | -2e-2 | Excessive wheel spin penalty |

## Success Criteria

- Base height > 0.42m (76% of standing height)
- Joint deviation < 0.5 rad from default
- Max joint velocity < 0.1 rad/s
- Orientation error < 0.1

## Deployment: Policy Switching

```python
if gravity_z > -0.5 or base_height < 0.25:
    action = recovery_policy(obs)
else:
    action = locomotion_policy(obs)
```

## Reference

- Paper: [Learning to Recover (arXiv:2506.05516)](https://arxiv.org/abs/2506.05516)
- Results: KYON 99.1%, Go2-W 97.8% success rate
- Wheel-leg coordination reduces joint torque 15-26%
