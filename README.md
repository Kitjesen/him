# Thunder-HIM: History-based Implicit Model for Quadruped Locomotion

> Locomotion training framework for the **Thunder** wheeled-legged quadruped robot,
> built on **Isaac Lab** + **RSL-RL** with **HIM** (History-based Implicit Model) and **SwAV** contrastive learning.

---

## Overview

Thunder-HIM implements the HIMLoco architecture for a 12-DOF legged + 4-wheel quadruped robot:

- **Asymmetric Actor-Critic**: actor uses estimated latent features; critic uses privileged observations (full state + height scan)
- **HIM Estimator**: encodes 5-frame proprioceptive history → velocity estimate (3D) + latent code
- **SwAV Contrastive Learning**: Swapped Assignment between Views for robust latent representations
- **Terrain Curriculum**: automatic difficulty progression from flat to rough terrain (stairs, boxes, slopes, up to level 9)

### Architecture

```
Observation History (5 frames × 57 dims = 285)
        │
        ▼
  HIM Estimator (Encoder)
        │
  ┌─────┴─────┐
  vel̂ (3)    latent (16)
        │
  current_obs (57) ──► Actor ──► actions (16)
                                  [12 legs + 4 wheels]

Critic ◄── privileged obs (height_scan + full state)
```

### Robot: Thunder

| Property | Value |
|----------|-------|
| DOF | 12 leg joints + 4 wheel joints = 16 total |
| Leg control | JointPosition (PD) |
| Wheel control | JointVelocity |
| History length | 5 frames |
| Policy obs dim | 285 (5 × 57) |
| Critic obs dim | 487 (privileged + height scan) |

---

## Repository Structure

```
thunder-him/
├── train_him.py                     # Main training entry point
├── play_him.py                      # Policy rollout & visualization
├── cli_args.py                      # CLI argument helpers
│
├── modules/
│   ├── him_actor_critic.py          # HIMActorCritic: actor + critic + std
│   └── him_estimator.py             # HIMEstimator: encoder + target + SwAV protos
│
├── runners/
│   └── him_on_policy_runner.py      # Training loop, obs reshaping, checkpointing
│
├── algorithms/
│   └── him_ppo.py                   # HIMPPO: PPO + estimation loss + swap loss
│
├── storage/
│   └── him_rollout_storage.py       # Rollout buffer for HIM training
│
├── utils/
│   ├── observation_reshaper.py      # Isaac Lab → HIM obs format conversion
│   └── export_him_policy.py        # ONNX export for deployment
│
├── config/thunder_hist/
│   ├── rough_env_cfg.py             # Rough terrain environment config
│   ├── flat_env_cfg.py              # Flat terrain environment config
│   ├── __init__.py                  # Gym task registration
│   └── agents/
│       └── rsl_rl_ppo_cfg.py        # PPO hyperparameters
│
└── mdp/
    ├── rewards.py                   # All reward functions
    └── curriculums.py               # Terrain & command curricula
```

---

## Training

### Prerequisites

- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) (v0.46+)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) (v3.0.1+)
- [robot_lab](https://github.com/fan-ziqi/robot_lab) task definitions

### Quick Start

```bash
# Fresh training
python train_him.py \
  --task RobotLab-Isaac-Velocity-Rough-Thunder-Hist-v0 \
  --num_envs 1024 \
  --headless

# Resume from checkpoint
python train_him.py \
  --task RobotLab-Isaac-Velocity-Rough-Thunder-Hist-v0 \
  --num_envs 1024 --headless \
  --resume \
  --load_run thunder_hist_rough/<run_dir> \
  --checkpoint model_400.pt \
  --run_name my_experiment
```

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_envs` | 1024 | Parallel environments |
| `num_steps_per_env` | 200 | Rollout length (matches HIMLoco paper) |
| `entropy_coef` | 0.01 | Exploration pressure |
| `learning_rate` | 1e-3 | Adaptive schedule |
| `latent_dim` | 16 | HIM latent code size |
| `num_prototype` | 32 | SwAV prototypes |
| `max_grad_norm` | 10.0 | Matches HIMLoco paper |
| `history_len` | 5 | Observation history frames |

### Observation Groups

```python
# Actor sees: proprioceptive history only (5 frames)
policy = ["ang_vel(3)", "gravity(3)", "cmd(3)", "jpos(16)", "jvel(16)", "actions(16)"]  # × 5 = 285

# Critic sees: privileged info (not available at deployment)
critic = ["base_lin_vel(3)", ...same as policy...] + height_scan(187)  # = 487
```

---

## Reward Design

| Reward | Weight | Description |
|--------|--------|-------------|
| `track_lin_vel_xy_exp` | +6.0 | XY velocity tracking |
| `track_ang_vel_z_exp` | +3.0 | Yaw rate tracking |
| `upward` | +2.0 | Keep body upright |
| `action_rate_l2` | -0.05 | Smooth actions |
| `feet_stumble` | -5.0 | Penalize stumbling |
| `joint_pos_limits` | -2.0 | Stay in joint limits |
| `contact_forces` | -5e-4 | Reduce impact forces |
| `lin_vel_z_l2` | -2.0 | No vertical bouncing |

---

## Terrain Curriculum

Rough terrain with 6 sub-types:
- Pyramid stairs (ascending/descending) — 40%
- Random boxes — 20%
- Random rough — 20%
- Pyramid slopes — 20%

Stair height range: 5 cm (level 0) → 23 cm (level 9)

Curriculum automatically advances robot to harder terrain when success rate is high.

---

## Deployment

Export trained policy to ONNX:
```bash
python utils/export_him_policy.py \
  --checkpoint logs/rsl_rl/thunder_hist_rough/<run>/model_best.pt
```

The exported model:
- **Input**: `[1, 285]` — 5-frame proprioceptive history
- **Output**: `[1, 16]` — joint position/velocity targets

---

## Training Results (v7c)

| Metric | Value |
|--------|-------|
| Max terrain level | ~8.7 / 9.0 |
| Mean reward | ~245 |
| Noise std | ~0.55 |
| Iterations | 500+ |
| Episode timeout rate | ~100% |

---

## References

- [HIMLoco: History-based Implicit Model for Locomotion](https://arxiv.org/abs/2312.12652)
- [SwAV: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)

---

## License

Apache-2.0 — see individual file headers.
