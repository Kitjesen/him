# Training Log

## Experiment Naming

Format: `mlp_v{N}[suffix]` — runs stored under `logs/rsl_rl/thunder_hist_rough/`

---

## Run History

| Run | Base | entropy_coef | Key Change | Best Terrain | Best Reward | Notes |
|-----|------|-------------|-----------|--------------|-------------|-------|
| v1 | fresh | 0.01 | baseline (no DR) | ~6 | ~235 | noise 1.0→0.73, vel_error 0.48 — training healthy |
| v6b | — | 0.001 | — | — | — | intermediate |
| v7 | v6b | 0.001 | — | — | — | |
| v7b | v7 | 0.001 | — | **8.71** | **~242** | noise→0.55, entropy too low → stuck at 8.71 |
| v7c | v7b/model_400 | **0.01** | restored entropy | 6.05 | 205 | noise stuck at 1.32, action_rate -2.49 → killed at iter 1773 |
| v7d | v7b/model_400 | **0.005** | reduced entropy | 6.0 stuck | 237 | noise 0.55→0.88 diverged, killed at iter 679 |
| v7e | v7b/model_400 | **0.001** | back to v7b config | ongoing | 252 | noise 0.53, let it run to 5000+ |

---

## Config Reference

Key file: `config/thunder_hist/agents/rsl_rl_ppo_cfg.py`

| Param | v7b | v7c |
|-------|-----|-----|
| `entropy_coef` | 0.001 | 0.01 |
| `num_steps_per_env` | 200 | 200 |
| `learning_rate` | 1e-3 | 1e-3 |
| `latent_dim` | 16 | 16 |
| `num_prototype` | 32 | 32 |

---

## Observations

- **entropy_coef=0.001**: noise converges well (0.74→0.51), terrain reaches 8.7 — correct setting
- **entropy_coef=0.005**: noise diverges (0.55→0.88), terrain stuck at 6 — too high
- **entropy_coef=0.01**: noise explodes (1.32+), action_rate -2.49 — far too high
- **Root cause**: entropy loss actively pushes noise_std UP; any coef >0.001 causes divergence on this task
- v7b stuck at 8.71 was NOT entropy issue — was killed too early (iter 563), needs 3000+ iter
- DR (domain randomization) consistently causes noise divergence — do not add until baseline is stable
