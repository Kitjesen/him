# Thunder HIM adaptation — 2026-09-09

Base: Kitjesen/him commit `9ecb91f58bff391d918b33acd01e6f4b9b69b069`.
Thunder environment: doso-train C9 commit `a34a786993cef37d4b5e4e430b6da4fcede4b837`.
Use `train_thunder.py`; archived upstream entries/configurations are not supported by this adapter.

## Corrections

- Target encoder consumes the explicit next observation, not the last/current history frame.
- Estimator losses exclude autoreset transitions: a new episode is not a physical successor of the previous episode.
- Clean next body velocity and proprioception are extracted through a validated Thunder critic layout and converted to actor scales.
- Timeout value bootstrap uses pre-reset terminal critic observations.
- Each environment has its own five-frame history, filled with the reset observation on reset.
- Checkpoints count completed PPO updates and save both optimizers, learning rates, RNG, terrain/gait difficulty and environment counters. Physics trajectories and episode-specific randomized state restart; this is not bitwise simulator resume.
- Latent dimensions are checked consistently; action standard deviation is positive; the PyTorch distribution API is no longer overwritten.
- JIT normalization uses a float exponent and exported outputs are checked against eager inference.

## Preserved contract

Thunder v4 physics, collision, nominal actuators, action order/scales, C9 rewards, terrain and domain randomization remain unchanged. Runtime guards pin source revision and URDF hash.

Input: 53 values per frame, five frames oldest to newest (265 total). Critic: 274. Actions: 16.
Actor does not receive true base velocity, a terrain scan or phase input.
Deployment must maintain the same history cache; this is not a drop-in replacement for an old 53-input stateless policy. C9 checkpoints cannot directly resume HIM.

## Verification

- `python -m pytest tests/test_thunder_him.py -q`: 9 passed.
- Focused Ruff checks on adapter, runner, entry and regression tests passed.
- Isaac on RTX5090: 64 environments, 24 steps/update; two updates, independent process resume, one additional update. Final checkpoint: `model_3.pt`.
- Resumed curriculum tensors matched saved tensors; JIT/eager inference parity passed.
- Observed update durations: 1.808 s, 1.365 s, 1.853 s. These are small validation runs, not production throughput measurements.
- First runtime attempt exposed an adapter callback keyword mismatch on resume; fixed before the complete second validation.

This validates the training pipeline only. No long training, stair success, ONNX export or physical-robot deployment is claimed. Additional predictive heads are not implemented.

Example (existing compatible Isaac environment; use a new output directory):

```bash
python train_thunder.py --thunder-root /path/to/pinned/doso-train \
  --output /path/to/new/run --num-envs 64 --updates 2
```

To resume, add `--resume /path/to/model_N.pt` and use a new output directory. `--updates` is the number of additional updates. Configuration/source must match. Check `SUCCESS.json`, not just the process exit code. Freeze specialist curriculum, rewards and checkpoint frequency separately before long training.

## Attribution and distribution

HIM means Hybrid Internal Model. Preserve upstream attribution and SPDX headers. No root license was found in the cloned Kitjesen repository; official HIMLoco declares CC BY-NC-SA 4.0. Do not treat public cloning as clearance for commercial redistribution.
