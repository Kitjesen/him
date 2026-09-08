"""Thunder-only observation contract. No Isaac imports; deployable history semantics."""
import torch

POLICY_NAMES = ['base_ang_vel', 'projected_gravity', 'velocity_commands', 'joint_pos', 'joint_vel', 'actions']
POLICY_DIMS = [3, 3, 3, 12, 16, 16]


class History:
    def __init__(self, num_envs, device, frames=5, frame_dim=53):
        self.data = torch.zeros(num_envs, frames, frame_dim, device=device)

    def reset(self, observation, ids=None):
        if ids is None:
            self.data[:] = observation[:, None, :]
        else:
            self.data[ids] = observation[ids, None, :]
        return self.data.flatten(1)

    def append(self, observation, dones):
        self.data[:, :-1] = self.data[:, 1:].clone()
        self.data[:, -1] = observation
        self.reset(observation, dones.bool())
        return self.data.flatten(1)


def validate_observations(manager):
    if manager.active_terms['policy'] != POLICY_NAMES:
        raise ValueError('Thunder policy term order changed')
    if manager.active_terms['critic'][:7] != ['base_lin_vel', *POLICY_NAMES]:
        raise ValueError('Thunder critic prefix changed; cannot extract HIM targets')
    if list(manager.group_obs_term_dim['policy']) != [(n,) for n in POLICY_DIMS]:
        raise ValueError('Thunder single-frame observation dimensions changed')
    if list(manager.group_obs_term_dim['critic'])[:7] != [(3,), *[(n,) for n in POLICY_DIMS]]:
        raise ValueError('Thunder critic term dimensions changed')
    if manager.group_obs_dim['critic'] != (274,):
        raise ValueError('Expected Thunder 274D critic')
    # The clean teacher uses critic terms; reject a changed signal or scale contract.
    policy_terms = manager._group_obs_term_cfgs['policy']
    critic_terms = manager._group_obs_term_cfgs['critic'][1:7]
    for pt, ct, scale in zip(policy_terms, critic_terms, [.25, 1., 1., 1., .05, 1.], strict=True):
        if pt.func != ct.func or pt.scale != scale or ct.scale != 1. or pt.clip != ct.clip:
            raise ValueError('Thunder policy/teacher signal or scaling contract changed')


def clean_targets(critic):
    if critic.ndim != 2 or critic.shape[1] != 274:
        raise ValueError('Expected explicit Thunder critic layout')
    observation = critic[:, 3:56].clone()
    observation[:, :3] *= .25
    observation[:, 21:37] *= .05
    return observation, critic[:, :3].clone()
