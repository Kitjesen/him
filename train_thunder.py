"""Explicit Thunder-v4 HIM entry. An update budget and output directory are mandatory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

EXPECTED_SOURCE = 'a34a786993cef37d4b5e4e430b6da4fcede4b837'
EXPECTED_URDF = '83ba02ca52863238ce4050fff7c122fb159e381fb106ba31e39e44bfc0b66af3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thunder-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.updates < 1 or args.num_envs < 1 or args.steps < 1 or args.num_envs*args.steps % 4:
        raise ValueError('Positive budget, environments and rollout divisible by four required')
    if args.output.exists():
        raise FileExistsError('Use a new output directory for each training/resume session')
    source_commit = subprocess.check_output(['git', '-C', str(args.thunder_root), 'rev-parse', 'HEAD'], text=True).strip()
    if source_commit != EXPECTED_SOURCE:
        raise ValueError('Unexpected Thunder source revision')
    if subprocess.check_output(['git', '-C', str(args.thunder_root), 'diff', 'HEAD', '--', 'source/doso_train'], text=True):
        raise ValueError('Thunder source has unreviewed tracked modifications')
    sys.path.insert(0, str(args.thunder_root/'source/doso_train'))
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False, device='cuda:0').app
    failed = False
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from doso_train.tasks.terrain_inward.rough_transfer_c9_env_cfg import DosoThunderRoughTransferC9EnvCfg
        from thunder_adapter import validate_observations
        from thunder_runner import ThunderRunner

        torch.set_num_threads(1)
        torch.manual_seed(42)
        cfg = DosoThunderRoughTransferC9EnvCfg()
        cfg.seed, cfg.scene.num_envs, cfg.sim.device = 42, args.num_envs, 'cuda:0'
        cfg.scene.robot.spawn.usd_dir = str(args.output/'usd')
        asset_hash = hashlib.sha256(Path(cfg.scene.robot.spawn.asset_path).read_bytes()).hexdigest()
        if asset_hash != EXPECTED_URDF:
            raise ValueError('Thunder v4 runtime URDF differs')
        if cfg.scene.robot.actuators['wheel'].effort_limit_sim != 17.:
            raise ValueError('Unexpected wheel actuator limit')
        for name in ('hip', 'thigh', 'calf'):
            actuator = cfg.scene.robot.actuators[name]
            if (actuator.stiffness, actuator.damping) != (90., 6.93):
                raise ValueError('Unexpected nominal leg PD')

        class TerminalCaptureEnv(ManagerBasedRLEnv):
            def him_training_state(self):
                terrain = self.scene.terrain
                gait = self.reward_manager.get_term_cfg('standard_motion_gait').func.state
                return dict(terrain_levels=terrain.terrain_levels.detach().cpu(),
                            terrain_types=terrain.terrain_types.detach().cpu(),
                            level=gait.level.detach().cpu(), level_cycles=gait.level_cycles.detach().cpu(),
                            height_levels=list(gait.height_levels), cycles_per_level=gait.cycles_per_level,
                            common_step_counter=self.common_step_counter,
                            sim_step_counter=self._sim_step_counter)

            def _reset_idx(self, env_ids):
                # Critic has no history: compute terminal value inputs before automatic reset.
                critic = self.observation_manager.compute_group('critic')
                if not hasattr(self, 'him_terminal_critic'):
                    self.him_terminal_critic = torch.zeros_like(critic)
                self.him_terminal_critic[env_ids] = critic[env_ids].detach()
                pending = getattr(self, 'him_pending_state', None)
                if pending is None:
                    super()._reset_idx(env_ids)
                    return
                # Restore difficulty before spawning; skip curriculum promotion/demotion
                # on this one reset because no episode from the checkpoint is simulated.
                terrain = self.scene.terrain
                gait = self.reward_manager.get_term_cfg('standard_motion_gait').func.state
                if pending['height_levels'] != list(gait.height_levels) or pending['cycles_per_level'] != gait.cycles_per_level:
                    raise ValueError('Gait curriculum contract changed')
                for owner, names in ((terrain, ('terrain_levels', 'terrain_types')), (gait, ('level', 'level_cycles'))):
                    for name in names:
                        target = getattr(owner, name)
                        value = pending[name].to(target.device)
                        if value.shape != target.shape or value.dtype != target.dtype or (value < 0).any():
                            raise ValueError('Invalid curriculum state: '+name)
                        target.copy_(value)
                terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
                self.scene.env_origins[:] = terrain.env_origins
                self.common_step_counter = pending['common_step_counter']
                self._sim_step_counter = pending['sim_step_counter']
                original_compute = self.curriculum_manager.compute
                self.curriculum_manager.compute = lambda env_ids: None
                try:
                    super()._reset_idx(env_ids)
                finally:
                    self.curriculum_manager.compute = original_compute
                restored = self.him_training_state()
                for name in ('terrain_levels', 'terrain_types', 'level', 'level_cycles'):
                    torch.testing.assert_close(restored[name], pending[name], rtol=0, atol=0)
                self.him_curriculum_restore_verified = True
                del self.him_pending_state

        env = RslRlVecEnvWrapper(TerminalCaptureEnv(cfg=cfg), clip_actions=None)
        try:
            validate_observations(env.unwrapped.observation_manager)
            if env.num_actions != 16:
                raise ValueError('Expected Thunder 16 actions')
            config = dict(steps=args.steps, save_interval=1,
                          policy=dict(estimator_latent_dim=16),
                          algorithm=dict(num_learning_epochs=5, num_mini_batches=4,
                                         learning_rate=1e-3, schedule='adaptive', gamma=.99, lam=.95,
                                         entropy_coef=.005, max_grad_norm=10.))
            code_root = Path(__file__).parent
            code_hashes = {p.relative_to(code_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in code_root.rglob('*.py')}
            provenance = dict(thunder_source=source_commit, thunder_urdf=asset_hash,
                              code_sha256=code_hashes, interface=[53,265,274,16],
                              task='Thunder C9 configuration, HIM architecture adaptation',
                              environment_overrides=['num_envs','seed','device','output_usd_cache'])
            runner = ThunderRunner(env, args.output, config, provenance)
            if args.resume:
                runner.load(args.resume)
            dump_yaml(str(args.output/'env.yaml'), cfg)
            (args.output/'config.json').write_text(json.dumps(config, indent=2))
            (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2))
            start = runner.iteration
            checkpoint = runner.learn(args.updates)
            exported = runner.export()
            receipt = dict(status='ADAPTATION_VALIDATED_NOT_LOCOMOTION_QUALIFIED', start_update=start,
                           completed_updates=runner.iteration, num_envs=args.num_envs,
                           curriculum_restore_verified=getattr(env.unwrapped, 'him_curriculum_restore_verified', False),
                           checkpoint=str(checkpoint), export=str(exported),
                           note='Original C9 rewards/terrain/DR retained; no stairs-specialist performance claim')
            (args.output/'SUCCESS.json').write_text(json.dumps(receipt, indent=2))
            print(json.dumps(receipt), flush=True)
        finally:
            env.close()
    except BaseException:
        import traceback
        traceback.print_exc()
        failed = True
    finally:
        # Isaac may exit in close(); emit explicit result before that, and require SUCCESS externally.
        if failed:
            print('ADAPTATION_FAILED', flush=True)
        app.close()
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    main()
