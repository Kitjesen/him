"""Explicit Thunder HIM runner; never discovers privileged groups heuristically."""
import json
import os
from pathlib import Path
import time

import torch

from algorithms.him_ppo import HIMPPO
from modules.him_actor_critic import HIMActorCritic
from thunder_adapter import History, clean_targets
from utils.export_him_policy import PolicyExporterHIM


class ThunderRunner:
    def __init__(self, env, output, config, provenance):
        self.env, self.output, self.config, self.provenance = env, Path(output), config, provenance
        self.device = env.device
        self.policy = HIMActorCritic(265, 274, 53, 16, **config['policy']).to(self.device)
        self.alg = HIMPPO(self.policy, device=self.device, **config['algorithm'])
        self.alg.init_storage(env.num_envs, config['steps'], (265,), (274,), (16,))
        self.history = History(env.num_envs, self.device)
        self.iteration = 0
        self.output.mkdir(parents=True, exist_ok=True)
        self.writer = None

    def save(self):
        path = self.output/f'model_{self.iteration}.pt'
        state = dict(format='thunder-him-v1', completed_updates=self.iteration,
                     model=self.policy.state_dict(), ppo_optimizer=self.alg.optimizer.state_dict(),
                     estimator_optimizer=self.policy.estimator.optimizer.state_dict(),
                     ppo_lr=self.alg.learning_rate, estimator_lr=self.policy.estimator.learning_rate,
                     config=self.config, provenance=self.provenance, rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     curriculum=(self.env.unwrapped.him_training_state()
                                 if hasattr(self.env.unwrapped, 'him_training_state') else None),
                     resume_semantics='optimizer/iteration continuation; simulator and rollout reset')
        temp = path.with_suffix('.tmp')
        torch.save(state, temp)
        os.replace(temp, path)
        return path

    def load(self, checkpoint):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state.get('format') != 'thunder-him-v1':
            raise ValueError('Not a Thunder HIM checkpoint; C9 is not directly resumable')
        if state['config'] != self.config or state['provenance'] != self.provenance:
            raise ValueError('Checkpoint config/provenance differs; explicit new branch required')
        self.policy.load_state_dict(state['model'], strict=True)
        self.alg.optimizer.load_state_dict(state['ppo_optimizer'])
        self.policy.estimator.optimizer.load_state_dict(state['estimator_optimizer'])
        self.alg.learning_rate = state['ppo_lr']
        self.policy.estimator.learning_rate = state['estimator_lr']
        self.iteration = int(state['completed_updates'])
        if state['curriculum'] is not None:
            self.env.unwrapped.him_pending_state = state['curriculum']
        torch.set_rng_state(state['rng'])
        if state['cuda_rng'] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda_rng'])

    def learn(self, updates):
        if updates < 1:
            raise ValueError('Explicit positive update budget required')
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(str(self.output/'tensorboard'))
        self.env.reset()
        obs = self.env.get_observations()
        history = self.history.reset(obs['policy'])
        try:
            for _ in range(updates):
                start = time.perf_counter()
                with torch.no_grad():
                    for _ in range(self.config['steps']):
                        actions = self.alg.act(history, obs['critic'])
                        new_obs, reward, dones, info = self.env.step(actions)
                        teacher, velocity = clean_targets(new_obs['critic'])
                        terminal_values = None
                        if 'time_outs' in info and info['time_outs'].any():
                            terminal_values = self.policy.evaluate(self.env.unwrapped.him_terminal_critic)
                        self.alg.process_env_step(reward, dones, info, new_obs['critic'],
                                                  teacher, velocity, terminal_values)
                        # Storage has copied the old history; update the cache exactly once.
                        history = self.history.append(new_obs['policy'], dones)
                        obs = new_obs
                    self.alg.compute_returns(obs['critic'])
                losses = self.alg.update()
                if not torch.isfinite(torch.tensor(losses)).all():
                    raise ValueError('Nonfinite training losses')
                if not all(torch.isfinite(p).all() for p in self.policy.parameters()):
                    raise ValueError('Nonfinite parameters')
                self.iteration += 1
                elapsed = time.perf_counter()-start
                receipt = dict(completed_updates=self.iteration, seconds_per_update=elapsed,
                               loss=dict(zip(['value', 'ppo', 'velocity', 'swap'], losses)),
                               num_envs=self.env.num_envs, samples=self.env.num_envs*self.config['steps'])
                (self.output/'progress.json').write_text(json.dumps(receipt, indent=2))
                for name, value in receipt['loss'].items():
                    self.writer.add_scalar('Loss/'+name, value, self.iteration)
                self.writer.add_scalar('Perf/seconds_per_update', elapsed, self.iteration)
                for name, value in info.get('log', {}).items():
                    if isinstance(value, (int, float)) or isinstance(value, torch.Tensor) and value.numel() == 1:
                        self.writer.add_scalar('Env/'+name, value, self.iteration)
                print(json.dumps(receipt), flush=True)
                if self.iteration % self.config['save_interval'] == 0:
                    self.save()
            return self.save()
        finally:
            self.writer.close()

    def export(self):
        exporter = PolicyExporterHIM(self.policy).cpu().eval()
        sample = torch.randn(8, 265)
        with torch.no_grad():
            expected = self.policy.act_inference(sample.to(self.device)).cpu()
            scripted = torch.jit.script(exporter)
            torch.testing.assert_close(scripted(sample), expected, rtol=1e-5, atol=1e-6)
        path = self.output/'policy_history265.pt'
        scripted.save(str(path))
        (self.output/'export_contract.json').write_text(json.dumps(dict(
            input=[None,265], output=[None,16], per_frame=53, history=5,
            history_order='oldest_to_newest', reset='fill all frames from reset observation',
            action_semantics='Thunder original normalized 12 leg + 4 wheel actions',
            completed_updates=self.iteration, jit_parity_passed=True), indent=2))
        return path
