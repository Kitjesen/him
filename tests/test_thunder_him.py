import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.him_actor_critic import HIMActorCritic  # noqa: E402
from thunder_adapter import History, clean_targets  # noqa: E402
from thunder_runner import ThunderRunner  # noqa: E402
from utils.export_him_policy import PolicyExporterHIM  # noqa: E402


@pytest.fixture
def policy():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    with contextlib.redirect_stdout(io.StringIO()):
        return HIMActorCritic(265, 274, 53, 16)


def test_history_per_environment_reset_and_chronology():
    history = History(2, 'cpu', frames=3, frame_dim=1)
    history.reset(torch.tensor([[1.], [2.]]))
    history.append(torch.tensor([[3.], [4.]]), torch.tensor([False, True]))
    torch.testing.assert_close(history.data[..., 0], torch.tensor([[1., 1., 3.], [4., 4., 4.]]))


def test_explicit_clean_target_scaling():
    critic = torch.arange(274).float()[None, :]
    target, vel = clean_targets(critic)
    torch.testing.assert_close(vel, critic[:, :3])
    torch.testing.assert_close(target[:, :3], critic[:, 3:6]*.25)
    torch.testing.assert_close(target[:, 21:37], critic[:, 24:40]*.05)
    torch.testing.assert_close(target[:, 37:], critic[:, 40:56])


def test_next_frame_and_terminal_mask(policy):
    seen = []
    handle = policy.estimator.target.register_forward_pre_hook(lambda m, a: seen.append(a[0].clone()))
    history = torch.randn(4, 265)
    future = torch.full((4, 53), 3.)
    vel = torch.full((4, 3), 4.)
    future[1] = torch.nan
    vel[1] = torch.nan
    before = policy.estimator.encoder[0].weight.detach().clone()
    losses = policy.estimator.update(history, future, vel, torch.tensor([True, False, True, True]))
    handle.remove()
    torch.testing.assert_close(seen[0], torch.full((3, 53), 3.))
    assert torch.isfinite(torch.tensor(losses)).all()
    assert not torch.equal(before, policy.estimator.encoder[0].weight)


def test_all_reset_skips_estimator_update(policy):
    before = {k: v.clone() for k, v in policy.estimator.state_dict().items()}
    losses = policy.estimator.update(torch.zeros(4, 265), torch.zeros(4, 53),
                                     torch.zeros(4, 3), torch.zeros(4, dtype=torch.bool))
    assert losses == (0., 0.)
    assert all(torch.equal(v, before[k]) for k, v in policy.estimator.state_dict().items())
    assert not policy.estimator.optimizer.state


def test_latent_dimension_and_std_are_valid():
    with contextlib.redirect_stdout(io.StringIO()):
        network = HIMActorCritic(265, 274, 53, 16, estimator_latent_dim=8)
    assert network.estimator.num_latent == 8
    assert network.act_inference(torch.zeros(2, 265)).shape == (2, 16)
    with torch.no_grad():
        network.log_std.fill_(-100)
    assert (network.std > 0).all()
    with pytest.raises(ValueError, match='integer'):
        HIMActorCritic(266, 274, 53, 16)


def test_actor_gradient_does_not_update_estimator(policy):
    policy.act_inference(torch.randn(4, 265)).sum().backward()
    assert all(p.grad is None for p in policy.estimator.parameters())
    assert policy.actor[0].weight.grad is not None


def test_jit_export_matches(policy, tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        exported = torch.jit.script(PolicyExporterHIM(policy))
    sample = torch.randn(4, 265)
    exported.save(str(tmp_path/'policy.pt'))
    loaded = torch.jit.load(str(tmp_path/'policy.pt'))
    torch.testing.assert_close(loaded(sample), policy.act_inference(sample))


class FakeEnv:
    num_envs, device = 4, 'cpu'

    def __init__(self):
        self.unwrapped = SimpleNamespace(him_terminal_critic=torch.ones(4, 274))
        self.counter = 0

    def reset(self):
        self.counter = 0

    def get_observations(self):
        return dict(policy=torch.zeros(4, 53), critic=torch.zeros(4, 274))

    def step(self, actions):
        self.counter += 1
        dones = torch.tensor([True, False, False, False])
        return self.get_observations(), torch.ones(4), dones, {'time_outs': dones}


def runner_config():
    return dict(steps=2, save_interval=1, policy=dict(estimator_latent_dim=16),
                algorithm=dict(num_learning_epochs=1, num_mini_batches=1, learning_rate=1e-3))


def test_actual_ppo_checkpoint_and_resume(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        runner = ThunderRunner(FakeEnv(), tmp_path/'first', runner_config(), {'source': 'test'})
        checkpoint = runner.learn(2)
        saved = torch.load(tmp_path/'first/model_1.pt', weights_only=True)
        assert saved['completed_updates'] == 1
        resumed = ThunderRunner(FakeEnv(), tmp_path/'second', runner_config(), {'source': 'test'})
        resumed.load(checkpoint)
        assert resumed.iteration == 2
        assert resumed.alg.optimizer.state
        assert resumed.policy.estimator.optimizer.state
        for a, b in zip(runner.policy.parameters(), resumed.policy.parameters(), strict=True):
            torch.testing.assert_close(a, b)
        resumed.learn(1)
        assert resumed.iteration == 3
        assert (tmp_path/'second/model_3.pt').exists()
        assert resumed.export().exists()
        wrong = ThunderRunner(FakeEnv(), tmp_path/'wrong', runner_config(), {'source': 'other'})
        with pytest.raises(ValueError, match='provenance'):
            wrong.load(checkpoint)


def test_timeout_requires_terminal_value(policy):
    from algorithms.him_ppo import HIMPPO
    with contextlib.redirect_stdout(io.StringIO()):
        alg = HIMPPO(policy)
    alg.init_storage(4, 1, (265,), (274,), (16,))
    with torch.no_grad():
        alg.act(torch.zeros(4, 265), torch.zeros(4, 274))
    with pytest.raises(ValueError, match='terminal'):
        alg.process_env_step(torch.ones(4), torch.ones(4), {'time_outs': torch.ones(4)},
                             torch.zeros(4, 274), torch.zeros(4, 53), torch.zeros(4, 3))
