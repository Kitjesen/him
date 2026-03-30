import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

from modules.him_actor_critic import HIMActorCritic
from storage.him_rollout_storage import HIMRolloutStorage


class HIMPPO:
    actor_critic: HIMActorCritic
    
    def __init__(
        self,
        actor_critic: HIMActorCritic,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        device: str = 'cpu',
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = HIMRolloutStorage.Transition()
        
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        print("\n✅ HIMPPO initialized:")
        print(f"   - Learning epochs: {num_learning_epochs}")
        print(f"   - Mini-batches: {num_mini_batches}")
        print(f"   - Clip param: {clip_param}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Device: {device}")
    
    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: tuple,
        critic_obs_shape: tuple,
        action_shape: tuple
    ):
        """Initialize rollout storage."""
        self.storage = HIMRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.device
        )
    
    def test_mode(self):
        """Set to evaluation mode."""
        self.actor_critic.eval()
    
    def train_mode(self):
        """Set to training mode."""
        self.actor_critic.train()
    
    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor):
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
        next_critic_obs: torch.Tensor
    ):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device),
                1
            )
        
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs: torch.Tensor):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
    
    def update(self) -> Tuple[float, float, float, float]:
        """Returns: (value_loss, surrogate_loss, estimation_loss, swap_loss)"""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0
        
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs
        )
        
        for (obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch,
             target_values_batch, advantages_batch, returns_batch,
             old_actions_log_prob_batch, old_mu_batch, old_sigma_batch) in generator:
            
            # Update Estimator (SwAV)
            # Following HIMLoco: use next timestep (t+1) velocity as supervision
            # This allows the model to learn to predict "what will be the velocity after executing the action"
            # NOTE: Do not pass PPO learning rate to estimator - it has its own learning rate
            # Only pass lr if we want to schedule estimator learning rate separately
            estimation_loss, swap_loss = self.actor_critic.estimator.update(
                obs_history=obs_batch,
                next_critic_obs=next_critic_obs_batch,
                lr=None,  # Use estimator's own learning rate, don't override with PPO's
            )
            
            # Update Actor and Critic (PPO)
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            
            # Adaptive learning rate
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) +
                        (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) /
                        (2.0 * torch.square(sigma_batch)) - 0.5,
                        axis=-1
                    )
                    kl_mean = torch.mean(kl)
                    
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate
            
            # PPO surrogate loss (clipped)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            
            # Value function loss (clipped)
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            
            loss = (surrogate_loss +
                    self.value_loss_coef * value_loss -
                    self.entropy_coef * entropy_batch.mean())
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimation_loss += estimation_loss
            mean_swap_loss += swap_loss
        
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        
        self.storage.clear()
        
        return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss

