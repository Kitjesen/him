import torch
import torch.nn as nn
from torch.distributions import Normal
from modules.him_estimator import HIMEstimator


def get_activation(act_name: str) -> nn.Module:
    activations = {
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'relu': nn.ReLU(),
        'silu': nn.SiLU(),
        'lrelu': nn.LeakyReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }
    
    if act_name not in activations:
        print(f"Warning: Unknown activation '{act_name}', using ELU")
        return nn.ELU()
    
    return activations[act_name]


class HIMActorCritic(nn.Module): 
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_one_step_obs: int,
        num_actions: int,
        actor_hidden_dims: list = [512, 256, 128],
        critic_hidden_dims: list = [512, 256, 128],
        activation: str = 'elu',
        init_noise_std: float = 1.0,
        # Estimator params
        estimator_latent_dim: int = 16,
        estimator_hidden_dims: list = [128, 64, 16],
        estimator_target_hidden_dims: list = [128, 64],
        estimator_lr: float = 1e-3,
        num_prototype: int = 32,
        estimation_loss_weight: float = 1.0,
        swap_loss_weight: float = 1.0,
        **kwargs
    ):
        if kwargs:
            print(f"HIMActorCritic got unexpected arguments: {list(kwargs.keys())}")
        
        super(HIMActorCritic, self).__init__()
        
        activation_fn = get_activation(activation)
        
        self.history_size = int(num_actor_obs / num_one_step_obs)
        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.num_one_step_obs = num_one_step_obs
        self.estimator_latent_dim = estimator_latent_dim
        
        # ================================================================
        # Estimator: obs_history → [velocity (3D), latent (latent_dim)]
        # Expects observations in HIM order (per-timestep)
        # ================================================================
        self.estimator = HIMEstimator(
            temporal_steps=self.history_size,
            num_one_step_obs=num_one_step_obs,
            enc_hidden_dims=estimator_hidden_dims,
            tar_hidden_dims=estimator_target_hidden_dims,
            activation=activation,
            learning_rate=estimator_lr,
            num_prototype=num_prototype,
            history_len=self.history_size,
            estimation_loss_weight=estimation_loss_weight,
            swap_loss_weight=swap_loss_weight,
        )
        
        # ================================================================
        # Actor: [current_obs, vel, latent] → actions
        # ================================================================
        mlp_input_dim_a = num_one_step_obs + 3 + estimator_latent_dim
        
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        
        self.actor = nn.Sequential(*actor_layers)

        mlp_input_dim_c = num_critic_obs
        
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        
        self.critic = nn.Sequential(*critic_layers)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        
        # Disable validation for speedup
        Normal.set_default_validate_args = False
        
        # Print architecture
        print("\n✅ HIMActorCritic initialized:")
        print(f"   - History size: {self.history_size}")
        print(f"   - One-step obs dim: {num_one_step_obs}")
        print(f"   - Estimator latent dim: {estimator_latent_dim}")
        print(f"   - Actor input dim: {mlp_input_dim_a} = {num_one_step_obs}(obs) + 3(vel) + {estimator_latent_dim}(latent)")
        print(f"   - Critic input dim: {mlp_input_dim_c}")
        print(f"   - Actor hidden dims: {actor_hidden_dims}")
        print(f"   - Critic hidden dims: {critic_hidden_dims}")
        print("\n✅ HIMActorCritic architecture:")
        print(self)
    
    def reset(self, dones=None):
        """Reset hidden states (not used for non-recurrent architecture)."""
        pass
    
    def forward(self):
        """Forward pass (not implemented, use act() and evaluate() instead)."""
        raise NotImplementedError("Use act() and evaluate() methods instead")
    
    @property
    def action_mean(self):
        """Get mean of current action distribution."""
        return self.distribution.mean
    
    @property
    def action_std(self):
        """Get standard deviation of current action distribution."""
        return self.distribution.stddev
    
    @property
    def entropy(self):
        """Get entropy of current action distribution."""
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, obs_history: torch.Tensor):
        """
        Update action distribution based on observation history.
        
        Expects observations in HIM order (per-timestep): [all_obs(t-n), ..., all_obs(t-0)]
        
        Args:
            obs_history: Observation history [batch, history_size * num_one_step_obs]
        """
        # Get velocity and latent from estimator (detached)
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
        
        # Extract current observation (last frame in HIM order)
        current_obs = obs_history[:, -self.num_one_step_obs:]
        
        # Actor input: current obs + estimated vel + estimated latent
        actor_input = torch.cat((current_obs, vel, latent), dim=-1)
        
        # Compute action mean
        mean = self.actor(actor_input)
        
        # Create distribution
        self.distribution = Normal(mean, mean * 0. + self.std)
    
    def act(self, obs_history: torch.Tensor, **kwargs):

        self.update_distribution(obs_history)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions: torch.Tensor):

        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, obs_history: torch.Tensor):
        """Inference mode action selection (deterministic)."""
        # Get velocity and latent from estimator
        vel, latent = self.estimator(obs_history)
        
        # Extract current observation (last frame in HIM order)
        current_obs = obs_history[:, -self.num_one_step_obs:]
        
        # Actor input
        actor_input = torch.cat((current_obs, vel, latent), dim=-1)
        
        # Return mean action
        actions_mean = self.actor(actor_input)
        return actions_mean
    
    def evaluate(self, critic_observations: torch.Tensor, **kwargs):

        value = self.critic(critic_observations)
        return value

