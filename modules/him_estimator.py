import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple

from utils.observation_reshaper import extract_current_velocity_isaac


class HIMEstimator(nn.Module):
    def __init__(
        self,
        temporal_steps: int,
        num_one_step_obs: int,
        enc_hidden_dims: list = [128, 64, 16],
        tar_hidden_dims: list = [128, 64],
        activation: str = 'elu',
        learning_rate: float = 1e-3,
        max_grad_norm: float = 10.0,
        num_prototype: int = 32,
        temperature: float = 3.0,
        history_len: int = 1,
        estimation_loss_weight: float = 1.0,
        swap_loss_weight: float = 1.0,
        **kwargs
    ):
        if kwargs:
            print(f"HIMEstimator.__init__ got unexpected arguments: {list(kwargs.keys())}")
        
        super(HIMEstimator, self).__init__()
        
        activation_fn = get_activation(activation)
        
        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self.num_prototype = num_prototype
        self.history_len = history_len
        self.estimation_loss_weight = estimation_loss_weight
        self.swap_loss_weight = swap_loss_weight
        
        # ================================================================
        # Encoder: Flatten history → [velocity, latent]
        # ================================================================
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation_fn]
            enc_input_dim = enc_hidden_dims[l]
        
        # Output: velocity (3D) + latent (num_latent)
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + 3)]
        self.encoder = nn.Sequential(*enc_layers)
        
        # ================================================================
        # Target: Current observation → latent
        # ================================================================
        tar_input_dim = self.num_one_step_obs
        tar_layers = []
        
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation_fn]
            tar_input_dim = tar_hidden_dims[l]
        
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)
        
        # ================================================================
        # Prototype: Learnable cluster centers for SwAV
        # ================================================================
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])
        
        # ================================================================
        # Optimizer
        # ================================================================
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Environment reference for direct velocity extraction (optional)
        self.env = None
        
    
    def set_env_reference(self, env):
        """
        Set environment reference for direct velocity extraction.
        
        Args:
            env: Environment instance (wrapped with RslRlVecEnvWrapper)
        """
        self.env = env
        print("✅ HIMEstimator: Environment reference set for direct velocity extraction")
    
    def get_latent(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get velocity and latent predictions (detached for inference).
        
        Args:
            obs_history: [batch, temporal_steps * num_one_step_obs]
        
        Returns:
            velocity: [batch, 3]
            latent: [batch, num_latent]
        """
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()
    
    def forward(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (with detach for actor usage).
        
        Args:
            obs_history: [batch, temporal_steps * num_one_step_obs]
        
        Returns:
            velocity: [batch, 3] (detached)
            latent: [batch, num_latent] (detached, normalized)
        """
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel.detach(), z.detach()
    
    def encode(self, obs_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observation history (without detach, for training).
        
        Args:
            obs_history: [batch, temporal_steps * num_one_step_obs]
        
        Returns:
            velocity: [batch, 3]
            latent: [batch, num_latent] (normalized)
        """
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2)
        return vel, z
    
    def update(
        self,
        obs_history: torch.Tensor,
        next_critic_obs: torch.Tensor,
        lr: float = None
    ) -> Tuple[float, float]:
        """
        Update estimator using SwAV contrastive learning.
        
        Training procedure:
        1. Extract ground truth velocity from next critic observation (timestep t+1)
        2. Encode history → predicted velocity + source latent
        3. Encode current obs → target latent
        4. Normalize latents and prototypes
        5. Compute SwAV loss with Sinkhorn-Knopp assignment
        6. Compute velocity estimation loss (MSE)
        7. Backprop and optimize
        
        Note: Following HIMLoco, we use next timestep (t+1) velocity as supervision.
        This allows the model to learn to predict "what will be the velocity after 
        executing the action", which is more useful for control.
        
        Args:
            obs_history: Historical observations in HIM order [batch, temporal_steps * num_one_step_obs]
                       Last timestep is t, so we predict velocity at timestep t+1 (after action execution)
            next_critic_obs: Next critic observation (timestep t+1) with history [t-3, t-2, t-1, t, t+1]
                           Used to extract ground truth velocity at timestep t+1
            lr: Optional learning rate override (for learning rate scheduling)
        
        Returns:
            estimation_loss: Velocity prediction MSE loss
            swap_loss: SwAV contrastive loss
        """
        # Update learning rate if provided (for learning rate scheduling)
        # NOTE: Only update if explicitly provided, otherwise use estimator's own learning rate
        if lr is not None:
            old_lr = self.learning_rate
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            # Debug: print if learning rate changed significantly
            if abs(old_lr - lr) > 1e-6:
                print(f"[DEBUG] Estimator LR changed: {old_lr:.6f} -> {lr:.6f}")
        
        # ================================================================
        # 1. Extract ground truth velocity from next critic observation (timestep t+1)
        # ================================================================
        # obs_history contains history [t-4, t-3, t-2, t-1, t], we predict velocity at timestep t+1
        # next_critic_obs contains history [t-3, t-2, t-1, t, t+1], we extract velocity at timestep t+1
        # This follows HIMLoco's design: predict "what will be the velocity after executing the action"
        vel = extract_current_velocity_isaac(next_critic_obs, self.history_len).detach()
        
        # ================================================================
        # 2. Extract current observation (for target encoder)
        # ================================================================
        # Observations are in HIM order (per-timestep), so current frame is at the end
        next_obs = obs_history[:, -self.num_one_step_obs:].detach()
        
        # ================================================================
        # 3. Encode history and current observation
        # ================================================================
        # IMPORTANT: Do not use encode() method here as it detaches obs_history
        # We need gradients to flow back through obs_history for training
        z_s = self.encoder(obs_history)  # Direct encoder call, no detach
        z_t = self.target(next_obs)
        
        pred_vel, z_s = z_s[..., :3], z_s[..., 3:]
        
        # Normalize latents (L2 normalization)
        z_s = F.normalize(z_s, dim=-1, p=2)
        z_t = F.normalize(z_t, dim=-1, p=2)
        
        # ================================================================
        # 4. Normalize prototypes (in-place, no gradient)
        # ================================================================
        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)
        
        # ================================================================
        # 5. Compute similarity scores (logits)
        # ================================================================
        score_s = z_s @ self.proto.weight.T  # [batch, num_prototype]
        score_t = z_t @ self.proto.weight.T  # [batch, num_prototype]
        
        # ================================================================
        # 6. Sinkhorn-Knopp: Generate soft assignments (pseudo-labels)
        # ================================================================
        with torch.no_grad():
            q_s = sinkhorn(score_s)  # [batch, num_prototype]
            q_t = sinkhorn(score_t)  # [batch, num_prototype]
        
        # ================================================================
        # 7. Compute SwAV loss (swapped prediction)
        # ================================================================
        # Use assignment from one view to predict the other
        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)
        
        # Swap: predict t using s's assignment, predict s using t's assignment
        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        
        # ================================================================
        # 8. Compute velocity estimation loss
        # ================================================================
        estimation_loss = F.mse_loss(pred_vel, vel)
        
        # ================================================================
        # 9. Total loss and optimization (with weights)
        # ================================================================
        total_loss = self.estimation_loss_weight * estimation_loss + self.swap_loss_weight * swap_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Return raw losses (before weighting) for logging
        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm for optimal transport.
    
    Generates balanced cluster assignments by iteratively normalizing
    rows and columns to satisfy marginal constraints.
    
    Args:
        out: Similarity scores [batch, num_prototype]
        eps: Temperature for exponential
        iters: Number of iterations
    
    Returns:
        Q: Normalized assignment matrix [batch, num_prototype]
    """
    Q = torch.exp(out / eps).T  # [num_prototype, batch]
    K, B = Q.shape  # K=num_prototype, B=batch_size
    Q /= Q.sum()
    
    for _ in range(iters):
        # Normalize rows: each prototype gets equal total weight (1/K)
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        
        # Normalize columns: each sample gets equal total weight (1/B)
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    
    return (Q * B).T  # [batch, num_prototype]


def get_activation(act_name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'relu': nn.ReLU(),
        'crelu': nn.ReLU(),
        'silu': nn.SiLU(),
        'lrelu': nn.LeakyReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }
    
    if act_name not in activations:
        print(f"Warning: Unknown activation '{act_name}', using ELU")
        return nn.ELU()
    
    return activations[act_name]

