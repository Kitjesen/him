import torch
from typing import List


def reshape_isaac_to_him(
    obs_flat: torch.Tensor,
    history_len: int,
    obs_dims: List[int]
) -> torch.Tensor:
    """
    Convert Isaac Lab observation order to HIM order.
    
    Isaac Lab: [var1(t-4...t-0), var2(t-4...t-0), ...]
    HIM: [all_vars(t-4), all_vars(t-3), ..., all_vars(t-0)]
    
    Args:
        obs_flat: Isaac Lab flattened observation [batch, total_dims]
        history_len: History length
        obs_dims: Dimension list for each variable
    
    Returns:
        Reordered observation [batch, total_dims] in HIM format
    """
    batch_size = obs_flat.shape[0]
    
    # Extract each variable's history
    var_histories = []
    offset = 0
    for var_dim in obs_dims:
        var_flat = obs_flat[:, offset:offset + var_dim * history_len]
        var_history = var_flat.reshape(batch_size, history_len, var_dim)
        var_histories.append(var_history)
        offset += var_dim * history_len
    
    # Reorganize by timestep
    timesteps = []
    for t in range(history_len):
        timestep_vars = [var_hist[:, t, :] for var_hist in var_histories]
        timestep_obs = torch.cat(timestep_vars, dim=-1)
        timesteps.append(timestep_obs)
    
    obs_him = torch.cat(timesteps, dim=-1)
    
    return obs_him


def extract_current_velocity_isaac(
    critic_obs_flat: torch.Tensor,
    history_len: int,
    vel_dim: int = 3
) -> torch.Tensor:
    """
    Extract current velocity from Isaac Lab critic observation.
    
    Args:
        critic_obs_flat: Critic flattened observation [batch, critic_dims]
        history_len: History length
        vel_dim: Velocity dimension (default 3)
    
    Returns:
        Current velocity [batch, vel_dim]
    """
    batch_size = critic_obs_flat.shape[0]
    vel_flat = critic_obs_flat[:, :vel_dim * history_len]
    vel_history = vel_flat.reshape(batch_size, history_len, vel_dim)
    vel_current = vel_history[:, -1, :]
    
    return vel_current


# Thunder_Hist observation configuration
THUNDER_HIST_POLICY_DIMS = [3, 3, 3, 16, 16, 16]  # [ang_vel, gravity, cmd, jpos, jvel, act]
THUNDER_HIST_HISTORY_LEN = 5


if __name__ == "__main__":
    print("=" * 80)
    print("Observation Reshaping Test")
    print("=" * 80)
    
    batch_size = 2
    history_len = 5
    obs_dims = [3, 3, 3]
    total_dims = sum(obs_dims) * history_len
    
    obs_isaac = torch.zeros(batch_size, total_dims)
    
    offset = 0
    for var_idx, var_dim in enumerate(obs_dims):
        for t in range(history_len):
            for d in range(var_dim):
                value = var_idx * 100 + t * 10 + d
                obs_isaac[:, offset + t * var_dim + d] = value
        offset += var_dim * history_len
    
    print(f"\n📥 Isaac Lab Order (by variable):")
    print(f"Shape: {obs_isaac.shape}")
    for i in range(0, 45, 3):
        print(f"  [{i:2d}:{i+3:2d}]: {obs_isaac[0, i:i+3].tolist()}")
    
    obs_him = reshape_isaac_to_him(obs_isaac, history_len, obs_dims)
    
    print(f"\n📤 HIM Order (by timestep):")
    print(f"Shape: {obs_him.shape}")
    for i in range(0, 45, 9):
        print(f"  [{i:2d}:{i+9:2d}]: {obs_him[0, i:i+9].tolist()}")
    
    # Extract current frame from HIM format (simple slice)
    num_one_step_obs = sum(obs_dims)
    current = obs_him[:, -num_one_step_obs:]
    
    print(f"\n🎯 Current Frame from HIM (simple slice):")
    print(f"Shape: {current.shape}")
    print(f"Data[0, :]: {current[0, :].tolist()}")
    print(f"Expected: [4.0, 5.0, 6.0, 104.0, 105.0, 106.0, 204.0, 205.0, 206.0]")
    print(f"\n✅ Just use: obs_history[:, -{num_one_step_obs}:] to get current frame!")

