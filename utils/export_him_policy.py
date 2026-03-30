# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""
Export HIM policy for deployment.

This module provides functions to export HIM (History-based Implicit Model) policies
for deployment, similar to HIMLoco's export functionality but adapted for Isaac Lab.
"""

import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PolicyExporterHIM(nn.Module):
    """
    Exporter for HIM policy deployment.

    This class creates a lightweight inference model containing:
    - Actor network
    - Estimator encoder (for velocity and latent prediction)

    The exported model can be used for deployment without requiring the full training setup.
    """

    def __init__(self, actor_critic):
        """
        Initialize the exporter.

        Args:
            actor_critic: HIMActorCritic instance containing actor and estimator
        """
        super().__init__()
        # Copy actor network
        self.actor = copy.deepcopy(actor_critic.actor)

        # Copy estimator encoder (only encoder is needed for inference)
        # The encoder takes history and outputs [velocity, latent]
        self.estimator_encoder = copy.deepcopy(actor_critic.estimator.encoder)

        # Store dimensions for forward pass
        self.num_one_step_obs = actor_critic.num_one_step_obs
        self.estimator_latent_dim = actor_critic.estimator_latent_dim

        print("\n✅ PolicyExporterHIM initialized:")
        print(f"   - Actor input dim: {actor_critic.num_one_step_obs + 3 + actor_critic.estimator_latent_dim}")
        print(f"   - Estimator encoder: {actor_critic.estimator.encoder}")
        print(f"   - One-step obs dim: {self.num_one_step_obs}")
        print(f"   - Latent dim: {self.estimator_latent_dim}")

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference.

        Args:
            obs_history: Observation history in HIM order [batch, history_len * num_one_step_obs]
                        Format: [all_vars(t-n), ..., all_vars(t-0)]

        Returns:
            actions: Predicted actions [batch, num_actions]
        """
        # Encode history to get velocity and latent
        parts = self.estimator_encoder(obs_history)
        vel = parts[..., :3]  # Extract velocity (3D)
        z = parts[..., 3:]  # Extract latent features

        # Normalize latent (L2 normalization)
        z = F.normalize(z, dim=-1, p=2)

        # Extract current observation (last frame in HIM order)
        current_obs = obs_history[:, -self.num_one_step_obs :]

        # Concatenate: [current_obs, vel, latent]
        actor_input = torch.cat((current_obs, vel, z), dim=-1)

        # Get action from actor
        actions = self.actor(actor_input)

        return actions

    def export(self, path: str, filename: str = "policy.pt"):
        """
        Export the model as TorchScript JIT module.

        Args:
            path: Directory path to save the exported model
            filename: Filename for the exported model (default: "policy.pt")
        """
        os.makedirs(path, exist_ok=True)
        export_path = os.path.join(path, filename)

        # Move to CPU for export
        self.to("cpu")
        self.eval()

        # Create TorchScript module
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(export_path)

        print(f"✅ Exported HIM policy to: {export_path}")
        print(f"   - Model size: {os.path.getsize(export_path) / 1024 / 1024:.2f} MB")


def export_him_policy_as_jit(actor_critic, path: str, filename: str = "policy.pt") -> None:
    """
    Export HIM policy as TorchScript JIT module.

    This function creates a lightweight inference model containing only the necessary
    components for deployment (actor + estimator encoder).

    Args:
        actor_critic: HIMActorCritic instance
        path: Directory path to save the exported model
        filename: Filename for the exported model (default: "policy.pt")
    """
    if not hasattr(actor_critic, "estimator"):
        raise ValueError("actor_critic must have an estimator attribute (HIMActorCritic)")

    exporter = PolicyExporterHIM(actor_critic)
    exporter.export(path, filename)


def export_him_policy_as_onnx(
    actor_critic, path: str, filename: str = "policy.onnx", input_shape: Optional[tuple] = None
) -> None:
    """
    Export HIM policy as ONNX model.

    Args:
        actor_critic: HIMActorCritic instance
        path: Directory path to save the exported model
        filename: Filename for the exported model (default: "policy.onnx")
        input_shape: Input shape for the model [batch_size, history_len * num_one_step_obs]
                    If None, uses [1, actor_critic.num_actor_obs]
    """
    if not hasattr(actor_critic, "estimator"):
        raise ValueError("actor_critic must have an estimator attribute (HIMActorCritic)")

    exporter = PolicyExporterHIM(actor_critic)
    exporter.to("cpu")
    exporter.eval()

    # Create dummy input
    if input_shape is None:
        input_shape = (1, actor_critic.num_actor_obs)

    dummy_input = torch.randn(input_shape)

    os.makedirs(path, exist_ok=True)
    export_path = os.path.join(path, filename)

    # Export to ONNX
    torch.onnx.export(
        exporter,
        dummy_input,
        export_path,
        input_names=["obs_history"],
        output_names=["actions"],
        dynamic_axes={"obs_history": {0: "batch_size"}, "actions": {0: "batch_size"}},
        opset_version=11,
    )

    print(f"✅ Exported HIM policy to ONNX: {export_path}")
    print(f"   - Model size: {os.path.getsize(export_path) / 1024 / 1024:.2f} MB")
