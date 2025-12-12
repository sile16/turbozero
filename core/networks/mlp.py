import flax.linen as nn
import jax.numpy as jnp
import chex
from typing import Sequence, Tuple

@chex.dataclass
class MLPConfig:
    """Configuration for the MLP network."""
    hidden_dims: Sequence[int]
    policy_head_out_size: int
    value_head_out_size: int = 4  # 4-way conditional head by default

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for policy-value tasks."""
    config: MLPConfig

    @nn.compact
    def __call__(self, x: chex.Array, train: bool) -> Tuple[chex.Array, chex.Array]: # Added Tuple import implicitly needed
        """Forward pass through the MLP.

        Args:
            x: Input observation (flattened).
            train: Boolean indicating if the model is in training mode (unused here, but kept for interface compatibility).

        Returns:
            Tuple containing (policy_logits, value head output).
        """
        # Ensure input is flattened and has a batch dimension
        if x.ndim == 1:
            x = x[None, ...] # Add batch dim if missing
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1)) 

        # Hidden layers
        for dim in self.config.hidden_dims:
            x = nn.Dense(features=dim)(x)
            x = nn.relu(x)

        # Policy head
        policy_logits = nn.Dense(features=self.config.policy_head_out_size, name='policy_head')(x)

        # Value head
        value = nn.Dense(features=self.config.value_head_out_size, name='value_head')(x)
        # Squeeze the value output if it's size 1
        if self.config.value_head_out_size == 1:
            value = jnp.squeeze(value, axis=-1)

        return policy_logits, value 
