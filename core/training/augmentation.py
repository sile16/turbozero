"""Data augmentation transforms for training.

Includes symmetry augmentation for games with rotational/reflective symmetries.

Transform functions have the signature:
    transform_fn(action_mask, policy_weights, env_state) ->
        (transformed_action_mask, transformed_policy_weights, transformed_env_state)
"""

from typing import Callable, List
import jax
import jax.numpy as jnp
import chex


def make_2048_symmetry_transforms() -> List[Callable]:
    """Creates 7 data transform functions for 2048 symmetry augmentation.

    2048 has 8 symmetries (D4 group):
    - 4 rotations (0°, 90°, 180°, 270°)
    - Each rotation can be combined with horizontal flip

    We return 7 transforms (excluding identity) since the original is already stored.

    Actions in 2048: 0=up, 1=right, 2=down, 3=left

    Returns:
        List of 7 transform functions, each applying a different symmetry.
    """

    # Action transformation matrices for each symmetry
    # Original: 0=up, 1=right, 2=down, 3=left
    # After rot90: up->right, right->down, down->left, left->up
    # So action i becomes (i+1)%4

    # Action mappings for each of the 8 symmetries
    # [sym_id] -> original_action -> new_action
    # For policy: we need inverse mapping - new policy at position j gets old policy at inverse[j]
    action_inverse_transforms = jnp.array([
        [0, 1, 2, 3],  # identity (sym 0)
        [3, 0, 1, 2],  # rot90 (sym 1) - inverse of [1,2,3,0]
        [2, 3, 0, 1],  # rot180 (sym 2)
        [1, 2, 3, 0],  # rot270 (sym 3) - inverse of [3,0,1,2]
        [0, 3, 2, 1],  # flip_h (sym 4)
        [1, 0, 3, 2],  # rot90 + flip_h (sym 5)
        [2, 1, 0, 3],  # rot180 + flip_h (sym 6)
        [3, 2, 1, 0],  # rot270 + flip_h (sym 7)
    ], dtype=jnp.int32)

    def make_transform(sym_id: int) -> Callable:
        """Create a transform function for a specific symmetry."""

        def transform(
            action_mask: chex.Array,
            policy_weights: chex.Array,
            env_state: chex.ArrayTree
        ):
            """Transform action_mask, policy_weights, and env_state.

            Args:
                action_mask: Shape (4,) legal action mask
                policy_weights: Shape (4,) policy weights
                env_state: Environment state containing observation

            Returns:
                Transformed (action_mask, policy_weights, env_state)
            """
            # Get inverse mapping for this symmetry
            inv_map = action_inverse_transforms[sym_id]

            # Transform policy weights: new_policy[i] = old_policy[inv_map[i]]
            new_policy_weights = policy_weights[inv_map]

            # Transform action mask the same way
            new_action_mask = action_mask[inv_map]

            # Transform observation in env_state
            # The observation is at env_state.observation with shape (4, 4, C)
            obs = env_state.observation

            # Apply rotation (0, 1, 2, or 3 times)
            rot_count = sym_id % 4
            new_obs = jnp.rot90(obs, k=rot_count, axes=(0, 1))

            # Apply horizontal flip if sym_id >= 4
            new_obs = jax.lax.cond(
                sym_id >= 4,
                lambda x: jnp.flip(x, axis=1),
                lambda x: x,
                new_obs
            )

            # Create new env_state with transformed observation
            # pgx uses .replace() instead of ._replace()
            new_env_state = env_state.replace(observation=new_obs)

            return new_action_mask, new_policy_weights, new_env_state

        return transform

    # Return transforms for symmetries 1-7 (exclude identity 0)
    transforms = [make_transform(i) for i in range(1, 8)]

    return transforms


def make_single_2048_symmetry_transform(sym_id: int) -> Callable:
    """Creates a single transform function for a specific 2048 symmetry.

    Useful for testing or when you want to apply a specific symmetry.

    Args:
        sym_id: Symmetry id (0-7)
            0: identity
            1: rot90
            2: rot180
            3: rot270
            4: flip_h
            5: rot90 + flip_h
            6: rot180 + flip_h
            7: rot270 + flip_h

    Returns:
        Transform function for the specified symmetry.
    """
    transforms = [None] + make_2048_symmetry_transforms()  # Add identity placeholder
    if sym_id == 0:
        # Identity transform
        def identity_transform(action_mask, policy_weights, env_state):
            return action_mask, policy_weights, env_state
        return identity_transform
    else:
        return transforms[sym_id]


# Standalone transform functions for observation and policy (for testing)
def transform_2048_observation(obs: chex.Array, sym_id: int) -> chex.Array:
    """Transform 2048 observation based on symmetry id.

    Args:
        obs: Observation of shape (4, 4, C)
        sym_id: Symmetry id (0-7)

    Returns:
        Transformed observation with same shape
    """
    # Apply rotation (0, 1, 2, or 3 times)
    rot_count = sym_id % 4
    new_obs = jnp.rot90(obs, k=rot_count, axes=(0, 1))

    # Apply horizontal flip if sym_id >= 4
    new_obs = jax.lax.cond(
        sym_id >= 4,
        lambda x: jnp.flip(x, axis=1),
        lambda x: x,
        new_obs
    )

    return new_obs


def transform_2048_policy(policy: chex.Array, sym_id: int) -> chex.Array:
    """Transform 2048 policy based on symmetry id.

    Args:
        policy: Policy of shape (4,)
        sym_id: Symmetry id (0-7)

    Returns:
        Transformed policy with same shape
    """
    # Inverse action mappings
    action_inverse_transforms = jnp.array([
        [0, 1, 2, 3],  # identity
        [3, 0, 1, 2],  # rot90
        [2, 3, 0, 1],  # rot180
        [1, 2, 3, 0],  # rot270
        [0, 3, 2, 1],  # flip_h
        [1, 0, 3, 2],  # rot90 + flip_h
        [2, 1, 0, 3],  # rot180 + flip_h
        [3, 2, 1, 0],  # rot270 + flip_h
    ], dtype=jnp.int32)

    inv_map = action_inverse_transforms[sym_id]
    return policy[inv_map]
