from typing import Dict

import chex
import jax.numpy as jnp

from core.evaluators.mcts.state import MCTSTree


def normalize_q_values(
    q_values: chex.Array, 
    child_n_values: chex.Array, 
    parent_q_value: float,
    epsilon: float
) -> chex.Array:
    """Normalize Q-values to be in the range [0, 1].
    
    Args:
    - `q_values`: Q-values to normalize
    - `child_n_values`: visit counts of child nodes
    - `parent_q_value`: Q-value of the parent node
    - `epsilon`: small value to avoid division by zero

    Returns:
    - (chex.Array): normalized Q-values
    """
    min_value = jnp.minimum(parent_q_value, jnp.min(q_values, axis=-1))
    max_value = jnp.maximum(parent_q_value, jnp.max(q_values, axis=-1))
    completed_by_min = jnp.where(child_n_values > 0, q_values, min_value)
    normalized = (completed_by_min - min_value) / (
        jnp.maximum(max_value - min_value, epsilon))
    return normalized


class MCTSActionSelector:
    """Base class for action selection in MCTS.
    
    Is callable, selects an action given a search tree state.
    """

    def __init__(self, epsilon: float = 1e-8): 
        """
        Args:
        - `epsilon`: small value to avoid division by zero
        """
        self.epsilon = epsilon


    def __call__(self, tree: MCTSTree, index: int) -> int:
        """Selects an action given a search tree state. Implemented by subclasses."""
        raise NotImplementedError()


    def get_config(self) -> Dict:
        """Returns the configuration of the action selector. Used for logging."""
        return {
            "epsilon": self.epsilon
        }


class PUCTSelector(MCTSActionSelector):
    """PUCT (Polynomial Upper Confidence Trees) action selector.
    
    This is the algorithm used for action selection within AlphaZero."""

    def __init__(self, 
        c: float = 1.0,
        epsilon: float = 1e-8, 
        q_transform = normalize_q_values
    ):
        """
        Args:
        - `c`: exploration constant (larger values encourage exploration)
        - `epsilon`: small value to avoid division by zero
        - `q_transform`: function applied to q-values before selection
        """
        super().__init__(epsilon=epsilon)
        self.c = c
        self.q_transform = q_transform


    def get_config(self) -> Dict:
        """Returns the configuration of the PUCT action selector. Used for logging."""
        return {
            "c": self.c,
            'q_transform': self.q_transform.__name__,
            **super().get_config()
        }


    def __call__(self, tree: MCTSTree, index: int) -> int:
        """Selects an action given a search tree state.

        Args:
        - `tree`: search tree
        - `index`: index of the node in the search tree to select an action to take from

        Returns:
        - (int): id of action to take
        """
        # get parent node and its current player
        node = tree.data_at(index)
        parent_player = node.embedding.current_player

        # get child q-values and visit counts
        q_values = tree.get_child_data('q', index)
        n_values = tree.get_child_data('n', index)

        # get child node indices and their current players
        child_indices = tree.edge_map[index]
        # For valid children, get their current_player; for NULL_INDEX use parent_player (no flip)
        safe_indices = jnp.maximum(child_indices, 0)
        child_players = tree.data.embedding.current_player[safe_indices]
        # Mask: only apply player comparison for existing children
        child_exists = child_indices != tree.NULL_INDEX

        # Calculate per-child discount: 1.0 if same player, -1.0 if different
        player_diff = jnp.abs(parent_player - child_players)
        per_child_discount = 1.0 - 2.0 * player_diff
        # For non-existent children (n=0, q=0), discount doesn't matter, but use 1.0 to be safe
        per_child_discount = jnp.where(child_exists, per_child_discount, 1.0)

        # Apply per-child discount to convert Q-values to parent's perspective
        adjusted_q_values = q_values * per_child_discount

        # normalize/transform q-values
        q_values = self.q_transform(adjusted_q_values, n_values, node.q, self.epsilon)
        # calculate U-values
        u_values = self.c * node.p * jnp.sqrt(node.n) / (n_values + 1)
        # PUCT = Q-value + U-value
        puct_values = q_values + u_values
        
        # === FIX: Apply legal action mask ===
        # Get the legal action mask from the node's embedding
        legal_action_mask = node.embedding.legal_action_mask
        # Mask out illegal actions by setting their PUCT value to negative infinity
        masked_puct_values = jnp.where(legal_action_mask, puct_values, jnp.finfo(puct_values.dtype).min)
        # === END FIX ===
        
        # select action with highest PUCT value (from legal actions)
        return masked_puct_values.argmax()
    
    

class MuZeroPUCTSelector(MCTSActionSelector):
    """Implements the variant of PUCT used in MuZero."""

    def __init__(self, 
        c1: float = 1.25, 
        c2: float = 19652, 
        epsilon: float = 1e-8,
        q_transform = normalize_q_values
    ):
        """
        Args:
        - `c1`: 1st exploration constant
        - `c2`: 2nd exploration constant
        - `epsilon`: small value to avoid division by zero
        - `q_transform`: function applied to q-values before selection
        """
        super().__init__(epsilon=epsilon)
        self.c1 = c1
        self.c2 = c2
        self.q_transform = q_transform
    

    def get_config(self) -> Dict:
        """Returns the configuration of the MuZero PUCT action selector. Used for logging."""
        return {
            "c1": self.c1,
            "c2": self.c2,
            "q_transform": self.q_transform.__name__,
            **super().get_config()
        }

    def __call__(self, tree: MCTSTree, index: int) -> int:
        """Selects an action given a search tree state.

        Args:
        - `tree`: search tree
        - `index`: index of the node in the search tree to select an action to take from

        Returns:
        - (int): id of action to take
        """
        # get parent node and its current player
        node = tree.data_at(index)
        parent_player = node.embedding.current_player

        # get child q-values and visit counts
        q_values = tree.get_child_data('q', index)
        n_values = tree.get_child_data('n', index)

        # get child node indices and their current players
        child_indices = tree.edge_map[index]
        safe_indices = jnp.maximum(child_indices, 0)
        child_players = tree.data.embedding.current_player[safe_indices]
        child_exists = child_indices != tree.NULL_INDEX

        # Calculate per-child discount: 1.0 if same player, -1.0 if different
        player_diff = jnp.abs(parent_player - child_players)
        per_child_discount = 1.0 - 2.0 * player_diff
        per_child_discount = jnp.where(child_exists, per_child_discount, 1.0)

        # Apply per-child discount to convert Q-values to parent's perspective
        adjusted_q_values = q_values * per_child_discount

        # normalize/transform q-values (passing adjusted and raw q_values for MuZero compat)
        q_values = self.q_transform(adjusted_q_values, q_values, n_values, node.q, self.epsilon)
        # calculate U-values
        base_term = node.p * jnp.sqrt(node.n) / (n_values + 1)
        log_term = jnp.log((node.n + self.c2 + 1) / self.c2) + self.c1
        u_values = base_term * log_term
        # PUCT = Q-value + U-value
        puct_values = q_values + u_values
        
        # === FIX: Apply legal action mask ===
        # Get the legal action mask from the node's embedding
        legal_action_mask = node.embedding.legal_action_mask
        # Mask out illegal actions by setting their PUCT value to negative infinity
        masked_puct_values = jnp.where(legal_action_mask, puct_values, jnp.finfo(puct_values.dtype).min)
        # === END FIX ===
        
        # select action with highest PUCT value (from legal actions)
        return masked_puct_values.argmax()
