
from typing import Dict

import chex
import jax 
import jax.numpy as jnp

from core.evaluators.mcts.state import MCTSTree
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.mcts.equity import normalize_value_probs_4way, terminal_value_probs_from_reward_4way
from core.types import StepMetadata


class _AlphaZero:
    """AlphaZero-specific logic for MCTS.
    Extends MCTS using the `AlphaZero` class, this class serves as a mixin to add AlphaZero-specific logic.
    """

    def __init__(self,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        **kwargs
    ):
        """
        Args:
        - `dirichlet_alpha`: magnitude of Dirichlet noise.
        - `dirichlet_epsilon`: proportion of root policy composed of Dirichlet noise.
        (see `MCTS` class for additional configuration)
        """
        super().__init__(**kwargs)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon


    def get_config(self) -> Dict:
        """Returns the configuration of the AlphaZero evaluator. Used for logging."""
        return {
            "dirichlet_alpha": self.dirichlet_alpha,
            "dirichlet_epsilon": self.dirichlet_epsilon,
            **super().get_config() #pylint: disable=no-member
        }


    def update_root(self, key: chex.PRNGKey, tree: MCTSTree, root_embedding: chex.ArrayTree, params: chex.ArrayTree, root_metadata: StepMetadata) -> MCTSTree:
        """Populates the root node of the search tree. Adds Dirichlet noise to the root policy.
        
        Args:
        - `key`: rng
        - `tree`: The search tree.
        - `root_embedding`: root environment state.
        - `params`: nn parameters.
        - `root_metadata`: metadata of the root environment state
        
        Returns:
        - `tree`: The updated search tree.
        """
        # evaluate the root state 
        root_key, dir_key = jax.random.split(key, 2)
        root_policy_logits, root_value_probs = self.eval_fn(root_embedding, params, root_key) #pylint: disable=no-member
        masked_logits = jnp.where(root_metadata.action_mask, root_policy_logits, jnp.finfo(root_policy_logits).min)
        root_policy = jax.nn.softmax(masked_logits)
        # 4-way value head
        normalized_value_probs = normalize_value_probs_4way(root_value_probs)
        normalized_value_probs = jax.lax.cond(
            root_metadata.terminated,
            lambda: terminal_value_probs_from_reward_4way(root_metadata.rewards[root_metadata.cur_player_id]),
            lambda: normalized_value_probs
        )
        root_value = self.value_equity_fn(normalized_value_probs, root_metadata) #pylint: disable=no-member

        # add Dirichlet noise to the root policy
        dirichlet_noise = jax.random.dirichlet(
            dir_key,
            alpha=jnp.full(
                [tree.branching_factor], 
                fill_value=self.dirichlet_alpha
            )
        )
        noisy_policy = (
            ((1-self.dirichlet_epsilon) * root_policy) +
            (self.dirichlet_epsilon * dirichlet_noise)
        )
        # re-normalize the policy
        new_logits = jnp.log(jnp.maximum(noisy_policy, jnp.finfo(noisy_policy).tiny))
        policy = jnp.where(root_metadata.action_mask, new_logits, jnp.finfo(noisy_policy).min)
        renorm_policy = jax.nn.softmax(policy)

        # update the root node
        # NOTE: The caller (MCTS.evaluate) already determines when update_root should be called
        # based on persist_tree and root_n. We unconditionally update here to honor the caller's
        # intent - this ensures Dirichlet noise is refreshed when persist_tree=False.
        root_node = tree.data_at(tree.ROOT_INDEX)
        updated_root = self._force_update_root_node(root_node, renorm_policy, root_value, normalized_value_probs, root_embedding)
        return tree.set_root(updated_root)

    @staticmethod
    def _force_update_root_node(root_node, root_policy, root_value, root_value_probs, root_embedding):
        """Force update root node, ignoring visit count.

        Unlike MCTS.update_root_node which preserves values if the node was visited,
        this method always updates the node with new values. This is needed for
        persist_tree=False semantics where each evaluate() should start fresh.
        """
        return root_node.replace(
            p=root_policy,
            q=root_value,
            value_probs=root_value_probs,
            n=jnp.array(1, dtype=jnp.int32),  # Reset to 1 visit
            embedding=root_embedding
        )
    

class AlphaZero(MCTS):
    """AlphaZero: Monte Carlo Tree Search + Neural Network Leaf Evaluation
     - https://arxiv.org/abs/1712.01815

    Most of the work is actually done in the `MCTS` class, which AlphaZero extends.
    This class can take an arbitrary MCTS backend, which is why we use a separate class `_AlphaZero`
    to handle the AlphaZero-specific logic, then combine them here.
    """

    def __new__(cls, base_type: type = MCTS):
        """Creates a new AlphaZero class that extends the given MCTS class."""
        assert issubclass(base_type, MCTS)
        cls_type = type("AlphaZero", (_AlphaZero, base_type), {})
        cls_type.__name__ = f'AlphaZero({base_type.__name__})'
        return cls_type
