"""Gumbel StochasticMCTS - StochasticMCTS with Gumbel-Top-k root action selection.

Combines:
- Gumbel-Top-k for efficient root action selection (from Gumbel AlphaZero paper)
- Stochastic MCTS for games with chance nodes (2048, backgammon)

This allows efficient training on stochastic games with far fewer simulations.
"""

from functools import partial
from typing import Dict, Optional, Tuple, Callable
import jax
import jax.numpy as jnp
import chex

from core.evaluators.mcts.stochastic_mcts import StochasticMCTS, DecisionStepFn, StochasticStepFn
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.state import MCTSTree, MCTSOutput, TraversalState, StochasticMCTSTree
from core.evaluators.mcts.gumbel import gumbel_top_k
from core.types import EvalFn, StepMetadata


class GumbelStochasticMCTS(StochasticMCTS):
    """StochasticMCTS with Gumbel-Top-k action selection at decision nodes.

    Key differences from standard StochasticMCTS:
    - At decision node roots: uses Gumbel-Top-k to sample k actions
    - At chance node roots: samples from stochastic probabilities (unchanged)
    - Non-root nodes: standard PUCT for decisions, probability matching for chance

    This provides the same benefits as GumbelMCTS:
    - Policy improvement with far fewer simulations
    - 50-100x speedup compared to standard MCTS

    Works with 2048, backgammon, and other stochastic games.
    """

    def __init__(
        self,
        eval_fn: EvalFn,
        action_selector: MCTSActionSelector,
        stochastic_action_probs: chex.Array,
        policy_size: int,
        max_nodes: int,
        num_iterations: int,
        gumbel_k: int = 16,
        decision_step_fn: Optional[DecisionStepFn] = None,
        stochastic_step_fn: Optional[StochasticStepFn] = None,
        temperature: float = 1.0,
        tiebreak_noise: float = 1e-8,
        persist_tree: bool = True,
        noise_scale: float = 0.05,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        min_num_iterations: int = 300,
        progressive_threshold: float = 1.0
    ):
        """
        Args:
            eval_fn: Leaf evaluation function
            action_selector: Action selector for NON-ROOT decision nodes
            stochastic_action_probs: Probability distribution over chance outcomes
            policy_size: Size of neural network policy output
            max_nodes: Max tree size
            num_iterations: Number of MCTS iterations (can be small, e.g., 16)
            gumbel_k: Number of actions to sample at root (default: 16)
            decision_step_fn: Step function for decision nodes
            stochastic_step_fn: Step function for chance nodes
            temperature: Temperature for final action selection
            ... (other args same as StochasticMCTS)
        """
        super().__init__(
            eval_fn=eval_fn,
            action_selector=action_selector,
            stochastic_action_probs=stochastic_action_probs,
            policy_size=policy_size,
            max_nodes=max_nodes,
            num_iterations=num_iterations,
            decision_step_fn=decision_step_fn,
            stochastic_step_fn=stochastic_step_fn,
            temperature=temperature,
            tiebreak_noise=tiebreak_noise,
            persist_tree=persist_tree,
            noise_scale=noise_scale,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            min_num_iterations=min_num_iterations,
            progressive_threshold=progressive_threshold
        )
        self.gumbel_k = gumbel_k

    def get_config(self) -> Dict:
        """Returns config including Gumbel parameters."""
        config = super().get_config() if hasattr(super(), 'get_config') else {}
        config.update({
            "gumbel_k": self.gumbel_k,
            "type": "GumbelStochasticMCTS"
        })
        return config

    def _decision_evaluate(
        self,
        key: chex.PRNGKey,
        eval_state: StochasticMCTSTree,
        env_state: chex.ArrayTree,
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
        decision_step_fn: DecisionStepFn,
        stochastic_step_fn: StochasticStepFn,
    ) -> MCTSOutput:
        """Gumbel-enhanced evaluation for decision nodes.

        Uses Gumbel-Top-k to select which actions to explore at root,
        then performs MCTS iterations cycling through those actions.
        """
        key, root_key, gumbel_key, sample_key = jax.random.split(key, 4)

        # Reset tree if needed
        root_n = eval_state.data.n[eval_state.ROOT_INDEX]
        should_reset = jnp.logical_and(
            not self.persist_tree,
            root_n > 0
        )
        eval_state = jax.lax.cond(
            should_reset,
            lambda: eval_state.reset(),
            lambda: eval_state
        )

        # Update root if needed
        root_n = eval_state.data.n[eval_state.ROOT_INDEX]
        should_update_root = jnp.logical_or(
            not self.persist_tree,
            root_n == 0
        )
        eval_state = jax.lax.cond(
            should_update_root,
            lambda: self.update_root(root_key, eval_state, env_state, params, root_metadata=root_metadata),
            lambda: eval_state
        )

        # Get root policy for Gumbel sampling
        root_policy = eval_state.data.p[eval_state.ROOT_INDEX]

        # Convert to logits
        root_logits = jnp.log(jnp.maximum(root_policy[:self.policy_size], 1e-8))

        # Get legal mask and pad if needed
        legal_mask = root_metadata.action_mask
        mask_size = legal_mask.shape[-1]
        if mask_size < self.policy_size:
            padded_mask = jnp.zeros(self.policy_size, dtype=bool)
            padded_mask = padded_mask.at[:mask_size].set(legal_mask)
            legal_mask = padded_mask

        # Clamp gumbel_k to policy_size (static computation for JIT)
        effective_k = min(self.gumbel_k, self.policy_size)

        # Sample k actions using Gumbel-Top-k
        selected_actions, _ = gumbel_top_k(
            gumbel_key, root_logits, effective_k, legal_mask
        )

        # Perform MCTS iterations
        iterate_fn = lambda k, tree, iteration: self._gumbel_iterate(
            k, tree, iteration, params, decision_step_fn, stochastic_step_fn, selected_actions
        )

        def scan_body(carry, iteration):
            state, k = carry
            k, iter_key = jax.random.split(k)
            new_state = iterate_fn(iter_key, state, iteration)
            return (new_state, k), None

        iterations = jnp.arange(self.num_iterations)
        (eval_state, _), _ = jax.lax.scan(
            scan_body,
            (eval_state, key),
            iterations
        )

        # Sample action
        action, policy_weights = self.sample_root_action(
            sample_key, eval_state, action_mask=root_metadata.action_mask
        )
        policy_weights = policy_weights[:self.policy_size]

        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=policy_weights
        )

    def _gumbel_iterate(
        self,
        key: chex.PRNGKey,
        tree: StochasticMCTSTree,
        iteration: int,
        params: chex.ArrayTree,
        decision_step_fn: DecisionStepFn,
        stochastic_step_fn: StochasticStepFn,
        selected_actions: chex.Array
    ) -> StochasticMCTSTree:
        """Single Gumbel iteration for stochastic games.

        At decision root: picks from Gumbel-selected actions
        At non-root: uses standard StochasticMCTS selection
        """
        traverse_key, expand_key, backprop_key = jax.random.split(key, 3)

        # Gumbel selection at root
        is_root_stochastic = self._is_chance_node(tree, tree.ROOT_INDEX)

        # For decision root: use Gumbel selection
        # For chance root: shouldn't reach here (handled by stochastic_evaluate)

        root_child_visits = tree.get_child_data('n', tree.ROOT_INDEX)[:self.policy_size]
        selected_visits = root_child_visits[selected_actions]

        # Pick action with minimum visits among selected
        min_visits = jnp.min(selected_visits)
        is_min_mask = selected_visits == min_visits
        action_idx = jnp.argmax(
            is_min_mask * (1.0 + 0.1 * (jnp.arange(len(selected_actions)) == (iteration % len(selected_actions))))
        )
        root_action = selected_actions[action_idx]

        # Check if we need to traverse deeper or expand from root
        root_child_exists = tree.is_edge(tree.ROOT_INDEX, root_action)
        root_child_idx = tree.edge_map[tree.ROOT_INDEX, root_action]

        def traverse_deeper():
            """Continue traversal from existing child."""
            safe_child_idx = jnp.maximum(root_child_idx, 0)
            child_terminated = tree.data.terminated[safe_child_idx]

            def do_traverse():
                # Use standard traversal from child
                init_state = TraversalState(parent=root_child_idx, action=self.cond_action_selector(traverse_key, tree, root_child_idx))

                def cond_fn(s):
                    child_idx = tree.edge_map[s.parent, s.action]
                    safe_idx = jnp.maximum(child_idx, 0)
                    return jnp.logical_and(
                        tree.is_edge(s.parent, s.action),
                        ~tree.data.terminated[safe_idx]
                    )

                def body_fn(s):
                    node_idx = tree.edge_map[s.parent, s.action]
                    iter_key = jax.random.fold_in(traverse_key, node_idx)
                    next_action = self.cond_action_selector(iter_key, tree, node_idx)
                    return TraversalState(parent=node_idx, action=next_action)

                return jax.lax.while_loop(cond_fn, body_fn, init_state)

            return jax.lax.cond(
                child_terminated,
                lambda: TraversalState(parent=tree.ROOT_INDEX, action=root_action),
                do_traverse
            )

        def expand_from_root():
            return TraversalState(parent=tree.ROOT_INDEX, action=root_action)

        traversal_state = jax.lax.cond(
            root_child_exists,
            traverse_deeper,
            expand_from_root
        )

        parent, action = traversal_state.parent, traversal_state.action

        # Expand based on parent type
        is_parent_stochastic = self._is_chance_node(tree, parent)

        def expand_stochastic():
            from core.evaluators.mcts.stochastic_mcts import ExpandResult
            result = self._expand_single_stochastic_child(
                expand_key, tree, parent, action, params, stochastic_step_fn
            )
            return result.tree, result.child_idx, result.value

        def expand_decision():
            from core.evaluators.mcts.stochastic_mcts import ExpandResult
            result = self._expand_single_child(
                expand_key, tree, parent, action, params, decision_step_fn
            )
            return result.tree, result.child_idx, result.value

        final_tree, final_child, final_value = jax.lax.cond(
            is_parent_stochastic,
            expand_stochastic,
            expand_decision
        )

        return self.backpropagate(backprop_key, final_tree, parent, final_child, final_value)
