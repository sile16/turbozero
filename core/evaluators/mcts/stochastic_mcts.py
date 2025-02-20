from typing import Dict, Optional, Tuple
import chex
from chex import dataclass
import jax
import jax.numpy as jnp

from core.evaluators.mcts.mcts import MCTS
from core.evaluators.mcts.state import MCTSTree, MCTSNode, TraversalState

@dataclass(frozen=True)
class StochasticMCTSNode(MCTSNode):
    """
    Extended MCTS node that supports stochastic events.
    
    Attributes:
      is_stochastic_node: Flag indicating if this node represents a stochastic event.
      stochastic_probs: A probability distribution over stochastic outcomes.
    """
    is_stochastic_node: bool = False
    stochastic_probs: Optional[chex.Array] = None

class StochasticMCTS(MCTS):
    """
    MCTS evaluator with explicit support for stochastic nodes.
    
    This class extends the original MCTS by:
      - Overriding the node creation method to produce StochasticMCTSNode instances.
      - Overriding the traverse method to handle stochastic nodes.
    
    The tree can contain both deterministic and stochastic nodes.
    """
    def __init__(self, stochastic_action_selector_fn, *args, **kwargs):
        """
        Args:
          stochastic_action_selector_fn: A function that takes a stochastic probability distribution and
                                  returns a sampled outcome (action index) for stochastic nodes.
        """
        super().__init__(*args, **kwargs)
        self.stochastic_action_selector_fn = stochastic_action_selector_fn
    
    

    @staticmethod
    def new_node(policy: chex.Array, value: float, embedding: chex.Array, terminated: bool,
                 is_stochastic_node: bool = False, stochastic_probs: Optional[chex.Array] = None) -> StochasticMCTSNode:
        """
        Creates a new StochasticMCTSNode.
        
        Args:
          policy: The policy vector for the node.
          value: The value estimate.
          embedding: The state embedding.
          terminated: Whether the state is terminal.
          is_stochastic_node: Set to True if this node represents a stochastic event.
          stochastic_probs: The probability distribution for the stochastic outcomes.
        
        Returns:
          A StochasticMCTSNode instance.
        """
        return StochasticMCTSNode(
            n=jnp.array(1, dtype=jnp.int32),
            p=policy,
            q=jnp.array(value, dtype=jnp.float32),
            terminated=jnp.array(terminated, dtype=jnp.bool_),
            embedding=embedding,
            is_stochastic_node=jnp.array(is_stochastic_node, dtype=jnp.bool_),
            stochastic_probs=stochastic_probs
        )

    def traverse(self, tree: MCTSTree) -> TraversalState:
        """
        Traverses the tree from the root until reaching a leaf or terminal node.
        
        When a stochastic node is encountered, the stochastic_action_selector_fn is used 
        to sample the outcome based on the node's stochastic_probs.
        
        Returns:
        A TraversalState containing the parent index and chosen action.
        """
        # Define condition function for while loop
        def cond_fn(state: TraversalState) -> bool:
            is_valid_edge = tree.is_edge(state.parent, state.action)
            # If it's a valid edge, check if child is terminated
            child_idx = tree.edge_map[state.parent, state.action]
            child_terminated = jnp.where(
                is_valid_edge,
                tree.data_at(child_idx).terminated,
                jnp.array(True, dtype=jnp.bool_)  # Default to terminated for invalid edges
            )
            return jnp.logical_and(is_valid_edge, ~child_terminated)
        
        # Define body function for while loop
        def body_fn(state: TraversalState) -> TraversalState:
            child_idx = tree.edge_map[state.parent, state.action]
            child_node = tree.data_at(child_idx)
            
            # Check if node is stochastic, and use appropriate action selector
            is_stochastic = child_node.is_stochastic_node
            
            def select_stochastic_action(_):
                # For stochastic nodes, use stochastic_action_selector_fn
                return self.stochastic_action_selector_fn(child_node.stochastic_probs)
                
            def select_regular_action(_):
                # For regular nodes, use the standard action selector
                return self.action_selector(tree, child_idx, self.discount)
            
            next_action = jax.lax.cond(
                is_stochastic,
                select_stochastic_action,
                select_regular_action,
                operand=None
            )
            
            return TraversalState(parent=child_idx, action=next_action)
        
        # Initial state: start at root with action selected by standard action_selector
        initial_state = TraversalState(
            parent=tree.ROOT_INDEX,
            action=self.action_selector(tree, tree.ROOT_INDEX, self.discount)
        )
        
        # Use JAX's functional while loop
        return jax.lax.while_loop(cond_fn, body_fn, initial_state)
    

    def backpropagate(self, key: chex.PRNGKey, tree: MCTSTree, parent: int, value: float) -> MCTSTree:
        def body_fn(state: BackpropState) -> Tuple[int, MCTSTree]:
            node_idx, tree, value = state.node_idx, state.tree, state.value
            node = tree.data_at(node_idx)
            # Get child q-values (with discount) and visit counts.
            child_q_values = tree.get_child_data('q', node_idx) * self.discount
            child_n_values = tree.get_child_data('n', node_idx)

            # If the node is stochastic, use its stored probabilities to aggregate child values.
            if isinstance(node, StochasticMCTSNode) and node.is_stochastic_node:
                # Compute expected value from stochastic outcomes.
                weighted_value = jnp.sum(node.stochastic_probs * child_q_values)
            else:
                # For deterministic nodes, proceed as in WeightedMCTS.
                normalized_q_values = normalize_q_values(
                    child_q_values, child_n_values, node.q, jnp.finfo(node.q).eps
                )
                if self.q_temperature > 0:
                    q_values = normalized_q_values ** (1 / self.q_temperature)
                    q_values_masked = jnp.where(child_n_values > 0,
                                                normalized_q_values,
                                                jnp.finfo(normalized_q_values).min)
                else:
                    # Use noise to break ties when temperature is 0.
                    noise = jax.random.uniform(key, shape=normalized_q_values.shape, maxval=self.tiebreak_noise)
                    noisy_q_values = normalized_q_values + noise
                    max_vector = jnp.full_like(noisy_q_values, jnp.finfo(noisy_q_values).min)
                    index_of_max = jnp.argmax(noisy_q_values)
                    max_vector = max_vector.at[index_of_max].set(1)
                    q_values = normalized_q_values
                    q_values_masked = max_vector
                child_weights = jax.nn.softmax(q_values_masked, axis=-1)
                weighted_value = jnp.sum(child_weights * q_values)
            
            # Update the node with the computed weighted value.
            node = node.replace(q=weighted_value)
            # Update visit counts and other metrics.
            node = self.visit_node(node, node.r)
            tree = tree.update_node(node_idx, node)
            # Propagate upward.
            return BackpropState(node_idx=tree.parents[node_idx], value=value, tree=tree)
        
        state = jax.lax.while_loop(
            lambda s: s.node_idx != s.tree.NULL_INDEX,
            body_fn,
            BackpropState(node_idx=parent, value=value, tree=tree)
        )
        return state.tree


def stochastic_action_selector_fn(stochastic_probs: jnp.ndarray, key) -> int:
    """
          During search, selects a stochastic action (dice roll) based on probabilities.
          stochastic_probs is an array of deterministic actions plus additional values for stochastic actions
          All the determinitic values should be 0, where stochastic are between 0-1 based on probability.
          sum(stochastic_probs) should = 1
    
    """
    actions = jnp.arange(stochastic_probs.shape[0])
    # During search we want to explore all possibilities weighted by probability
    return jax.random.choice(key, a=actions, p=stochastic_probs)