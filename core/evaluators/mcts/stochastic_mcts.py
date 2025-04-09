from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import chex
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.state import BackpropState, MCTSNode, MCTSTree, TraversalState, MCTSOutput
from core.evaluators.mcts.mcts import MCTS
from core.types import EnvStepFn, EvalFn, StepMetadata


class StochasticMCTS(MCTS):
    """Batched implementation of Monte Carlo Tree Search (MCTS).
    
    Not stateful. This class operates on 'MCTSTree' state objects.
    
    Compatible with `jax.vmap`, `jax.pmap`, `jax.jit`, etc."""
    def __init__(self,
        eval_fn: EvalFn,
        action_selector: MCTSActionSelector,
        stochastic_action_probs: chex.Array,
        branching_factor: int,
        max_nodes: int,
        num_iterations: int,
        discount: float = -1.0,
        temperature: float = 1.0,
        tiebreak_noise: float = 1e-8,
        persist_tree: bool = True,
        noise_scale: float = 0.05
    ):
        """
        Args:
        - `eval_fn`: leaf node evaluation function (env_state -> (policy_logits, value))
        - `action_selector`: action selection function (eval_state -> action)
        - `branching_factor`: max number of actions (== children per node)
        - `max_nodes`: allocated size of MCTS tree, any additional nodes will not be created, 
                but values from out-of-bounds leaf nodes will still backpropagate
        - `num_iterations`: number of MCTS iterations to perform per evaluate call
        - `discount`: discount factor for MCTS (default: -1.0)
            - use a negative discount in two-player games (e.g. -1.0)
            - use a positive discount in single-player games (e.g. 1.0)
        - `temperature`: temperature for root action selection (default: 1.0)
        - `tiebreak_noise`: magnitude of noise to add to policy weights for breaking ties (default: 1e-8)
        - `persist_tree`: whether to persist search tree state between calls to `evaluate` (default: True)
        - `noise_scale`: scale of noise to add to delta values in stochastic action selection (default: 0.05)
        """
        super().__init__(eval_fn, action_selector, branching_factor, max_nodes, num_iterations, discount, temperature, tiebreak_noise, persist_tree)

        self.stochastic_action_probs = stochastic_action_probs
        self.noise_scale = noise_scale

    def stochastic_action_selector(self, key: chex.PRNGKey, tree, node_idx, discount):
        """Select an action from the node, no use in exploring at this point so we just use the stochastic action probs.
        i.e. we are rolling the dice to get the next deterministic state.
        
        Args:
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from
        """
        # Properly handle the key for jax.random.choice which expects a single key
        choice_key, _ = jax.random.split(key)
        action = jax.random.choice(choice_key, len(self.stochastic_action_probs), p=self.stochastic_action_probs)
            
        return jnp.array(action, dtype=jnp.int32)
    
    
    def deterministic_action_selector(self, key: chex.PRNGKey, tree: MCTSTree, node_idx: int, discount: float) -> int:
        """Just a wrapper function to call the action selector removing the key argument.
        
        Args:
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from
        """
        return self.action_selector(tree, node_idx, discount)
    
    def explore_stochastic_action_selector(self, key: chex.PRNGKey, tree, node_idx, discount):
        """Called when traversing the tree and exploring possibilities.
        
        Selects the action with the largest discrepancy between the child's observed
        visit frequency and its theoretical stochastic probability.
        
        Args:
        - `key`: rng
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from
        - `discount`: discount factor (not used in this selector)
        
        Returns:
        - The action with the largest delta between observed visit frequency and stochastic probability
        """
        # Get the number of stochastic actions
        num_stochastic_actions = len(self.stochastic_action_probs)
        
        # Get the mapping from edge indices to child node indices, but only for stochastic actions
        child_indices = tree.edge_map[node_idx, :num_stochastic_actions]
        
        # Create a mask for edges that exist (not NULL_INDEX)
        child_exists_mask = child_indices != tree.NULL_INDEX
        
        # For vectorized child visit count retrieval, we need to handle NULL_INDEX
        # We'll replace NULL_INDEX with 0 for array access, then mask the result
        safe_indices = jnp.maximum(child_indices, 0)  # Replace -1 with 0 for safe access
        
        # Vectorized access to visit counts - get n for all safe_indices
        # Shape: (num_stochastic_actions,)
        all_n = tree.data.n[safe_indices]
        
        # Apply mask to zero out non-existent children
        # Shape: (num_stochastic_actions,)
        child_visits = jnp.where(child_exists_mask, all_n, 0)
        
        # Calculate total visits
        total_visits = jnp.sum(child_visits)
        
        # Normalize visit counts; handle the case where total_visits=0
        normalized_visits = jnp.where(
            total_visits > 0,
            child_visits / total_visits,
            # If total_visits=0, use a uniform distribution
            jnp.ones_like(child_visits, dtype=jnp.float32) / num_stochastic_actions
        )
        
        # Calculate deltas between actual visit frequencies and theoretical probabilities
        # Now both arrays have shape (num_stochastic_actions,)
        delta = self.stochastic_action_probs - normalized_visits 
        
        # Add some noise to break ties
        key, noise_key = jax.random.split(key)
        delta = delta + jax.random.normal(noise_key, delta.shape) * self.noise_scale
        
        # Return the action with the biggest delta
        return jnp.argmax(delta)
    
    

    def cond_action_selector(self, key: chex.PRNGKey, tree: MCTSTree, node_idx: int, discount: float) -> int:
        """Select an action from the node, picks the right action selector based on the node type.
        
        Args:
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from
        """

        # Create lambda functions that capture all required parameters
        return jax.lax.cond(
            StochasticMCTS.is_node_idx_stochastic(tree, node_idx), 
            lambda k, t, n, d: self.explore_stochastic_action_selector(k, t, n, d),
            lambda k, t, n, d: self.deterministic_action_selector(k, t, n, d),
            key, tree, node_idx, discount
        )
        
    
    def det_value_policy(self, embedding, params, eval_key, metadata, player_reward) -> Tuple[float, chex.Array]:
        """Get value and policy for deterministic nodes."""
        policy_logits, value = self.eval_fn(embedding, params, eval_key)
        policy_logits = jnp.where(metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min)
        policy = jax.nn.softmax(policy_logits)
        value = jnp.where(metadata.terminated, player_reward, value)
        return value, policy

    def stochastic_value_policy(self, tree, parent_idx, embedding, params, eval_key, metadata, player_reward) -> Tuple[float, chex.Array]:
        """Get value and policy for stochastic nodes.
        
        For stochastic nodes:
        - If parent exists (not root), set value to the parent's Q-value
        - Otherwise use 0.0 as a neutral value
        - Set policy to all zeros as stochastic nodes don't have meaningful policies
        """
        # Get parent's Q-value if parent exists (not ROOT_INDEX)
        parent_is_root = parent_idx == tree.NULL_INDEX  # No parent (this is root)
        
        # For root stochastic nodes, use a neutral value (0.0)
        # For non-root stochastic nodes, inherit parent's Q-value
        parent_q = jnp.where(parent_is_root, 
                             0.0,  # Default value if root
                             tree.data_at(parent_idx).q)  # Parent's Q-value
        
        # Just use zeros for policy since stochastic nodes don't have meaningful policies
        policy = jnp.zeros((self.branching_factor,), dtype=jnp.float32)
        
        return parent_q, policy

    def iterate(self, key: chex.PRNGKey, tree: MCTSTree, params: chex.ArrayTree, env_step_fn: EnvStepFn) -> MCTSTree:
        """ Performs one iteration of MCTS.
        1. Traverse to leaf node.
        2. Evaluate Leaf Node
        3. Expand Leaf Node (add to tree)
        4. Backpropagate

        Args:
        - `tree`: MCTSTree to evaluate
        - `params`: parameters to pass to the the leaf evaluation function
        - `env_step_fn`: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
        - (MCTSTree): updated MCTSTree
        """
        # traverse from root -> leaf
        traversal_state = self.traverse(key, tree)
        parent, action = traversal_state.parent, traversal_state.action
        # get env state (embedding) for leaf node
        embedding = tree.data_at(parent).embedding
        
        # Split key for step function and evaluation
        step_key, eval_key, key = jax.random.split(key, 3)
        
        # Call step function with the required key parameter
        new_embedding, metadata = env_step_fn(embedding, action, step_key)
        
        player_reward = metadata.rewards[metadata.cur_player_id]
        
        # Check if the new state is stochastic
        is_stochastic = new_embedding.is_stochastic
        
        # Evaluate leaf node - use different evaluation based on stochastic vs deterministic
        value, policy = jax.lax.cond(
            is_stochastic,
            lambda: self.stochastic_value_policy(tree, parent, new_embedding, params, eval_key, metadata, player_reward),
            lambda: self.det_value_policy(new_embedding, params, eval_key, metadata, player_reward)
        )
        
        # add leaf node to tree
        node_exists = tree.is_edge(parent, action)
        node_idx = tree.edge_map[parent, action]

        node_data = jax.lax.cond(
            node_exists,
            lambda: self.visit_node(node=tree.data_at(node_idx), value=value, p=policy, terminated=metadata.terminated, embedding=new_embedding),
            lambda: self.new_node(policy=policy, value=value, embedding=new_embedding, terminated=metadata.terminated)
        )

        tree = jax.lax.cond(
            node_exists,
            lambda: tree.update_node(index=node_idx, data = node_data),
            lambda: tree.add_node(parent_index=parent, edge_index=action, data=node_data)
        )
        # backpropagate
        return self.backpropagate(key, tree, parent, value)


    def traverse(self, key: chex.PRNGKey, tree: MCTSTree) -> TraversalState:
        """ Traverse from the root node until an unvisited leaf node is reached.
        
        Args:
        - `tree`: MCTSTree to evaluate
        
        Returns:
        - (TraversalState): state of the traversal
            - `parent`: index of the parent node
            - `action`: action to take from the parent node
        """

        # continue while:
        # - there is an existing edge corresponding to the chosen action
        # - AND the child node connected to that edge is not terminal
        def cond_fn(state: TraversalState) -> bool:
            return jnp.logical_and(
                tree.is_edge(state.parent, state.action),
                ~(tree.data_at(tree.edge_map[state.parent, state.action]).terminated)
                # TODO: maximum depth
            )
        
        # each iterration:
        # - get the index of the child node connected to the chosen action
        # - choose the action to take from the child node
        def body_fn(state: TraversalState) -> TraversalState:
            node_idx = tree.edge_map[state.parent, state.action]
            action = self.cond_action_selector(key, tree, node_idx, self.discount)
            return TraversalState(parent=node_idx, action=action)
        
        # choose the action to take from the root
        root_action = self.cond_action_selector(key, tree, tree.ROOT_INDEX, self.discount)
        # traverse from root to leaf
        return jax.lax.while_loop(
            cond_fn, body_fn, 
            TraversalState(parent=tree.ROOT_INDEX, action=root_action)
        )


    def evaluate(self, #pylint: disable=arguments-differ
        key: chex.PRNGKey,
        eval_state: MCTSTree, 
        env_state: chex.ArrayTree,
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
        env_step_fn: EnvStepFn,
        **kwargs
    ) -> MCTSOutput:
        """Performs `self.num_iterations` MCTS iterations on an `MCTSTree`.
        Samples an action to take from the root node after search is completed.
        
        Args:
        - `eval_state`: `MCTSTree` to evaluate, could be empty or partially complete
        - `env_state`: current environment state
        - `root_metadata`: metadata for the root node of the tree
        - `params`: parameters to pass to the the leaf evaluation function
        - `env_step_fn`: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
        - (MCTSOutput): contains new tree state, selected action, root value, and policy weights
        """
        tree = eval_state
        
        # Define lambda functions that capture all required parameters
        def true_fn():
            return self.stochastic_evaluate(key, eval_state, env_state, root_metadata, params, env_step_fn)
        
        def false_fn():
            return MCTS.evaluate(self, key, eval_state, env_state, root_metadata, params, env_step_fn)
        
        return jax.lax.cond(
            StochasticMCTS.is_node_idx_stochastic(tree, tree.ROOT_INDEX),
            true_fn,
            false_fn
        )

    def stochastic_evaluate(self, #pylint: disable=arguments-differ
        key: chex.PRNGKey,
        eval_state: MCTSTree, 
        env_state: chex.ArrayTree,
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
        env_step_fn: EnvStepFn,
        **kwargs
    ) -> MCTSOutput:
        """Performs `self.num_iterations` MCTS iterations on an `MCTSTree`.
        Samples an action to take from the root node after search is completed.
        
        Args:
        - `eval_state`: `MCTSTree` to evaluate, could be empty or partially complete

        """
        # Get the action using stochastic_action_selector
        action = self.stochastic_action_selector(key, eval_state, eval_state.ROOT_INDEX, self.discount)
        
        # Convert to integer32 to match MCTS.evaluate output type 
        
        
        # Create zero policy weights of the right shape
        policy_weights = jnp.zeros((self.branching_factor,), dtype=jnp.float32)

        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=policy_weights
        )

    def backpropagate(self, key: chex.PRNGKey, tree: MCTSTree, parent: int, value: float) -> MCTSTree: #pylint: disable=unused-argument
        """Backpropagate the value estimate from the leaf node to the root node and update visit counts.
        
        Args:
        - `key`: rng
        - `tree`: MCTSTree to evaluate
        - `parent`: index of the parent node (in most cases, this is the new node added to the tree this iteration)
        - `value`: value estimate of the leaf node

        Returns:
        - (MCTSTree): updated search tree
        """

        def body_fn(state: BackpropState) -> Tuple[int, MCTSTree]:
            node_idx, value, tree = state.node_idx, state.value, state.tree
            # apply discount to value estimate
            value *= self.discount
            node = tree.data_at(node_idx)
            # increment visit count and update value estimate
            new_node = self.visit_node(node, value)
            tree = tree.update_node(node_idx, new_node)
            # go to parent 
            return BackpropState(node_idx=tree.parents[node_idx], value=value, tree=tree)
        
        # backpropagate while the node is a valid node
        # the root has no parent, so the loop will terminate 
        # when the parent of the root is visited
        state = jax.lax.while_loop(
            lambda s: s.node_idx != s.tree.NULL_INDEX, body_fn, 
            BackpropState(node_idx=parent, value=value, tree=tree)
        )
        return state.tree

    @staticmethod
    def stochastic_new_q_value(node: MCTSNode, value: float) -> float:
        """Calculate the new Q value for a node.
        
        Args:
        - `node`: MCTSNode to calculate the new Q value for
        - `value`: value estimate of the leaf node
        """
        # do a normalized weighted sum of the q values of the children
        # get the q values of the children
        q_values = node.q
        # get the visit counts of the children
        visit_counts = node.n
        # normalize the visit counts
        visit_counts = visit_counts / jnp.sum(visit_counts)
        # do a weighted sum of the q values
        return jnp.sum(q_values * visit_counts)

    @staticmethod
    def deterministic_new_q_value(node: MCTSNode, value: float) -> float:
        """Calculate the new Q value for a node.

        Args:
        - `node`: MCTSNode to calculate the new Q value for
        - `value`: value estimate of the leaf node
        """
        return((node.q * node.n) + value) / (node.n + 1)

    @staticmethod
    def is_node_stochastic(node: MCTSNode) -> jnp.bool_:
        """Check if the node is stochastic.
        
        Returns a JAX boolean that can be used with jax.lax.cond.
        """
        # Access the is_stochastic flag directly without conversion to Python bool
        # which would cause tracing errors
        return node.embedding.is_stochastic
    
    @staticmethod
    def is_node_idx_stochastic(tree: MCTSTree, node_idx: int) -> jnp.bool_:
        """Check if the node is stochastic.
        
        Returns a JAX boolean that can be used with jax.lax.cond.
        """
        # Access the is_stochastic flag directly from the embedded state
        # Don't convert to Python bool as that would break JAX tracing
        return tree.data_at(node_idx).embedding.is_stochastic

    @staticmethod
    def visit_node(
        node: MCTSNode,
        value: float,
        p: Optional[chex.Array] = None,
        terminated: Optional[bool] = None,
        embedding: Optional[chex.ArrayTree] = None
    ) -> MCTSNode:
        """ Update the visit counts and value estimate of a node.

        Args:
        - `node`: MCTSNode to update
        - `value`: value estimate to update the node with

        ( we could optionally overwrite the following: )
        - `p`: policy weights to update the node with
        - `terminated`: whether the node is terminal
        - `embedding`: embedding to update the node with

        Returns:
        - (MCTSNode): updated MCTSNode
        """
        # update running value estimate
        q_value = jax.lax.cond(
            StochasticMCTS.is_node_stochastic(node), 
            StochasticMCTS.stochastic_new_q_value, 
            StochasticMCTS.deterministic_new_q_value, 
            node, value
        )
        # update other node attributes
        if p is None:
            p = node.p
        if terminated is None:
            terminated = node.terminated
        if embedding is None:
            embedding = node.embedding
        return node.replace(
            n=node.n + 1, # increment visit count
            q=q_value,
            p=p,
            terminated=terminated,
            embedding=embedding
        )

