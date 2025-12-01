from typing import Tuple
import jax
import jax.numpy as jnp
import chex
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.state import BackpropState, MCTSNode, MCTSTree, TraversalState, MCTSOutput
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.alphazero import AlphaZero
from core.types import EnvStepFn, EvalFn, StepMetadata


class StochasticMCTS(AlphaZero(MCTS)):
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
        noise_scale: float = 0.05,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        min_num_iterations: int = 300
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
        - `dirichlet_alpha`: magnitude of Dirichlet noise for AlphaZero (default: 0.3)
        - `dirichlet_epsilon`: proportion of root policy composed of Dirichlet noise (default: 0.25)
        """
        super().__init__(dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon, eval_fn=eval_fn, action_selector=action_selector, branching_factor=branching_factor, max_nodes=max_nodes, num_iterations=num_iterations, discount=discount, temperature=temperature, tiebreak_noise=tiebreak_noise, persist_tree=persist_tree)

        self.stochastic_action_probs = stochastic_action_probs
        self.noise_scale = noise_scale
        self.max_num_iterations = self.num_iterations
        self.min_num_iterations = min(min_num_iterations, self.max_num_iterations)
        


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
        self.num_iterations = max(int(self.max_num_iterations * (1 - self.temperature)), self.min_num_iterations)
        
        # Store super class reference for use in nested function
        super_evaluate = super().evaluate
        
        # Define lambda functions that capture all required parameters
        def true_fn():
            return self.stochastic_evaluate(key, eval_state, env_state, root_metadata, params, env_step_fn)
        
        def false_fn():
            return super_evaluate(key, eval_state, env_state, root_metadata, params, env_step_fn)
        
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
        """.
        Samples an action to take from the root based on it's stochastic actions.
        
        Args:
        - `eval_state`: `MCTSTree` to evaluate, could be empty or partially complete

        """
        # Only update root if tree is empty or persist_tree=False
        key, root_key = jax.random.split(key)
        # OPTIMIZATION: Direct array access instead of data_at() which reconstructs entire node
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

        # Get the action using stochastic_action_selector
        action = self.stochastic_action_sample(key, eval_state, eval_state.ROOT_INDEX, self.discount)
        
        # Create a dummy policy with the correct shape (branching_factor)
        # This will be ignored during training since we're in a stochastic state
        policy_weights = jnp.zeros(self.branching_factor)
        
        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=policy_weights
        )
    

    def stochastic_action_sample(self, key: chex.PRNGKey, tree, node_idx, discount):
        """Select an action from the node, not use in exploring at this point so we just use the stochastic action probs.
        i.e. we are rolling the dice to get the next deterministic state.

        Args:
        - `key`: rng key (used directly, no split needed)
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from
        """
        # OPTIMIZATION: Use key directly instead of splitting and discarding half
        action = jax.random.choice(key, len(self.stochastic_action_probs), p=self.stochastic_action_probs)

        return jnp.array(action, dtype=jnp.int32)
    
    
    def deterministic_action_selector(self, key: chex.PRNGKey, tree: MCTSTree, node_idx: int) -> int:
        """called when traversing the tree and exploring possibilities, a wrapper to remove the key

        Args:
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from

        Optimized: Direct array access instead of data_at() to avoid full node reconstruction.
        """
        # Direct access to current_player array
        current_player = tree.data.embedding.current_player[node_idx]

        # Get child players - direct array slice
        child_indices = tree.edge_map[node_idx]

        # Safe access to child players (handling NULL_INDEX)
        safe_indices = jnp.maximum(child_indices, 0)
        child_players = tree.data.embedding.current_player[safe_indices]

        # Calculate per-child discount based on player transitions
        # Formula: 1.0 - 2.0 * |current_player - child_player|
        # - Same player (0 vs 0 or 1 vs 1): 1.0 - 2.0 * 0 = 1.0 (keep value)
        # - Different player (0 vs 1 or 1 vs 0): 1.0 - 2.0 * 1 = -1.0 (invert value)
        # This produces a vector of discounts (one per child), which JAX broadcasting
        # handles correctly when multiplied with Q-values in the action selector.
        discounts = 1.0 - 2.0 * jnp.abs(current_player - child_players)

        return self.action_selector(tree, node_idx, discounts)
    
    def stochastic_action_selector(self, key: chex.PRNGKey, tree, node_idx) -> int:
        """Called when traversing the tree and exploring possibilities.
        
        Selects the action with the largest discrepancy between the child's observed
        visit frequency and its theoretical stochastic probability.  This works when probabilities
        are fairly similar and there isn't some very unlikely action that results in a large reward. 
        
        Args:
        - `key`: rng
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from
        - `discount`: discount factor (not used in this selector)
        
        Returns:
        - The action with the largest delta between observed visit frequency and stochastic probability
        """
        
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
        # NOTE: This selects actions that are under-visited relative to their probability.
        # LIMITATION: This approach may miss low-probability but high-reward actions.
        # If some stochastic outcomes have small probabilities but lead to significantly
        # better rewards, this selector will under-explore them since it only balances
        # visit frequency with theoretical probability.
        # POTENTIAL IMPROVEMENT: Could incorporate value estimates or UCB-like exploration
        # bonus for high-value but low-probability actions, e.g.:
        # delta = (self.stochastic_action_probs - normalized_visits) + exploration_bonus
        # where exploration_bonus considers both probability and estimated value.
        delta = self.stochastic_action_probs - normalized_visits 
        
        # Add some noise to break ties
        key, noise_key = jax.random.split(key)
        delta = delta + jax.random.normal(noise_key, delta.shape) * self.noise_scale
        
        # Return the action with the biggest delta
        return jnp.argmax(delta)
    
    

    def cond_action_selector(self, key: chex.PRNGKey, tree: MCTSTree, node_idx: int) -> int:
        """Select an action from the node, picks the right action selector based on the node type.
        We can't just override the action_selector in the base class as we need to pass in the key.
        
        Args:
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from
        """

        # Create lambda functions that capture all required parameters
        return jax.lax.cond(
            StochasticMCTS.is_node_idx_stochastic(tree, node_idx), 
            self.stochastic_action_selector,
            self.deterministic_action_selector,
            key, tree, node_idx
        )
    
    
    def value_policy(self, embedding, params, eval_key, metadata, player_reward) -> Tuple[float, chex.Array]:
        """Get value and policy for deterministic nodes."""
        # Get policy and value from eval_fn for deterministic states
        policy_logits, value = self.eval_fn(embedding, params, eval_key)
        policy_logits = jnp.where(metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min)
        policy = jax.nn.softmax(policy_logits)
        value = jnp.where(metadata.terminated, player_reward, value)
        
        return value, policy


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
        # OPTIMIZATION: Direct access to embedding instead of data_at() which reconstructs entire node
        embedding = jax.tree_util.tree_map(lambda x: x[parent], tree.data.embedding)
        
        # Split key for step function and evaluation
        step_key, eval_key, key = jax.random.split(key, 3)
        
        new_embedding, metadata = env_step_fn(embedding, action, step_key)
        player_reward = metadata.rewards[metadata.cur_player_id] # we need to check this, is correct.
        
        value, policy = self.value_policy(new_embedding, params, eval_key, metadata, player_reward)

        # add leaf node to tree
        node_exists = tree.is_edge(parent, action)
        node_idx = tree.edge_map[parent, action]

        # OPTIMIZATION: For existing nodes, only update n/q stats (not the full node with embedding)
        # For new nodes, create the full node and add to tree
        def update_existing_node():
            # Only update n and q - embedding/policy/terminated are unchanged for revisits
            return StochasticMCTS._update_node_stats(tree, node_idx, value)

        def add_new_node():
            node_data = self.new_node(policy=policy, value=value, embedding=new_embedding, terminated=metadata.terminated)
            return tree.add_node(parent_index=parent, edge_index=action, data=node_data)

        tree = jax.lax.cond(node_exists, update_existing_node, add_new_node)

        # Get the correct node index after adding/updating
        # OPTIMIZATION: Use jnp.where instead of jax.lax.cond for simple scalar selection
        node_idx = jnp.where(node_exists, node_idx, tree.next_free_idx - 1)
        
        # backpropagate
        return self.backpropagate(key, tree, parent, node_idx, value) 


    def traverse(self, key: chex.PRNGKey, tree: MCTSTree) -> TraversalState:
        """Traverse from the root node until an unvisited leaf node is reached.
        Needed to be overridden as we need to pass in the key to action selector.
        And to use different action selectors for stochastic and deterministic nodes.
        Args:
        - `tree`: MCTSTree to evaluate
        
        Returns:
        - (TraversalState): state of the traversal
            - `parent`: index of the parent node
            - `action`: action to take from the parent node
        """
        
        # Split key for root action and body iterations
        root_key, body_key = jax.random.split(key)

        # continue while:
        # - there is an existing edge corresponding to the chosen action
        # - AND the child node connected to that edge is not terminal
        def cond_fn(state: TraversalState) -> bool:
            # OPTIMIZATION: Direct array access instead of data_at() for terminated check
            child_idx = tree.edge_map[state.parent, state.action]
            # Use safe index for array access (in case child_idx is NULL_INDEX)
            safe_child_idx = jnp.maximum(child_idx, 0)
            child_terminated = tree.data.terminated[safe_child_idx]
            return jnp.logical_and(
                tree.is_edge(state.parent, state.action),
                ~child_terminated
                # TODO: maximum depth
            )

        # each iterration:
        # - get the index of the child node connected to the chosen action
        # - choose the action to take from the child node
        def body_fn(state: TraversalState) -> TraversalState:
            node_idx = tree.edge_map[state.parent, state.action]
             
            # Generate unique key for this iteration based on parent and action
            # This ensures different randomness for each traversal step
            iteration_key = jax.random.fold_in(body_key, node_idx * tree.branching_factor + state.action)
            action = self.cond_action_selector(iteration_key, tree, node_idx)
            return TraversalState(parent=node_idx, action=action)
        
        # the action to take from the root
        root_action = self.cond_action_selector(root_key, tree, tree.ROOT_INDEX)
        # traverse from root to leaf
        return jax.lax.while_loop(
            cond_fn, body_fn,
            TraversalState(parent=tree.ROOT_INDEX, action=root_action)
        )



    def calculate_discount_factor(self, tree: MCTSTree, node_idx: int, other_idx: int) -> float:
        """Calculate discount factor to convert values between player perspectives.

        Args:
            - `tree`: MCTSTree containing the nodes
            - `node_idx`: Index of the current node
            - `other_idx`: Index of the node to compare with

        Returns:
            - float: 1.0 if players are same, -1.0 if different

        Optimized: Direct array access instead of data_at() to avoid full node reconstruction.
        """
        # Direct access to current_player array instead of reconstructing entire node
        current_player = tree.data.embedding.current_player[node_idx]
        other_player = tree.data.embedding.current_player[other_idx]

        # Calculate discount factor
        player_diff = jnp.abs(current_player - other_player)
        return 1.0 - 2.0 * player_diff

    def compute_expectimax_value(self, tree: MCTSTree, node_idx: int) -> float:
        """Compute the expected value at a stochastic node using expectimax.

        For stochastic nodes, the value is the weighted sum of child values
        based on their stochastic action probabilities:
            V(node) = Σ P(action) * V(child_action)

        Args:
        - `tree`: MCTSTree containing the nodes
        - `node_idx`: index of the stochastic node

        Returns:
        - float: expected value across all stochastic outcomes
        """
        num_stochastic_actions = len(self.stochastic_action_probs)

        # Get child indices for stochastic actions
        child_indices = tree.edge_map[node_idx, :num_stochastic_actions]

        # Check which children exist
        child_exists_mask = child_indices != tree.NULL_INDEX

        # Safe access to child Q-values
        safe_indices = jnp.maximum(child_indices, 0)
        child_q_values = tree.data.q[safe_indices]

        # Apply mask - use 0 for non-existent children (neutral contribution)
        masked_q_values = jnp.where(child_exists_mask, child_q_values, 0.0)

        # Compute expected value: weighted sum of child Q-values
        # Only include probabilities for children that exist
        masked_probs = jnp.where(child_exists_mask, self.stochastic_action_probs, 0.0)

        # Normalize probabilities to sum to 1 for existing children
        prob_sum = jnp.sum(masked_probs)
        normalized_probs = jnp.where(prob_sum > 0, masked_probs / prob_sum, masked_probs)

        expected_value = jnp.sum(normalized_probs * masked_q_values)

        return expected_value

    @staticmethod
    def _update_node_stats(tree: MCTSTree, node_idx: int, value: float) -> MCTSTree:
        """Update only n and q for a node (optimized partial update).

        OPTIMIZATION: Instead of reconstructing the entire node with data_at() and
        updating all fields including the heavy embedding, we directly update only
        the n and q arrays. This avoids copying the entire game state.

        Args:
        - `tree`: MCTSTree to update
        - `node_idx`: index of the node to update
        - `value`: value to incorporate into running average

        Returns:
        - MCTSTree with updated n and q at node_idx
        """
        # Read current values directly from arrays
        old_n = tree.data.n[node_idx]
        old_q = tree.data.q[node_idx]

        # Compute new values
        new_n = old_n + 1
        new_q = (old_q * old_n + value) / new_n

        # Update only n and q arrays (not the entire node)
        new_data = tree.data.replace(
            n=tree.data.n.at[node_idx].set(new_n),
            q=tree.data.q.at[node_idx].set(new_q)
        )

        return tree.replace(data=new_data)

    def backpropagate(self, key: chex.PRNGKey, tree: MCTSTree, parent: int, child: int, value: float) -> MCTSTree:
        """Backpropagate the value estimate from the leaf node to the root node and update visit counts.

        Uses expectimax for stochastic nodes: the value propagated from a stochastic node
        is the expected value (weighted sum of child values by their probabilities).

        Args:
        - `key`: rng
        - `tree`: MCTSTree to evaluate
        - `parent`: index of the parent node
        - `child`: index of the child node
        - `value`: value estimate of the leaf node

        Returns:
        - (MCTSTree): updated search tree
        """
        # Store reference to self for use in nested function
        compute_expectimax = self.compute_expectimax_value

        def body_fn(state: BackpropState) -> BackpropState:
            node_idx, value, tree = state.node_idx, state.value, state.tree

            # OPTIMIZATION: Use partial update (only n and q) instead of full node reconstruction
            tree = StochasticMCTS._update_node_stats(tree, node_idx, value)

            # go to parent
            parent_idx = tree.parents[node_idx]

            child_and_parent_value_discount = self.calculate_discount_factor(tree, parent_idx, node_idx)

            # For stochastic nodes, use expectimax value instead of sampled value
            # The expectimax value is the weighted sum of all child values
            is_stochastic = StochasticMCTS.is_node_idx_stochastic(tree, node_idx)

            # Use expectimax value for stochastic nodes, sampled value for deterministic
            # OPTIMIZATION: expectimax calculation is now INSIDE the cond, so it only
            # runs for stochastic nodes (lazy evaluation)
            value_to_propagate = jax.lax.cond(
                is_stochastic,
                lambda: compute_expectimax(tree, node_idx),  # Only computed when stochastic
                lambda: value
            )

            value_to_propagate *= child_and_parent_value_discount

            return BackpropState(node_idx=parent_idx, value=value_to_propagate, tree=tree)

        child_and_parent_value_discount = self.calculate_discount_factor(tree, parent, child)

        value *= child_and_parent_value_discount
        state = BackpropState(node_idx=parent, value=value, tree=tree)

        # Backpropagate until we reach the root
        state = jax.lax.while_loop(
            lambda s: s.node_idx != s.tree.ROOT_INDEX,
            body_fn,
            state
        )

        # Update the root node - also use optimized partial update
        is_root_stochastic = StochasticMCTS.is_node_idx_stochastic(state.tree, state.node_idx)
        final_value = jax.lax.cond(
            is_root_stochastic,
            lambda: compute_expectimax(state.tree, state.node_idx),
            lambda: state.value
        )

        tree = StochasticMCTS._update_node_stats(state.tree, state.node_idx, final_value)

        return tree

    @staticmethod
    def is_node_stochastic(node: MCTSNode):
        """Check if the node is stochastic.

        Returns a JAX boolean that can be used with jax.lax.cond.
        """
        # Check if embedding has _is_stochastic attribute, default to False if not
        is_stochastic = getattr(node.embedding, '_is_stochastic', None)
        if is_stochastic is None:
            return jnp.array(False)
        return is_stochastic

    @staticmethod
    def is_node_idx_stochastic(tree: MCTSTree, node_idx: int):
        """Check if the node is stochastic.

        Returns a JAX boolean that can be used with jax.lax.cond.
        Optimized: Direct array access instead of data_at() to avoid full node reconstruction.
        """
        # Direct access to _is_stochastic array instead of reconstructing entire node
        is_stochastic = getattr(tree.data.embedding, '_is_stochastic', None)
        if is_stochastic is None:
            return jnp.array(False)
        return is_stochastic[node_idx]
