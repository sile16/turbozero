from typing import Tuple
import jax
import jax.numpy as jnp
import chex
from chex import dataclass
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.state import BackpropState, MCTSNode, MCTSTree, TraversalState, MCTSOutput
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.alphazero import AlphaZero
from core.types import EnvStepFn, EvalFn, StepMetadata


@dataclass(frozen=True)
class ExpandResult:
    """Result of expanding child node(s)."""
    tree: MCTSTree
    child_idx: int
    value: float


class StochasticMCTS(AlphaZero(MCTS)):
    """MCTS for games with stochastic transitions (e.g., Pig, backgammon).

    This class handles games where `is_stochastic=True` indicates a CHANCE NODE - a state
    where the environment determines the next outcome randomly, not the player.

    IMPORTANT: Your `env_step_fn` must handle both decision and chance nodes:
    - For decision nodes (is_stochastic=False): call `env.step(state, action, key)`
    - For chance nodes (is_stochastic=True): call `env.stochastic_step(state, outcome)`

    Example for pgx Pig (JAX-compatible):
    ```python
    def make_pig_step_fn(env):
        def step_fn(state, action, key):
            is_stochastic = getattr(state, '_is_stochastic', jnp.array(False))
            new_state = jax.lax.cond(
                is_stochastic,
                lambda _: env.stochastic_step(state, action),  # Chance node
                lambda _: env.step(state, action, key),        # Decision node
                operand=None
            )
            metadata = StepMetadata(...)
            return new_state, metadata
        return step_fn
    ```

    When the root node is a chance node (is_stochastic=True), this class:
    - Skips MCTS search (no player decision to make)
    - Samples an action from `stochastic_action_probs`
    - Returns a dummy policy (zeros) - not used for training at chance nodes

    When traversing the tree:
    - At decision nodes: uses neural network policy with PUCT selection
    - At chance nodes: explores outcomes proportionally to `stochastic_action_probs`
    - Backpropagation uses expectimax at chance nodes (weighted average by probability)

    Compatible with `jax.vmap`, `jax.pmap`, `jax.jit`, etc."""
    def __init__(self,
        eval_fn: EvalFn,
        action_selector: MCTSActionSelector,
        stochastic_action_probs: chex.Array,
        branching_factor: int,
        max_nodes: int,
        num_iterations: int,
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
        - `temperature`: temperature for root action selection (default: 1.0)
        - `tiebreak_noise`: magnitude of noise to add to policy weights for breaking ties (default: 1e-8)
        - `persist_tree`: whether to persist search tree state between calls to `evaluate` (default: True)
        - `noise_scale`: scale of noise to add to delta values in stochastic action selection (default: 0.05)
        - `dirichlet_alpha`: magnitude of Dirichlet noise for AlphaZero (default: 0.3)
        - `dirichlet_epsilon`: proportion of root policy composed of Dirichlet noise (default: 0.25)
        """
        super().__init__(dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon, eval_fn=eval_fn, action_selector=action_selector, branching_factor=branching_factor, max_nodes=max_nodes, num_iterations=num_iterations, temperature=temperature, tiebreak_noise=tiebreak_noise, persist_tree=persist_tree)

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
        is_root_stochastic = StochasticMCTS.is_node_idx_stochastic(tree, tree.ROOT_INDEX)

        # Compute both outputs and select based on root stochastic status
        # This avoids jax.lax.cond which can cause tracing issues with vmap
        stochastic_output = self.stochastic_evaluate(key, eval_state, env_state, root_metadata, params, env_step_fn)
        standard_output = super().evaluate(key, eval_state, env_state, root_metadata, params, env_step_fn)

        # Select output using tree_map with where
        return jax.tree_util.tree_map(
            lambda a, b: jnp.where(is_root_stochastic, a, b),
            stochastic_output,
            standard_output
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
        # Reset tree if persist_tree=False to clear stale child nodes
        key, root_key = jax.random.split(key)

        # OPTIMIZATION: Direct array access instead of data_at() which reconstructs entire node
        root_n = eval_state.data.n[eval_state.ROOT_INDEX]
        should_reset = jnp.logical_and(
            not self.persist_tree,
            root_n > 0  # Only reset if tree has been used
        )
        eval_state = jax.lax.cond(
            should_reset,
            lambda: eval_state.reset(),
            lambda: eval_state
        )

        # Update root if tree is empty (after reset or first call) or persist_tree=False
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
        action = self.stochastic_action_sample(key, eval_state, eval_state.ROOT_INDEX)
        
        # Create a dummy policy with the correct shape (branching_factor)
        # This will be ignored during training since we're in a stochastic state
        policy_weights = jnp.zeros(self.branching_factor)
        
        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=policy_weights
        )
    

    def stochastic_action_sample(self, key: chex.PRNGKey, tree, node_idx):
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
        """Called when traversing the tree and exploring possibilities, a wrapper to remove the key.

        Args:
        - `key`: rng key (unused for deterministic action selection)
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from

        Note: The action_selector dynamically calculates the discount by comparing
        parent's current_player with each child's current_player, so no explicit
        discount parameter is needed.
        """
        return self.action_selector(tree, node_idx)
    
    def stochastic_action_selector(self, key: chex.PRNGKey, tree, node_idx) -> int:
        """Called when traversing the tree and exploring possibilities.

        Expansion strategy:
        - If any stochastic outcomes have not been expanded yet at this node, expand them first
          (highest-probability unexpanded outcome, with small noise for tie-breaking).
        - Once all outcomes exist, select the action with the largest discrepancy between the child's
          observed visit frequency and its theoretical stochastic probability (probability-matching).

        Selects the action with the largest discrepancy between the child's observed
        visit frequency and its theoretical stochastic probability.  This works when probabilities
        are fairly similar and there isn't some very unlikely action that results in a large reward.

        Args:
        - `key`: rng
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from

        Returns:
        - The action with the largest delta between observed visit frequency and stochastic probability
        """

        num_stochastic_actions = len(self.stochastic_action_probs)

        # Get the mapping from edge indices to child node indices, but only for stochastic actions
        child_indices = tree.edge_map[node_idx, :num_stochastic_actions]

        # Create a mask for edges that exist (not NULL_INDEX)
        child_exists_mask = child_indices != tree.NULL_INDEX
        unexpanded_mask = ~child_exists_mask
        has_unexpanded = jnp.any(unexpanded_mask)

        # Split key for both branches
        k1, k2 = jax.random.split(key)

        # Compute expand_unexpanded action
        # Prefer higher-probability outcomes first (reduces early expectimax bias)
        masked_probs_expand = jnp.where(unexpanded_mask, self.stochastic_action_probs, -jnp.inf)
        _, noise_key1 = jax.random.split(k1)
        masked_probs_expand = masked_probs_expand + jax.random.normal(noise_key1, masked_probs_expand.shape) * self.noise_scale
        expand_action = jnp.argmax(masked_probs_expand)

        # Compute probability_match action
        # For vectorized child visit count retrieval, we need to handle NULL_INDEX
        safe_indices = jnp.maximum(child_indices, 0)  # Replace -1 with 0 for safe access

        # Vectorized access to visit counts
        all_n = tree.data.n[safe_indices]

        # Apply mask to zero out non-existent children
        child_visits = jnp.where(child_exists_mask, all_n, 0)

        # Calculate total visits
        total_visits = jnp.sum(child_visits)

        # Normalize visit counts; handle the case where total_visits=0
        normalized_visits = jnp.where(
            total_visits > 0,
            child_visits / total_visits,
            jnp.ones_like(child_visits, dtype=jnp.float32) / num_stochastic_actions
        )

        # Calculate deltas between actual visit frequencies and theoretical probabilities
        delta = self.stochastic_action_probs - normalized_visits

        # Add some noise to break ties
        _, noise_key2 = jax.random.split(k2)
        delta = delta + jax.random.normal(noise_key2, delta.shape) * self.noise_scale

        # Return the action with the biggest delta
        match_action = jnp.argmax(delta)

        # Select between expand and match based on has_unexpanded
        return jnp.where(has_unexpanded, expand_action, match_action)
    
    

    def cond_action_selector(self, key: chex.PRNGKey, tree: MCTSTree, node_idx: int) -> int:
        """Select an action from the node, picks the right action selector based on the node type.
        We can't just override the action_selector in the base class as we need to pass in the key.

        Args:
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from
        """
        is_stochastic = StochasticMCTS.is_node_idx_stochastic(tree, node_idx)

        # Compute both actions and select based on node type
        # This avoids nested conds which can cause tracing issues with vmap
        stochastic_action = self.stochastic_action_selector(key, tree, node_idx)
        deterministic_action = self.deterministic_action_selector(key, tree, node_idx)

        return jnp.where(is_stochastic, stochastic_action, deterministic_action)
    
    
    def value_policy(self, embedding, params, eval_key, metadata, player_reward) -> Tuple[float, chex.Array]:
        """Get value and policy for deterministic nodes."""
        # Get policy and value from eval_fn for deterministic states
        policy_logits, value = self.eval_fn(embedding, params, eval_key)
        policy_logits = jnp.where(metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min)
        policy = jax.nn.softmax(policy_logits)
        value = jnp.where(metadata.terminated, player_reward, value)
        
        return value, policy


    def iterate(self, key: chex.PRNGKey, tree: MCTSTree, params: chex.ArrayTree, env_step_fn: EnvStepFn) -> MCTSTree:
        """Performs one iteration of MCTS with stochastic support.

        Key design: Check stochasticity ONCE after traverse, then branch to either:
        - Single child expansion (deterministic parent)
        - All children expansion (stochastic parent with unexpanded children)

        For vmap compatibility, we compute both paths and use jnp.where to select.
        This ensures uniform computation across the batch.
        """
        traverse_key, expand_key, backprop_key = jax.random.split(key, 3)

        # Traverse from root to leaf
        traversal_state = self.traverse(traverse_key, tree)
        parent, action = traversal_state.parent, traversal_state.action

        # Check ONCE if parent is stochastic
        is_parent_stochastic = StochasticMCTS.is_node_idx_stochastic(tree, parent)

        # Check if all stochastic children already exist
        all_children_exist = self._all_stochastic_children_exist(tree, parent)

        # We need expand_all when: parent is stochastic AND not all children exist yet
        needs_expand_all = jnp.logical_and(is_parent_stochastic, ~all_children_exist)

        # Compute BOTH expansion strategies (for vmap compatibility)
        # JAX will trace both paths but only execute the selected one at runtime
        single_result = self._expand_single_child(expand_key, tree, parent, action, params, env_step_fn)
        all_result = self._expand_all_stochastic_children(expand_key, tree, parent, params, env_step_fn)

        # Select result based on whether we need to expand all stochastic children
        final_tree = jax.tree_util.tree_map(
            lambda a, b: jnp.where(needs_expand_all, a, b),
            all_result.tree, single_result.tree
        )
        final_value = jnp.where(needs_expand_all, all_result.value, single_result.value)
        final_child = jnp.where(needs_expand_all, all_result.child_idx, single_result.child_idx)

        # Backpropagate from the expanded child
        return self.backpropagate(backprop_key, final_tree, parent, final_child, final_value)

    def _all_stochastic_children_exist(self, tree: MCTSTree, node_idx: int) -> bool:
        """Check if all stochastic outcome children exist for a node."""
        num_stochastic = len(self.stochastic_action_probs)
        child_indices = tree.edge_map[node_idx, :num_stochastic]
        return jnp.all(child_indices != tree.NULL_INDEX)

    def _expand_single_child(self, key: chex.PRNGKey, tree: MCTSTree, parent: int, action: int,
                              params: chex.ArrayTree, env_step_fn: EnvStepFn) -> ExpandResult:
        """Expand a single child node (standard MCTS expansion)."""
        step_key, eval_key = jax.random.split(key)

        # Get parent embedding and step
        embedding = jax.tree_util.tree_map(lambda x: x[parent], tree.data.embedding)
        new_embedding, metadata = env_step_fn(embedding, action, step_key)
        player_reward = metadata.rewards[metadata.cur_player_id]

        value, policy = self.value_policy(new_embedding, params, eval_key, metadata, player_reward)

        node_exists = tree.is_edge(parent, action)
        node_idx = tree.edge_map[parent, action]

        def update_existing_node():
            return StochasticMCTS._update_node_stats(tree, node_idx, value)

        def add_new_node():
            node_data = self.new_node(policy=policy, value=value, embedding=new_embedding, terminated=metadata.terminated)
            return tree.add_node(parent_index=parent, edge_index=action, data=node_data)

        updated_tree = jax.lax.cond(node_exists, update_existing_node, add_new_node)
        child_idx = updated_tree.edge_map[parent, action]

        return ExpandResult(tree=updated_tree, child_idx=child_idx, value=value)

    def _expand_all_stochastic_children(self, key: chex.PRNGKey, tree: MCTSTree, parent: int,
                                         params: chex.ArrayTree, env_step_fn: EnvStepFn) -> ExpandResult:
        """Expand ALL stochastic outcome children at once.

        This ensures proper expectimax values from the first backpropagation by having
        all children available for the weighted average computation.

        Uses jax.lax.scan for vmap compatibility - no dynamic branching.
        """
        num_stochastic = len(self.stochastic_action_probs)

        # Get parent embedding
        parent_embedding = jax.tree_util.tree_map(lambda x: x[parent], tree.data.embedding)

        # Generate keys for each child
        keys = jax.random.split(key, num_stochastic + 1)
        child_keys = keys[1:]  # One key per stochastic outcome

        # Expand all children using scan (sequential but traceable)
        def expand_one_child(carry, inputs):
            tree, parent_emb = carry
            action, child_key = inputs
            step_key, eval_key = jax.random.split(child_key)

            # Step the environment
            new_embedding, metadata = env_step_fn(parent_emb, action, step_key)
            player_reward = metadata.rewards[metadata.cur_player_id]

            # Get value and policy
            value, policy = self.value_policy(new_embedding, params, eval_key, metadata, player_reward)

            # Check if child already exists
            child_exists = tree.is_edge(parent, action)
            existing_idx = tree.edge_map[parent, action]

            # Create node data
            node_data = self.new_node(policy=policy, value=value, embedding=new_embedding, terminated=metadata.terminated)

            # Add node or update existing - use jnp.where to avoid cond
            # If child exists, update stats; if not, add new node
            tree_with_new = tree.add_node(parent_index=parent, edge_index=action, data=node_data)
            tree_updated = StochasticMCTS._update_node_stats(tree, existing_idx, value)

            # Select which tree to use based on whether child exists
            new_tree = jax.tree_util.tree_map(
                lambda a, b: jnp.where(child_exists, a, b),
                tree_updated, tree_with_new
            )

            return (new_tree, parent_emb), value

        # Create action indices
        actions = jnp.arange(num_stochastic, dtype=jnp.int32)

        # Run scan over all stochastic actions
        (final_tree, _), child_values = jax.lax.scan(
            expand_one_child,
            (tree, parent_embedding),
            (actions, child_keys)
        )

        # Compute expectimax value across all children
        expectimax_value = jnp.sum(self.stochastic_action_probs * child_values)

        # Return first child index (all children exist now, backprop will use expectimax)
        first_child_idx = final_tree.edge_map[parent, 0]

        return ExpandResult(tree=final_tree, child_idx=first_child_idx, value=expectimax_value)


    def traverse(self, key: chex.PRNGKey, tree: MCTSTree) -> TraversalState:
        """Traverse from the root node until an unvisited leaf node is reached.
        Uses different action selectors for stochastic and deterministic nodes.
        """
        # Split key for root action and body iterations
        root_key, body_key = jax.random.split(key)

        def cond_fn(state: TraversalState) -> bool:
            child_idx = tree.edge_map[state.parent, state.action]
            safe_child_idx = jnp.maximum(child_idx, 0)
            child_terminated = tree.data.terminated[safe_child_idx]
            return jnp.logical_and(
                tree.is_edge(state.parent, state.action),
                ~child_terminated
            )

        def body_fn(state: TraversalState) -> TraversalState:
            node_idx = tree.edge_map[state.parent, state.action]
            iteration_key = jax.random.fold_in(body_key, node_idx * tree.branching_factor + state.action)
            action = self.cond_action_selector(iteration_key, tree, node_idx)
            return TraversalState(parent=node_idx, action=action)

        root_action = self.cond_action_selector(root_key, tree, tree.ROOT_INDEX)
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
        """Backpropagate with expectimax for stochastic nodes."""
        compute_expectimax = self.compute_expectimax_value
        root_index = tree.ROOT_INDEX

        def body_fn(state: BackpropState) -> BackpropState:
            node_idx, value, tree = state.node_idx, state.value, state.tree
            tree = StochasticMCTS._update_node_stats(tree, node_idx, value)
            parent_idx = tree.parents[node_idx]
            child_and_parent_value_discount = self.calculate_discount_factor(tree, parent_idx, node_idx)
            is_stochastic = StochasticMCTS.is_node_idx_stochastic(tree, node_idx)
            expectimax_value = compute_expectimax(tree, node_idx)
            value_to_propagate = jnp.where(is_stochastic, expectimax_value, value)
            value_to_propagate *= child_and_parent_value_discount
            return BackpropState(node_idx=parent_idx, value=value_to_propagate, tree=tree)

        child_and_parent_value_discount = self.calculate_discount_factor(tree, parent, child)
        value *= child_and_parent_value_discount
        state = BackpropState(node_idx=parent, value=value, tree=tree)

        state = jax.lax.while_loop(
            lambda s: s.node_idx != root_index,
            body_fn,
            state
        )

        is_root_stochastic = StochasticMCTS.is_node_idx_stochastic(state.tree, state.node_idx)
        expectimax_root = compute_expectimax(state.tree, state.node_idx)
        final_value = jnp.where(is_root_stochastic, expectimax_root, state.value)
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
