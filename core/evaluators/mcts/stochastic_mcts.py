from typing import Tuple
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

    def stochastic_action_sample(self, key: chex.PRNGKey, tree, node_idx, discount):
        """Select an action from the node, not use in exploring at this point so we just use the stochastic action probs.
        i.e. we are rolling the dice to get the next deterministic state.
        
        Args:
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from
        """
        # Properly handle the key for jax.random.choice which expects a single key
        choice_key, _ = jax.random.split(key)
        action = jax.random.choice(choice_key, len(self.stochastic_action_probs), p=self.stochastic_action_probs)
            
        return jnp.array(action, dtype=jnp.int32)
    
    
    def deterministic_action_selector(self, key: chex.PRNGKey, tree: MCTSTree, node_idx: int) -> int:
        """called when traversing the tree and exploring possibilities, a wrapper to remove the key
        
        Args:
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from
        """

        current_player = tree.data_at(node_idx).embedding.current_player
            
        # Get child players
        child_indices = tree.edge_map[node_idx]
        
        # Safe access to child players
        safe_indices = jnp.maximum(child_indices, 0)
        child_players = tree.data.embedding.current_player[safe_indices]
        
        # Create a discount mask for each child based on player difference
        # We want discount = -1.0 when players differ, 1.0 when same
        # The PUCT selector will use this to adjust Q-values
        discount = 1.0  # Default: use raw values (same player)
        
        # Check if ANY child has a different player (simple approach)
        # If so, we need to tell PUCT to apply discount
        # this will hold true for backgammon but isn't generalizable for all games, 
        # would need to have the PUCT selector take a discount arrays for each child.

        has_diff_player = jnp.any(jnp.abs(child_players - current_player))
        discount = jnp.where(has_diff_player, -1.0, 1.0)

        return self.action_selector(tree, node_idx, discount)
    
    def stochastic_action_selector(self, key: chex.PRNGKey, tree, node_idx):
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
    
    

    def cond_action_selector(self, key: chex.PRNGKey, tree: MCTSTree, node_idx: int) -> int:
        """Select an action from the node, picks the right action selector based on the node type.
        
        Args:
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from
        """

        # Create lambda functions that capture all required parameters
        return jax.lax.cond(
            StochasticMCTS.is_node_idx_stochastic(tree, node_idx), 
            lambda k, t, n: self.stochastic_action_selector(k, t, n),
            lambda k, t, n: self.deterministic_action_selector(k, t, n),
            key, tree, node_idx
        )
        
    
    def det_value_policy(self, embedding, params, eval_key, metadata, player_reward) -> Tuple[float, chex.Array]:
        """Get value and policy for deterministic nodes."""
        # For stochastic states, use the same approach as stochastic_value_policy
        is_stochastic = embedding.is_stochastic
        
        # Get policy and value from eval_fn for deterministic states
        policy_logits, value = self.eval_fn(embedding, params, eval_key)
        policy_logits = jnp.where(metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min)
        policy = jax.nn.softmax(policy_logits)
        value = jnp.where(metadata.terminated, player_reward, value)
        
        # For stochastic states, use uniform policy over legal actions
        legal_actions = metadata.action_mask
        num_legal_actions = jnp.sum(legal_actions)
        uniform_prob = jnp.where(num_legal_actions > 0, 
                                 1.0 / num_legal_actions, 
                                 0.0)
        
        # Use uniform policy and 0.0 value for stochastic states
        policy = jnp.where(is_stochastic, uniform_prob, policy)
        value = jnp.where(is_stochastic, 0.0, value)
        
        return value, policy

    def stochastic_value_policy(self, tree, parent_idx, embedding, params, eval_key, metadata, player_reward) -> Tuple[float, chex.Array]:
        """Get value and policy for stochastic nodes.
        
        For stochastic nodes:
        - If parent exists (not root), set value to the parent's Q-value
        - Otherwise use 0.0 as a neutral value
        - Set policy to parent's policy, unless the root node
        """
        # Get parent's Q-value if parent exists (not ROOT_INDEX)
        parent_is_root = parent_idx == tree.NULL_INDEX  # No parent (this is root)
        
        # For root stochastic nodes, use a neutral value (0.0)
        # For non-root stochastic nodes, inherit parent's Q-value
        parent_q = jnp.where(parent_is_root, 
                             0.0,  # Default value if root
                             tree.data_at(parent_idx).q)  # Parent's Q-value
        
        # Use a uniform policy over legal actions instead of all zeros to avoid numerical issues
        legal_actions = metadata.action_mask
        num_legal_actions = jnp.sum(legal_actions)
        uniform_prob = jnp.where(num_legal_actions > 0, 
                                 1.0 / num_legal_actions, 
                                 0.0)  # Handle the case where there are no legal actions
        
        policy = jnp.where(parent_is_root, uniform_prob, tree.data_at(parent_idx).p)
        
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
        # Lets still eval, as a deterministic action was taken to arrive here.
        #value, policy = self.det_value_policy(new_embedding, params, eval_key, metadata, player_reward)
        
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
        """Traverse from the root node until an unvisited leaf node is reached.
        
        Args:
        - `tree`: MCTSTree to evaluate
        
        Returns:
        - (TraversalState): state of the traversal
            - `parent`: index of the parent node
            - `action`: action to take from the parent node
        """
        def body_fn(state: TraversalState) -> TraversalState:
            node_idx = tree.edge_map[state.parent, state.action]
            
            # Get current node's player
            
            # Call action selector with calculated discount
            action = self.cond_action_selector(key, tree, node_idx)
            return TraversalState(parent=node_idx, action=action)
        
        root_action = self.cond_action_selector(key, tree, tree.ROOT_INDEX)
        
        return jax.lax.while_loop(
            lambda s: jnp.logical_and(
                tree.is_edge(s.parent, s.action),
                ~(tree.data_at(tree.edge_map[s.parent, s.action]).terminated)
            ),
            body_fn,
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
        """.
        Samples an action to take from the root based on it's stochastic actions.
        
        Args:
        - `eval_state`: `MCTSTree` to evaluate, could be empty or partially complete

        """
        # Get the action using stochastic_action_selector
        action = self.stochastic_action_sample(key, eval_state, eval_state.ROOT_INDEX, self.discount)
        
        # return root node policy weights
        ## sum the policy weights of the node
        policy_weights = eval_state.data_at(eval_state.ROOT_INDEX).p

        ## I don't think we need to handle the case of where policy_weights_sum is 0
        # policy_weights_sum = jnp.sum(policy_weights)

        # if policy_weights_sum is 0 lets call the eval function to get the policy weights
        # using lax.cond to avoid conditional branches in the graph
        # eval_key, key = jax.random.split(key)
        
        # Create a function to generate policy from eval_fn
        #def generate_policy(_):
        #     policy_logits, _ = self.eval_fn(env_state, params, eval_key)
        #    return jax.nn.softmax(policy_logits)
        
        # policy_weights = jax.lax.cond(
        #     policy_weights_sum == 0,
        #     generate_policy,
        #     lambda _: policy_weights,
        #     None  # Dummy operand, not used directly by either branch
        # )

        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=policy_weights
        )
    
    def calculate_discount_factor(self, tree: MCTSTree, node_idx: int, other_idx: int = None) -> float:
        """Calculate discount factor to convert values between player perspectives.
        
        Args:
            - `tree`: MCTSTree containing the nodes
            - `node_idx`: Index of the current node
            - `other_idx`: Index of the node to compare with
                        
        Returns:
            - float: 1.0 if players are same, -1.0 if different
        """
        current_player = tree.data_at(node_idx).embedding.current_player
        
        # If other_idx is not provided or invalid, return 1.0 (no discount)
        if other_idx is None or other_idx == tree.NULL_INDEX:
            return 1.0
        
        # Get other player
        other_player = tree.data_at(other_idx).embedding.current_player
        
        # Calculate discount factor
        player_diff = jnp.abs(current_player - other_player)
        return 1.0 - 2.0 * player_diff

    def backpropagate(self, key: chex.PRNGKey, tree: MCTSTree, parent: int, value: float) -> MCTSTree:
        """Backpropagate the value estimate from the leaf node to the root node and update visit counts.
        
        Args:
        - `key`: rng
        - `tree`: MCTSTree to evaluate
        - `parent`: index of the parent node
        - `value`: value estimate of the leaf node

        Returns:
        - (MCTSTree): updated search tree
        """
        def body_fn(state: BackpropState) -> BackpropState:
            node_idx, value, tree = state.node_idx, state.value, state.tree
            node = tree.data_at(node_idx)
            
            # Store raw value in current node (from node's own perspective)
            new_node = self.visit_node(node, value)
            tree = tree.update_node(node_idx, new_node)
            
            # Apply discount ONLY for passing value to parent
            # This adjusts the value to the parent's perspective
            parent_idx = tree.parents[node_idx]
            
            # Get parent player safely (parent might be NULL_INDEX)
            safe_parent_idx = jnp.maximum(parent_idx, 0)  # Use 0 as safe index if parent is NULL
            is_valid_parent = parent_idx != tree.NULL_INDEX
            
            # Get player info
            current_player = node.embedding.current_player
            parent_player = tree.data_at(safe_parent_idx).embedding.current_player
            
            # Apply discount when players differ and parent is valid
            player_diff = jnp.abs(current_player - parent_player)
            discount_factor = 1.0 - 2.0 * player_diff
            
            # Only apply discount if parent is valid
            value = jnp.where(is_valid_parent, value * discount_factor, value)
            
            return BackpropState(node_idx=parent_idx, value=value, tree=tree)
        
        state = jax.lax.while_loop(
            lambda s: s.node_idx != s.tree.NULL_INDEX, 
            body_fn, 
            BackpropState(node_idx=parent, value=value, tree=tree)
        )
        return state.tree


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