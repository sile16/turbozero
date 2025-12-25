from functools import partial
from typing import Dict, Optional, Tuple
import jax
import chex
import jax.numpy as jnp
from core.evaluators.evaluator import Evaluator
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.state import BackpropState, MCTSNode, MCTSTree, TraversalState, MCTSOutput
from core.trees.tree import init_tree
from core.types import EnvStepFn, EvalFn, StepMetadata

class MCTS(Evaluator):
    """Batched implementation of Monte Carlo Tree Search (MCTS).

    Not stateful. This class operates on 'MCTSTree' state objects.

    Compatible with `jax.vmap`, `jax.pmap`, `jax.jit`, etc."""
    def __init__(self,
        eval_fn: EvalFn,
        action_selector: MCTSActionSelector,
        branching_factor: int,
        max_nodes: int,
        num_iterations: int,
        temperature: float = 1.0,
        tiebreak_noise: float = 1e-8,
        persist_tree: bool = True
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
        """
        super().__init__()
        self.eval_fn = eval_fn
        self.num_iterations = num_iterations
        self.branching_factor = branching_factor
        self.max_nodes = max_nodes
        self.action_selector = action_selector
        self.temperature = temperature
        self.tiebreak_noise = tiebreak_noise
        self.persist_tree = persist_tree


    def get_config(self) -> Dict:
        """returns a config object for checkpoints"""
        return {
            "eval_fn": self.eval_fn.__name__,
            "num_iterations": self.num_iterations,
            "branching_factor": self.branching_factor,
            "max_nodes": self.max_nodes,
            "action_selection_config": self.action_selector.get_config(),
            "temperature": self.temperature,
            "tiebreak_noise": self.tiebreak_noise,
            "persist_tree": self.persist_tree
        }


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
        # Reset tree if persist_tree=False to clear stale child nodes
        # This is critical: without reset, child nodes from previous states persist
        # and can cause MCTS to explore illegal actions
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

        # perform 'num_iterations' iterations of MCTS
        # OPTIMIZATION: Use fori_loop with key carry instead of pre-allocating all keys
        # This avoids O(num_iterations) key array allocation and reduces memory/compile time
        iterate_fn = partial(self.iterate, params=params, env_step_fn=env_step_fn)

        def scan_body(carry, _):
            state, k = carry
            k, iter_key = jax.random.split(k)
            new_state = iterate_fn(iter_key, state)
            return (new_state, k), None

        (eval_state, _), _ = jax.lax.scan(scan_body, (eval_state, key), None, length=self.num_iterations)

        # sample action based on root visit counts
        # (also get normalized policy weights for training purposes)
        # Pass action_mask to ensure policy_weights only have mass on legal actions
        action, policy_weights = self.sample_root_action(key, eval_state, action_mask=root_metadata.action_mask)
        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=policy_weights
        )
    

    def get_value(self, state: MCTSTree) -> chex.Array:
        """Returns value estimate of the environment state stored in the root node of the tree.

        Args:
        - `state`: MCTSTree to evaluate

        Returns:
        - (chex.Array): value estimate of the environment state stored in the root node of the tree
        """
        # OPTIMIZATION: Direct array access instead of data_at()
        return state.data.q[state.ROOT_INDEX]
    

    def update_root(self, key: chex.PRNGKey, tree: MCTSTree, root_embedding: chex.ArrayTree, 
                    params: chex.ArrayTree, **kwargs) -> MCTSTree: #pylint: disable=unused-argument
        """Populates the root node of an MCTSTree.
        
        Args:
        - `key`: rng
        - `tree`: MCTSTree to update
        - `root_embedding`: root environment state
        - `params`: nn parameters

        Returns:
        - (MCTSTree): updated MCTSTree
        """
        # evaluate root state
        root_policy_logits, root_value = self.eval_fn(root_embedding, params, key)
        root_policy = jax.nn.softmax(root_policy_logits)
        # update root node
        root_node = tree.data_at(tree.ROOT_INDEX)
        root_node = self.update_root_node(root_node, root_policy, root_value, root_embedding)
        return tree.set_root(root_node)
    
    
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
        # Get Keys:
        step_key, eval_key, key = jax.random.split(key, 3)

        # traverse from root -> leaf
        traversal_state = self.traverse(tree)
        parent, action = traversal_state.parent, traversal_state.action
        # get env state (embedding) for leaf node
        # OPTIMIZATION: Direct access to embedding instead of data_at() which reconstructs entire node
        embedding = jax.tree_util.tree_map(lambda x: x[parent], tree.data.embedding)
        new_embedding, metadata = env_step_fn(embedding, action, step_key)
        player_reward = metadata.rewards[metadata.cur_player_id]
        # evaluate leaf node

        policy_logits, value = self.eval_fn(new_embedding, params, eval_key)
        policy_logits = jnp.where(metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min)
        policy = jax.nn.softmax(policy_logits)
        value = jnp.where(metadata.terminated, player_reward, value)
        # add leaf node to tree
        node_exists = tree.is_edge(parent, action)
        node_idx = tree.edge_map[parent, action]

        # OPTIMIZATION: For existing nodes, only update n/q stats (not the full node with embedding)
        # For new nodes, create the full node and add to tree
        def update_existing_node():
            return MCTS._update_node_stats(tree, node_idx, value)

        def add_new_node():
            node_data = self.new_node(policy=policy, value=value, embedding=new_embedding, terminated=metadata.terminated)
            return tree.add_node(parent_index=parent, edge_index=action, data=node_data)

        tree = jax.lax.cond(node_exists, update_existing_node, add_new_node)

        # Get child node index (either existing or newly added)
        child_idx = tree.edge_map[parent, action]

        # backpropagate
        return self.backpropagate(key, tree, parent, child_idx, value)


    def traverse(self, tree: MCTSTree) -> TraversalState:
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
            action = self.action_selector(tree, node_idx)
            return TraversalState(parent=node_idx, action=action)

        # choose the action to take from the root
        root_action = self.action_selector(tree, tree.ROOT_INDEX)
        # traverse from root to leaf
        return jax.lax.while_loop(
            cond_fn, body_fn, 
            TraversalState(parent=tree.ROOT_INDEX, action=root_action)
        )


    @staticmethod
    def calculate_discount_factor(tree: MCTSTree, node_idx: int, other_idx: int) -> float:
        """Calculate discount factor to convert values between player perspectives.

        Args:
            - `tree`: MCTSTree containing the nodes
            - `node_idx`: Index of the current node
            - `other_idx`: Index of the node to compare with

        Returns:
            - float: 1.0 if players are same, -1.0 if different

        This handles multi-action-per-turn games correctly by only flipping
        the sign when the current player actually changes between nodes.
        """
        # Direct access to current_player array
        current_player = tree.data.embedding.current_player[node_idx]
        other_player = tree.data.embedding.current_player[other_idx]

        # Calculate discount factor: 1.0 if same player, -1.0 if different
        player_diff = jnp.abs(current_player - other_player)
        return 1.0 - 2.0 * player_diff

    def backpropagate(self, key: chex.PRNGKey, tree: MCTSTree, parent: int, child: int, value: float) -> MCTSTree: #pylint: disable=unused-argument
        """Backpropagate the value estimate from the leaf node to the root node and update visit counts.

        Args:
        - `key`: rng
        - `tree`: MCTSTree to evaluate
        - `parent`: index of the parent node
        - `child`: index of the child node (the newly expanded node)
        - `value`: value estimate of the leaf node

        Returns:
        - (MCTSTree): updated search tree
        """

        def body_fn(state: BackpropState) -> BackpropState:
            node_idx, value, tree = state.node_idx, state.value, state.tree

            # Update current node with current value
            tree = MCTS._update_node_stats(tree, node_idx, value)

            # Get parent index
            parent_idx = tree.parents[node_idx]

            # Calculate discount based on whether player changed between parent and current
            # This correctly handles multi-action-per-turn games (e.g., multiple rolls in Pig)
            discount = MCTS.calculate_discount_factor(tree, parent_idx, node_idx)
            value *= discount

            return BackpropState(node_idx=parent_idx, value=value, tree=tree)

        # Calculate initial discount from parent to child
        initial_discount = MCTS.calculate_discount_factor(tree, parent, child)
        value *= initial_discount

        # backpropagate while the node is a valid node
        # the root has no parent, so the loop will terminate
        # when the parent of the root is visited
        state = jax.lax.while_loop(
            lambda s: s.node_idx != s.tree.NULL_INDEX, body_fn,
            BackpropState(node_idx=parent, value=value, tree=tree)
        )
        return state.tree


    def sample_root_action(self, key: chex.PRNGKey, tree: MCTSTree, action_mask: Optional[chex.Array] = None) -> Tuple[int, chex.Array]:
        """Sample an action based on the root visit counts.

        Args:
        - `key`: rng
        - `tree`: MCTSTree to evaluate
        - `action_mask`: optional mask of legal actions (True=legal, False=illegal)

        Returns:
        - (Tuple[int, chex.Array]): sampled action, normalized policy weights
        """
        # get root visit counts
        action_visits = tree.get_child_data('n', tree.ROOT_INDEX)

        # CRITICAL FIX: Zero out visit counts for illegal actions
        # When persist_tree=True, after re-rooting via get_subtree(), the tree may
        # contain visit counts from exploring children that were legal in a PREVIOUS
        # game state but are illegal in the CURRENT state. We must mask these out
        # to avoid putting probability mass on illegal actions in policy_weights.
        if action_mask is not None:
            # Handle shape mismatch: mask may be smaller than action_visits (stochastic games)
            mask_size = action_mask.shape[-1]
            visits_size = action_visits.shape[-1]
            if mask_size < visits_size:
                # Pad mask with False (illegal) for extra actions
                padded_mask = jnp.zeros(visits_size, dtype=bool)
                padded_mask = padded_mask.at[:mask_size].set(action_mask)
                action_mask = padded_mask
            action_visits = jnp.where(action_mask, action_visits, 0)

        # normalize visit counts to get policy weights
        total_visits = action_visits.sum(axis=-1)
        policy_weights = action_visits / jnp.maximum(total_visits, 1)
        policy_weights = jnp.where(total_visits > 0, policy_weights, 1 / self.branching_factor)

        # zero temperature == argmax
        if self.temperature == 0:
            # break ties by adding small amount of noise
            noise = jax.random.uniform(key, shape=policy_weights.shape, maxval=self.tiebreak_noise)
            noisy_policy_weights = policy_weights + noise
            return jnp.argmax(noisy_policy_weights), policy_weights
        
        # apply temperature 
        policy_weights_t = policy_weights ** (1/self.temperature)
        # re-normalize 
        policy_weights_t /= policy_weights_t.sum()
        # sample action
        action = jax.random.choice(key, policy_weights_t.shape[-1], p=policy_weights_t)
        # return original policy weights (we train on the policy before temperature is applied)
        return action, policy_weights


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
        q_value = ((node.q * node.n) + value) / (node.n + 1)
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

    @staticmethod
    def new_node(policy: chex.Array, value: float, embedding: chex.ArrayTree, terminated: bool) -> MCTSNode:
        """Create a new MCTSNode.
        
        Args:
        - `policy`: policy weights
        - `value`: value estimate
        - `embedding`: environment state embedding
            - 'embedding' because in some MCTS use-cases, e.g. MuZero, we store an embedding of the state 
               rather than the state itself. In AlphaZero, this is just the entire environment state.
        - `terminated`: whether the state is terminal

        Returns:
        - (MCTSNode): initialized MCTSNode
        """
        return MCTSNode(
            n=jnp.array(1, dtype=jnp.int32), # init visit count to 1
            p=policy,
            q=jnp.array(value, dtype=jnp.float32),
            terminated=jnp.array(terminated, dtype=jnp.bool_),
            embedding=embedding
        )
    

    @staticmethod
    def update_root_node(root_node: MCTSNode, root_policy: chex.Array, root_value: float, root_embedding: chex.ArrayTree) -> MCTSNode:
        """Update the root node of the search tree.
        
        Args:
        - `root_node`: node to update
        - `root_policy`: policy weights
        - `root_value`: value estimate
        - `root_embedding`: environment state embedding
        
        Returns:
        - (MCTSNode): updated root node
        """
        visited = root_node.n > 0
        # For arrays (policy, value, visit count), use jnp.where
        new_policy = jnp.where(visited, root_node.p, root_policy)
        new_value = jnp.where(visited, root_node.q, root_value)
        new_visits = jnp.where(visited, root_node.n, 1)
        
        # For the embedding, use jax.lax.cond to handle the state object
        new_embedding = jax.lax.cond(
            visited,
            lambda _: root_node.embedding,
            lambda _: root_embedding,
            operand=None
        )
        
        return root_node.replace(
            p=new_policy,
            q=new_value,
            n=new_visits,
            embedding=new_embedding
        )
    

    def reset(self, state: MCTSTree) -> MCTSTree:
        """Resets the internal state of MCTS.
        
        Args:
        - `state`: evaluator state

        Returns:
        - (MCTSTree): reset evaluator state
        """
        return state.reset()


    def step(self, state: MCTSTree, action: int) -> MCTSTree:
        """Update the internal state of MCTS after taking an action in the environment.
        
        Args:
        - `state`: evaluator state
        - `action`: action taken in the environment
        
        Returns:
        - (MCTSTree): updated evaluator state
        """
        
        if self.persist_tree:
            # get subtree corresponding to action taken if persist_tree is True
            return state.get_subtree(action)
        # just reset to an empty tree if persist_tree is False
        return state.reset()


    def get_tree_stats(self, tree_state: MCTSTree) -> Dict:
        """Extract statistics from the MCTS tree. JAX-compatible, no loops or conditionals.
        
        Args:
        - `tree_state`: MCTS tree state
        
        Returns:
        - Dict: Tree statistics
        """
        # Get the root node
        root_node = tree_state.data_at(tree_state.ROOT_INDEX)
        
        # Basic tree statistics
        stats = {
            "mcts/tree_size": tree_state.next_free_idx,
            "mcts/root_visits": root_node.n,
        }
        
        # Child statistics using vectorized operations
        child_visits = tree_state.get_child_data('n', tree_state.ROOT_INDEX)
        total_child_visits = jnp.sum(child_visits)
        
        # Calculate statistics without loops or conditionals
        stats["mcts/exploration_pct"] = (
            100.0 * (1.0 - jnp.max(child_visits) / jnp.maximum(total_child_visits, 1.0))
        )
        
        # Root node value
        stats["mcts/root_q_value"] = root_node.q
        
        return stats
    
    def init(self, template_embedding: chex.ArrayTree, *args, **kwargs) -> MCTSTree: #pylint: disable=arguments-differ
        """Initializes the internal state of the MCTS evaluator.
        
        Args:
        - `template_embedding`: template environment state embedding
            - not stored, just used to initialize data structures to the correct shape

        Returns:
        - (MCTSTree): initialized MCTSTree
        """
        return init_tree(self.max_nodes, self.branching_factor, self.new_node(
            policy=jnp.zeros((self.branching_factor,)),
            value=0.0,
            embedding=template_embedding,
            terminated=False
        ))
