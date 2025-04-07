from functools import partial
from typing import Dict, Optional, Tuple
import jax
import chex
import jax.numpy as jnp
import datetime
import time  # Added for debugging
from core.evaluators.evaluator import Evaluator
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.state import BackpropState, MCTSNode, MCTSTree, TraversalState, MCTSOutput
from core.trees.tree import init_tree
from core.types import EnvStepFn, EvalFn, StepMetadata
from jax.debug import print as dprint

# Import field from flax.struct if MCTSTree uses it, else from dataclasses
# Assuming flax.struct based on typical JAX usage
from flax import struct

# Add debug level control
DEBUG_LEVEL = 0  # 0 = minimal debug output, 1 = important messages, 2 = verbose
# Add a debug print function
def debug_print(msg, *args, level=1, **kwargs):
    pass
    # if level <= DEBUG_LEVEL:
    #     time_stamp = time.time()
    #     print(f"[DEBUG-STOCHASTIC_MCTS-{time_stamp}] {msg}", *args, **kwargs)

@struct.dataclass
class StochasticMCTSTree(MCTSTree):
    """Extends MCTSTree to track stochastic nodes."""
    node_is_stochastic: chex.Array

    # If __post_init__ is used in MCTSTree for initialization, 
    # we might need to override or extend it here. 
    # However, initialization often happens outside the class itself.
    # We will handle initialization where the tree is created.


class StochasticMCTS(Evaluator):
    """Batched implementation of Monte Carlo Tree Search (MCTS) supporting stochastic nodes.
    
    Not stateful. This class operates on 'StochasticMCTSTree' state objects.
    
    Compatible with `jax.vmap`, `jax.pmap`, `jax.jit`, etc."""
    def __init__(self,
        eval_fn: EvalFn,
        action_selector: MCTSActionSelector,
        branching_factor: int,
        max_nodes: int,
        num_iterations: int,
        stochastic_action_probs: chex.Array,
        discount: float = -1.0,
        temperature: float = 1.0,
        tiebreak_noise: float = 1e-8,
        persist_tree: bool = True,
        debug_level: int = 0,
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
        - `debug_level`: controls verbosity of debug output (0=minimal, 1=important, 2=verbose)
        """
        super().__init__(discount=discount)
        self.eval_fn = eval_fn
        self.num_iterations = num_iterations
        self.branching_factor = branching_factor
        self.max_nodes = max_nodes
        self.action_selector = action_selector
        self.temperature = temperature
        self.tiebreak_noise = tiebreak_noise
        self.persist_tree = persist_tree
        self.stochastic_action_probs = stochastic_action_probs
        
        # Set global debug level
        global DEBUG_LEVEL
        DEBUG_LEVEL = debug_level
        
        debug_print(f"StochasticMCTS initialized with num_iterations={num_iterations}, max_nodes={max_nodes}", level=1)


    def get_config(self) -> Dict:
        """returns a config object for checkpoints"""
        return {
            "eval_fn": self.eval_fn.__name__,
            "num_iterations": self.num_iterations,
            "branching_factor": self.branching_factor,
            "max_nodes": self.max_nodes,
            "action_selection_config": self.action_selector.get_config(),
            "discount": self.discount,
            "temperature": self.temperature,
            "tiebreak_noise": self.tiebreak_noise,
            "persist_tree": self.persist_tree
        }


    def evaluate(self, #pylint: disable=arguments-differ
        key: chex.PRNGKey,
        eval_state: StochasticMCTSTree, 
        env_state: chex.ArrayTree, # This is the potentially stochastic state
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
        env_step_fn: EnvStepFn, # Make sure this is passed
        stochastic_action_probs: Optional[chex.Array] = None, # Pass probabilities if available
        **kwargs
    ) -> MCTSOutput:
        """Performs `self.num_iterations` MCTS iterations on an `StochasticMCTSTree`.
        Samples an action to take from the root node after search is completed.
        Handles initial and final stochastic states.
        
        Args:
        - `eval_state`: `StochasticMCTSTree` to evaluate, could be empty or partially complete
        - `env_state`: current environment state (potentially stochastic)
        - `root_metadata`: metadata for the root node of the tree
        - `params`: parameters to pass to the the leaf evaluation function
        - `env_step_fn`: env step fn: (env_state, action) -> (new_env_state, metadata)
        - `stochastic_action_probs`: Probabilities for stochastic actions (required if env can be stochastic)

        Returns:
        - (MCTSOutput): contains new tree state (rooted at a deterministic node), 
                      selected action (stochastic or deterministic), and policy weights.
        """
        debug_print(f"evaluate called on env_state with is_stochastic={env_state.is_stochastic}", level=1)

        root_metadata = kwargs.get('root_metadata', None)  # May be provided directly
        # If not, create from env_state
        if root_metadata is None:
            root_metadata = StepMetadata(
                action_mask=env_state.legal_action_mask,
                rewards=env_state.rewards,
                terminated=env_state.terminated,
                cur_player_id=env_state.current_player,
                step=getattr(env_state, "_step_count", 0) # use 0 as fallback
            )

        # Split key for various needs
        root_key, action_key = jax.random.split(key)
        debug_print(f"Root is stochastic: {env_state.is_stochastic}, Tree size: {eval_state.next_free_idx}", level=1)

        # 1. If eval_state is empty (first call), initialize it
        initialized = eval_state.next_free_idx > 0
        debug_print(f"Tree initialized: {initialized}", level=2)
        
        # Define proper helper methods for stochastic/deterministic roots
        def _select_from_stochastic_root_wrapper(operand):
            debug_print("Entering _select_from_stochastic_root_wrapper", level=2)
            result = self._select_from_stochastic_root(
                root_key, action_key, eval_state, env_state, params, env_step_fn, root_metadata
            )
            debug_print("Exiting _select_from_stochastic_root_wrapper", level=2)
            return result
            
        def _evaluate_deterministic_root_wrapper(operand):
            debug_print("Entering _evaluate_deterministic_root_wrapper", level=2)
            result = self._evaluate_deterministic_root(
                root_key, action_key, eval_state, env_state, params, env_step_fn, root_metadata
            )
            debug_print("Exiting _evaluate_deterministic_root_wrapper", level=2)
            return result
        
        # For stochastic root, make selections then run update_root on the deterministic result
        debug_print(f"About to call jax.lax.cond with is_stochastic={env_state.is_stochastic}", level=2)
        final_eval_state, final_action, final_policy_weights = jax.lax.cond(
            env_state.is_stochastic,
            _select_from_stochastic_root_wrapper,
            _evaluate_deterministic_root_wrapper,
            operand=None  # No operand needed but must be passed
        )
        debug_print(f"Selected action {final_action}, resulting tree size: {final_eval_state.next_free_idx}", level=1)
        
        return MCTSOutput(eval_state=final_eval_state, action=final_action, policy_weights=final_policy_weights)
    

    def get_value(self, state: StochasticMCTSTree) -> chex.Array:
        """Returns value estimate of the environment state stored in the root node of the tree.

        Args:
        - `state`: StochasticMCTSTree to evaluate

        Returns:
        - (chex.Array): value estimate of the environment state stored in the root node of the tree
        """
        return state.data_at(state.ROOT_INDEX).q
    

    def update_root(self, key: chex.PRNGKey, eval_state: StochasticMCTSTree, env_state: chex.ArrayTree, 
              params: chex.ArrayTree, root_metadata: StepMetadata = None, env_step_fn: EnvStepFn = None) -> StochasticMCTSTree:
        """Ensure the tree's root matches the current environment state.
        Handles both full resets and incremental updates.
        Sets up the state for further search iterations and action selection.
        
        Args:
        - `key`: random key
        - `eval_state`: current tree state (may be empty or outdated)
        - `env_state`: current environment state (definitive source of truth)
        - `params`: neural network parameters for evaluation
        - `root_metadata`: optional pre-computed metadata about the root state
        - `env_step_fn`: environment step function
        
        Returns:
        - Updated tree with root matching the current environment state
        """
        debug_print(f"update_root called, env_state is_stochastic={env_state.is_stochastic}, tree size={eval_state.next_free_idx}", level=1)
        
        # <<< Ensure env_state is treated as deterministic HERE >>>
        # Although the assertion was removed, the logic within update_root (initialize_tree/search_more)
        # fundamentally assumes a deterministic state. If PGX's stochastic_step returns a state
        # still marked as stochastic, we might need to override that assumption here if problems persist.
        # For now, assume the caller (_select_from_stochastic_root) handles ensuring a deterministic state is passed.
        
        # 1. Always initialize if empty
        initialized = eval_state.next_free_idx > 0 
        deterministic_root_state = env_state
        
        # Define our initialization and search functions for jax.lax.cond
        def initialize_tree(operand):
            # Unpack operand *without* env_step_fn
            key, eval_state, env_state, params, root_metadata = operand 
            # env_step_fn is available via closure
            debug_print("Initializing fresh tree", level=1)
            split_key, eval_key = jax.random.split(key)
            
            # Initialize a fresh tree with the root state
            template_node = MCTSNode(
                p=jnp.zeros(self.branching_factor),
                q=jnp.array(0.0, dtype=jnp.float32),
                n=jnp.array(0, dtype=jnp.int32),  # Make sure it's a JAX array
                embedding=deterministic_root_state, 
                terminated=jnp.array(False)
            )
            fresh_tree_base = init_tree(self.max_nodes, self.branching_factor, template_node)
            fresh_tree = StochasticMCTSTree(
                next_free_idx=fresh_tree_base.next_free_idx,
                parents=fresh_tree_base.parents,
                edge_map=fresh_tree_base.edge_map,
                data=fresh_tree_base.data,
                node_is_stochastic=jnp.zeros(self.max_nodes, dtype=jnp.bool_)
            )
            
            # If the root is deterministic, evaluate it to get policy and value
            policy_logits, value = self.eval_fn(deterministic_root_state, params, eval_key)
            
            # Apply action masking and softmax
            policy_logits = jnp.where(root_metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min)
            policy = jax.nn.softmax(policy_logits)
            
            # Initialize the root node with the game state, policy and value
            root_data = self.new_node(policy=policy,
                            value=value,
                            embedding=deterministic_root_state,
                            terminated=root_metadata.terminated)
            
            # Debug print to check the value and type of n in root_data
            #dprint("initialize_tree: root_data.n = {n}, type = {t}", n=root_data.n, t=type(root_data.n))
            
            # Update root's data with the evaluated policy and value
            updated_tree = fresh_tree.set_root(root_data)
            
            # Debug print to verify updated_tree's root node n is set correctly
            #dprint("initialize_tree: After set_root, updated_tree.data_at(0).n = {n}", n=updated_tree.data_at(0).n)
            
            # --- Use jax.lax.fori_loop for MCTS iterations --- 
            def iterate_body(i, loop_tree):
                debug_print(f"initialize_tree: Iteration {i}", level=2)
                # === DEBUG ADDED ===
                debug_print(f"  initialize_tree loop {i}: BEFORE iterate, tree size = {loop_tree.next_free_idx}", level=1)
                # === END DEBUG ===
                iter_key = jax.random.fold_in(split_key, i)
                # Call iterate
                result_tree = self.iterate(iter_key, loop_tree, params, env_step_fn)
                # Debug print root visit count after iteration
                #dprint("initialize_tree loop {i}: root_n={n}", i=i, n=result_tree.data_at(result_tree.ROOT_INDEX).n)
                # === DEBUG ADDED ===
                debug_print(f"  initialize_tree loop {i}: AFTER iterate, tree size = {result_tree.next_free_idx}", level=1)
                # === END DEBUG ===
                return result_tree
            
            debug_print(f"About to run {self.num_iterations} iterations in initialize_tree", level=2)
            updated_tree = jax.lax.fori_loop(
                0, self.num_iterations, iterate_body, updated_tree
            )
            
            debug_print(f"Tree initialized with {updated_tree.next_free_idx} nodes", level=1)
            return updated_tree
        
        def search_more(operand):
            # Unpack operand
            key, eval_state, env_state, params, root_metadata = operand
            # env_step_fn available via closure
            
            debug_print(f"Tree already initialized, updating root and running additional iterations, current size: {eval_state.next_free_idx}", level=2)
            update_key, iter_key = jax.random.split(key)

            # <<< FIX: Explicitly update root node embedding and re-evaluate >>>
            # Evaluate the *current* env_state to get fresh policy/value for the root
            policy_logits, value = self.eval_fn(env_state, params, update_key)
            policy_logits = jnp.where(root_metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min)
            policy = jax.nn.softmax(policy_logits)

            # Get the current root node data
            current_root_node = eval_state.data_at(eval_state.ROOT_INDEX)

            # Update the root node's embedding, policy, and value (but keep visit count/existing Q if needed)
            # Using the dedicated static method is cleaner
            updated_root_data = self.update_root_node(
                root_node=current_root_node, 
                root_policy=policy, 
                root_value=value, 
                root_embedding=env_state # Use the current env_state!
            )
            
            # Put the updated root data back into the tree
            tree_with_updated_root = eval_state.update_node(eval_state.ROOT_INDEX, updated_root_data)
            # <<< END FIX >>>

            # --- Use jax.lax.fori_loop for MCTS iterations --- 
            def iterate_body(i, loop_tree):
                debug_print(f"search_more: Iteration {i}", level=2)
                # Fold key correctly for iterations
                loop_iter_key = jax.random.fold_in(iter_key, i)
                # Call iterate
                result_tree = self.iterate(loop_iter_key, loop_tree, params, env_step_fn)
                return result_tree
                
            debug_print(f"About to run {self.num_iterations} iterations in search_more", level=2)
            # Start iterations from the tree with the updated root
            updated_tree = jax.lax.fori_loop(
                0, self.num_iterations, iterate_body, tree_with_updated_root
            )
            
            debug_print(f"Additional iterations complete, tree size: {updated_tree.next_free_idx}", level=2)
            return updated_tree
        
        # 3. Run initialize_tree or search_more based on whether the tree is already initialized
        debug_print(f"Calling jax.lax.cond with initialized={initialized}", level=2)
        updated_tree = jax.lax.cond(
            initialized,
            search_more,
            initialize_tree,
            operand=(key, eval_state, deterministic_root_state, params, root_metadata)
        )
        
        debug_print("update_root completed", level=2)
        return updated_tree

    def _expand_deterministic_node(self, key: chex.PRNGKey, tree: StochasticMCTSTree, 
                    parent_idx: int, action: int,
                    params: chex.ArrayTree, env_step_fn: EnvStepFn) -> Tuple[StochasticMCTSTree, int, chex.Array]:
        """Expands a deterministic node at (parent_idx, action).
        Returns the updated tree, the index of the expanded node, and value."""
        
        debug_print(f"Expanding deterministic node: parent={parent_idx}, action={action}", level=2)
        #dprint("_expand_deterministic_node ENTRY: parent={p}, action={a}", p=parent_idx, a=action)
        
        # CRITICAL! Must have env_step_fn to expand nodes
        if env_step_fn is None:
            debug_print("WARNING: Cannot expand deterministic node without environment step function", level=1)
            # Use placeholder return value in case this is hit during compilation
            return tree, 0, jnp.array(0.0)
        
        parent_node = tree.data_at(parent_idx)
        parent_embedding = parent_node.embedding
        
        # Environment step
        step_key, eval_key = jax.random.split(key)
        debug_print(f"Calling env_step_fn for deterministic expansion with action={action}", level=2)
        new_embedding, new_metadata = env_step_fn(parent_embedding, action, step_key)
        debug_print(f"env_step_fn returned state with is_stochastic={new_embedding.is_stochastic}", level=2)
        
        # Evaluate the new state (or stochastic node)
        is_new_node_stochastic = new_embedding.is_stochastic
        
        debug_print("Evaluating leaf node", level=2)
        policy_logits, value = self.eval_fn(new_embedding, params, eval_key)
        
        # Apply action masking and softmax
        policy_logits = jnp.where(new_metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min)
        policy = jax.nn.softmax(policy_logits)
        
        # === FIX: If the new node represents a stochastic state, don't use its NaN policy ===
        # Store zeros instead. The selection should rely on Q-values once visits happen.
        policy = jnp.where(is_new_node_stochastic, 
                           jnp.zeros_like(policy), 
                           policy)
        # === END FIX ===

        # === ADDED DEBUG: Log mask and policy before node creation/update ===
        debug_print(f"  _expand_deterministic_node: For parent={parent_idx}, action={action}", level=1)
        debug_print(f"    New node mask (first 10): {new_metadata.action_mask[:10]}", level=1)
        debug_print(f"    New node policy (first 10): {policy[:10]}", level=1)
        debug_print(f"    Is Action 0 legal in new mask? {new_metadata.action_mask[0]}", level=1)
        # === END DEBUG ===

        # Check if we've already seen this node
        existing_node_idx = tree.edge_map[parent_idx, action]
        node_exists = tree.is_edge(parent_idx, action)
        debug_print(f"Node exists: {node_exists}, existing_node_idx: {existing_node_idx}", level=2)
        
        # Helper to create a default node if needed
        def create_default_node():
             return MCTSNode( 
                n=jnp.array(0, dtype=jnp.int32),
                p=jnp.zeros(self.branching_factor, dtype=jnp.float32),
                q=jnp.array(0.0, dtype=jnp.float32),
                terminated=jnp.array(False),
                embedding=tree.data_at(0).embedding # Use root embedding as template shape
            )

        existing_node_placeholder = jax.lax.cond(
            node_exists,
            lambda: tree.data_at(existing_node_idx),
            create_default_node # Pass function reference
        )
        
        # Case where node already exists
        def update_existing_node(operand):
            debug_print("Updating existing node", level=2)
            tree, next_idx, existing_idx, is_stochastic, node_data = operand
            updated_node = node_data.replace(
                p=policy, 
                q=value,
                embedding=new_embedding,
                terminated=new_metadata.terminated
            )
            updated_tree = tree.update_node(index=existing_idx, data=updated_node)
            return updated_tree, existing_idx
            
        # Case where node doesn't exist yet
        def add_new_node(operand):
            debug_print("Adding new node to tree", level=2)
            tree, next_idx, _existing_idx, is_stochastic, _node_data = operand
            new_node_idx = next_idx
            new_node = MCTSNode(
                n=jnp.array(0, dtype=jnp.int32), 
                p=policy, 
                q=value, 
                terminated=new_metadata.terminated,
                embedding=new_embedding
            )
            # Correct arguments: parent_index, edge_index (action), data
            updated_tree = tree.add_node(parent_idx, action, new_node)
            # updated_tree = updated_tree.data_replace(
            #     'node_is_stochastic', new_node_idx, is_stochastic
            # )
            # <<< FIX: Use standard JAX array update >>>
            updated_tree = updated_tree.replace(node_is_stochastic=updated_tree.node_is_stochastic.at[new_node_idx].set(is_stochastic))
            debug_print(f"Added new node at index {new_node_idx}, is_stochastic={is_stochastic}", level=2)
            return updated_tree, new_node_idx
            
        # operand uses the placeholder created above
        operand = (tree, tree.next_free_idx, existing_node_idx, is_new_node_stochastic, existing_node_placeholder) 
        
        # Execute the appropriate branch
        updated_tree, expanded_idx = jax.lax.cond(
            node_exists, 
            update_existing_node,
            add_new_node,
            operand=operand
        )
        
        debug_print(f"Returning from _expand_deterministic_node with value={value}", level=2)
        #dprint("_expand_deterministic_node EXIT: expanded_idx={idx}, value={v}", idx=expanded_idx, v=value)
        return updated_tree, expanded_idx, value
    
    
    def _expand_stochastic_node(self, key: chex.PRNGKey, tree: StochasticMCTSTree, 
                    stochastic_node_idx: int, params: chex.ArrayTree, 
                    env_step_fn: EnvStepFn) -> Tuple[StochasticMCTSTree, int, chex.Array]:
        """Expands a stochastic node by considering all possible outcomes.
        Ensures all stochastic actions have corresponding child nodes.
        Returns the updated tree, and the final weighted state value.
        
        In subsequent iterations, traversal can continue through this node.
        """
        
        debug_print(f"Expanding stochastic node {stochastic_node_idx}", level=2)
        #dprint("_expand_stochastic_node ENTRY: node_idx={idx}", idx=stochastic_node_idx)
        
        # CRITICAL! Must have env_step_fn to expand nodes
        if env_step_fn is None:
            debug_print("WARNING: Cannot expand stochastic node without environment step function", level=1)
            # Use placeholder return value in case this is hit during compilation
            return tree, stochastic_node_idx, jnp.array(0.0)
        
        # Get the stochastic node's embedding
        stochastic_node = tree.data_at(stochastic_node_idx)
        stochastic_embedding = stochastic_node.embedding
        
        # Get the number of stochastic outcomes and their probabilities
        num_stochastic_outcomes = len(self.stochastic_action_probs)
        
        debug_print(f"Preparing to expand {num_stochastic_outcomes} stochastic outcomes", level=2)
        
        # Check if the stochastic node is already fully expanded
        def is_fully_expanded(node_idx):
            # Check if each stochastic action has a corresponding child node
            def check_all_edges(i, all_exist):
                edge_exists = tree.is_edge(node_idx, i)
                return jnp.logical_and(all_exist, edge_exists)
            
            # Start with True and check all edges
            all_edges_exist = jax.lax.fori_loop(
                0, num_stochastic_outcomes, check_all_edges, jnp.array(True)
            )
            
            return all_edges_exist
        
        # If already fully expanded, just recalculate the weighted value and return
        def handle_already_expanded(operand):
            input_tree, input_node_idx = operand
            debug_print(f"Stochastic node {input_node_idx} already fully expanded, recalculating value", level=2)
            
            # Recalculate the weighted value from all child nodes
            def sum_child_values(i, acc_value):
                child_exists = input_tree.is_edge(input_node_idx, i)
                child_idx = input_tree.edge_map[input_node_idx, i]
                child_q = jnp.where(child_exists, input_tree.data_at(child_idx).q, 0.0)
                prob = self.stochastic_action_probs[i]
                return acc_value + prob * child_q
            
            weighted_value = jax.lax.fori_loop(
                0, num_stochastic_outcomes, sum_child_values, 0.0
            )
            
            # Update the stochastic node's Q-value with the recalculated weighted sum
            node_data = input_tree.data_at(input_node_idx)
            updated_node = node_data.replace(
                q=weighted_value,
                n=node_data.n + 1  # Still increment visit count
            )
            
            updated_tree = input_tree.update_node(input_node_idx, updated_node)
            
            debug_print(f"Recalculated weighted value: {weighted_value}", level=2)
            
            return updated_tree, input_node_idx, weighted_value
        
        # Process each stochastic action - expand any unexpanded outcomes
        def expand_all_outcomes(operand):
            input_tree, input_node_idx = operand
            debug_print(f"Expanding all outcomes for stochastic node {input_node_idx}", level=2)
            
            # Process each stochastic action
            def expand_one_stochastic_outcome(i, scan_val):
                action_idx = i
                debug_print(f"Processing stochastic action {action_idx}", level=2)
                
                scan_tree, accumulated_value = scan_val
                prob = self.stochastic_action_probs[action_idx]
                
                # Take the action in the environment
                step_key, eval_key = jax.random.split(jax.random.fold_in(key, action_idx))
                
                debug_print(f"Calling env_step_fn for stochastic action {action_idx}", level=2)
                # Original step function call with key passing
                child_embedding, child_metadata = env_step_fn(stochastic_embedding, action_idx, step_key)
                debug_print(f"env_step_fn for stochastic action returned state with is_stochastic={child_embedding.is_stochastic}", level=2)
                
                # Evaluate the child state 
                debug_print(f"Evaluating child state for stochastic action {action_idx}", level=2)
                policy_logits, value = self.eval_fn(child_embedding, params, eval_key)
                
                # Apply action masking and softmax
                policy_logits = jnp.where(child_metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min)
                policy = jax.nn.softmax(policy_logits)
                
                # Get the child node index
                # child_exists, child_node_idx = scan_tree.node_exists(stochastic_node_idx, action_idx)
                child_node_idx = scan_tree.edge_map[input_node_idx, action_idx]
                child_exists = scan_tree.is_edge(input_node_idx, action_idx)
                
                # Determine the node index (existing or new)
                new_child_node_idx = jax.lax.cond(
                    child_exists,
                    lambda: child_node_idx,
                    lambda: scan_tree.next_free_idx
                )
                
                # Create the child node using ONLY defined fields
                child_node = MCTSNode(
                    n=jnp.array(0, dtype=jnp.int32), # Start visits at 0
                    p=policy,
                    q=value, # Initial q is evaluated value
                    terminated=child_metadata.terminated,
                    embedding=child_embedding
                )
                
                # Add the child node to the tree
                # Correct arguments: parent_index, edge_index (action), data
                add_new_branch = lambda: scan_tree.add_node(input_node_idx, action_idx, child_node)
                updated_tree = jax.lax.cond(
                    child_exists,
                    lambda: scan_tree.update_node(index=child_node_idx, data=child_node),
                    add_new_branch
                )
                
                # Update stochastic flags for this node
                is_child_stochastic = child_embedding.is_stochastic
                # <<< FIX: Use standard JAX array update >>>
                updated_tree = updated_tree.replace(node_is_stochastic=updated_tree.node_is_stochastic.at[new_child_node_idx].set(is_child_stochastic))

                debug_print(f"Adding child node for stochastic action {action_idx} at index {new_child_node_idx}", level=2)
                
                # Accumulate weighted value
                new_accumulated_value = accumulated_value + prob * value
                
                debug_print(f"Accumulated value: {new_accumulated_value} after adding weighted value {prob * value}", level=2)
                
                # return updated_tree, new_accumulated_value
                # <<< FIX: Return structure must match scan carry: ((tree, acc), y) >>>
                return (updated_tree, new_accumulated_value), None

            # Initialize accumulator with zero value and the original tree
            initial_accumulated_value = jnp.array(0.0, dtype=jnp.float32)
            
            debug_print(f"Starting fori_loop for {num_stochastic_outcomes} stochastic outcomes", level=2)
            # Use scan to accumulate values
            updated_tree, total_weighted_value = jax.lax.scan(
                lambda val, i: expand_one_stochastic_outcome(i, val),
                (input_tree, initial_accumulated_value),
                jnp.arange(num_stochastic_outcomes)
            )[0]
            
            debug_print(f"Completed stochastic expansion with total_weighted_value={total_weighted_value}", level=2)
            
            # Update the stochastic node's Q-value with the weighted sum
            stochastic_node_data = updated_tree.data_at(input_node_idx)
            updated_node = stochastic_node_data.replace(
                q=total_weighted_value,
                n=stochastic_node_data.n + 1  # Increment the visit count
            )
            
            updated_tree = updated_tree.update_node(input_node_idx, updated_node)
            
            debug_print(f"Updated stochastic node with weighted value={total_weighted_value}", level=2)
            return updated_tree, input_node_idx, total_weighted_value
        
        # Check if the node is already fully expanded
        fully_expanded = is_fully_expanded(stochastic_node_idx)
        debug_print(f"Stochastic node {stochastic_node_idx} fully expanded? {fully_expanded}", level=2)
        
        # Either recalculate value of fully expanded node or expand all outcomes
        result_tree, result_idx, result_value = jax.lax.cond(
            fully_expanded,
            handle_already_expanded,
            expand_all_outcomes,
            (tree, stochastic_node_idx)
        )
        
        #dprint("_expand_stochastic_node EXIT: node_idx={idx}, value={v}", idx=stochastic_node_idx, v=total_weighted_value)
        # Return the updated tree, the stochastic node's index, and the weighted value
        return result_tree, result_idx, result_value

    def iterate(self, key: chex.PRNGKey, tree: StochasticMCTSTree, params: chex.ArrayTree, env_step_fn: EnvStepFn) -> StochasticMCTSTree:
        """ Performs one iteration of MCTS.
        1. Traverse to leaf node (or stochastic node).
        2. Evaluate & Expand Node (handles deterministic vs stochastic)
        3. Backpropagate

        Args:
        - `tree`: StochasticMCTSTree to evaluate
        - `params`: parameters to pass to the the leaf evaluation function
        - `env_step_fn`: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
        - (StochasticMCTSTree): updated StochasticMCTSTree
        """
        debug_print(f"Starting MCTS iteration. Tree node count: {tree.next_free_idx}")
        expand_key, backprop_key = jax.random.split(key)
        
        # 1. Traverse from root -> leaf or stochastic node boundary
        traversal_state = self.traverse(tree)
        parent_idx, action = traversal_state.parent, traversal_state.action
        #dprint("Iterate: traverse returned parent={p}, action={a}", p=parent_idx, a=action)
        # === DEBUG ADDED ===
        debug_print(f"Iterate: Traversal result parent={parent_idx}, action={action}", level=1)
        # === END DEBUG ===

        # 2. Determine if the action from parent leads to a stochastic outcome/node
        edge_exists = tree.is_edge(parent_idx, action)
        child_idx_if_exists = tree.edge_map[parent_idx, action]
        debug_print(f"Edge exists: {edge_exists}, child_idx_if_exists: {child_idx_if_exists}")
        
        # If edge exists, just check if the child node is marked as stochastic
        # If edge doesn't exist, assume deterministic since we can't check without env_step_fn
        is_stochastic_expansion = jnp.where(
            edge_exists, 
            tree.node_is_stochastic[child_idx_if_exists],
            jnp.array(False)
        )
        debug_print(f"Is stochastic expansion: {is_stochastic_expansion}")

        # 3. Expand Node (conditionally)
        # Each expansion function returns (updated_tree, node_idx_to_backprop_from, value_to_backprop)
        # Define expand stochastic wrapper
        def do_expand_stochastic(operand):
            nonlocal child_idx_if_exists
            debug_print(f"Calling _expand_stochastic_node with node_idx={child_idx_if_exists}")
            result = self._expand_stochastic_node(expand_key, tree, child_idx_if_exists, params, env_step_fn)
            debug_print(f"_expand_stochastic_node returned with backprop_start_idx={result[1]}, value={result[2]}")
            return result
            
        # Define expand deterministic wrapper
        def do_expand_deterministic(operand):
            debug_print(f"Calling _expand_deterministic_node with parent_idx={parent_idx}, action={action}")
            result = self._expand_deterministic_node(expand_key, tree, parent_idx, action, params, env_step_fn)
            debug_print(f"_expand_deterministic_node returned with backprop_start_idx={result[1]}, value={result[2]}")
            return result
        
        # RESTORED TRY/EXCEPT
        try:  # <-- Restore try
            # We need to handle the case where the edge exists and leads to a stochastic node
            debug_print(f"About to call jax.lax.cond with is_stochastic_expansion={is_stochastic_expansion}")
            tree, backprop_start_idx, value_to_backprop = jax.lax.cond(
                is_stochastic_expansion,
                do_expand_stochastic,
                do_expand_deterministic,
                operand=None
            )
            # === DEBUG ADDED ===
            debug_print(f"Iterate: Expansion decision: is_stochastic={is_stochastic_expansion}, edge_exists={edge_exists}, backprop_start_idx={backprop_start_idx}", level=1)
            # === END DEBUG ===
            debug_print(f"Expansion complete, backprop_start_idx={backprop_start_idx}, value_to_backprop={value_to_backprop}")
            # === DEBUG ADDED ===
            # Debug print expansion result
            #dprint("Iterate: expansion returned tree_next_free={nfi}, start_idx={idx}, value={v}", 
            #        nfi=tree.next_free_idx, idx=backprop_start_idx, v=value_to_backprop)
        except Exception as e: # <-- Restore except
            debug_print(f"Exception in expansion: {e}")
            # Return tree unchanged if expansion fails
            return tree

        # 4. Backpropagate
        # RESTORED TRY/EXCEPT
        try: # <-- Restore try
            # Note: The 'value' from deterministic expansion is the NN output. 
            # The 'value' from stochastic expansion will be the calculated weighted average.
            # The 'parent'/'backprop_start_idx' is where backprop begins.
            debug_print(f"About to call backpropagate() with start_node_idx={backprop_start_idx}, value={value_to_backprop}")
            # === DEBUG ADDED ===
            # Print the start index being passed
            #dprint("Iterate: backprop_start_idx = {idx}", idx=backprop_start_idx)
            result_tree = self.backpropagate(backprop_key, tree, backprop_start_idx, value_to_backprop)
            #debug_print("Backpropagation complete")
            return result_tree
        except Exception as e: # <-- Restore except
            debug_print(f"Exception in backpropagate: {e}")
            # Return tree unchanged if backprop fails
            return tree


    def traverse(self, tree: StochasticMCTSTree) -> TraversalState:
        """ Traverse from the root node until an unvisited leaf node or an unexpanded stochastic node is reached.
        Now supports traversing through already-expanded stochastic nodes.
        
        Args:
        - `tree`: StochasticMCTSTree to evaluate
        
        Returns:
        - (TraversalState): state of the traversal
            - `parent`: index of the node where traversal stopped.
            - `action`: action selected from the `parent` node.
                      The edge (`parent`, `action`) might be unvisited, or it might lead to an unexpanded stochastic node.
        """
        debug_print(f"Starting traversal from root (node {tree.ROOT_INDEX})")
        # Choose the action to take from the root initially
        try:
            root_action = self.action_selector(tree, tree.ROOT_INDEX, self.discount)
            initial_state = TraversalState(parent=tree.ROOT_INDEX, action=root_action)
            debug_print(f"Initial action selected: {root_action}")
        except Exception as e:
            debug_print(f"Exception in action selection: {e}")
            # Return a default state if action selection fails
            return TraversalState(parent=tree.ROOT_INDEX, action=0)

        def cond_fn(state: TraversalState) -> bool:
            """Continue traversal if:
            1. The selected edge exists AND 
            2. The child node it leads to is not terminal AND
            3. Either:
               a. The child is deterministic, OR
               b. The child is a stochastic node that has been fully expanded
            """
            debug_print(f"Checking cond_fn with parent={state.parent}, action={state.action}")
            edge_exists = tree.is_edge(state.parent, state.action)
            
            # Get potential child index only if edge exists, otherwise use a dummy index (-1)
            child_idx = jnp.where(edge_exists, tree.edge_map[state.parent, state.action], -1)
            
            # Check child properties only if the index is valid
            is_child_stochastic = jnp.where(child_idx >= 0, tree.node_is_stochastic[child_idx], False)
            is_child_terminal = jnp.where(child_idx >= 0, tree.data_at(child_idx).terminated, False)
            
            # NEW: Check if stochastic node is fully expanded
            # A stochastic node is fully expanded if all its stochastic action outcomes 
            # have corresponding child nodes in the tree
            def is_stochastic_node_fully_expanded(idx):
                # Check if each stochastic action has a corresponding child node
                num_stochastic_outcomes = len(self.stochastic_action_probs)
                
                # Use for loop to check if all stochastic actions have edges
                def check_all_edges(i, all_exist):
                    edge_exists = tree.is_edge(idx, i)
                    return jnp.logical_and(all_exist, edge_exists)
                
                # Start with True and check all edges
                all_edges_exist = jax.lax.fori_loop(
                    0, num_stochastic_outcomes, check_all_edges, jnp.array(True)
                )
                
                return all_edges_exist
            
            # Check if stochastic child is fully expanded (only matters if the child exists and is stochastic)
            stochastic_fully_expanded = jnp.where(
                jnp.logical_and(child_idx >= 0, is_child_stochastic),
                is_stochastic_node_fully_expanded(child_idx),
                jnp.array(False)  # Doesn't matter if child isn't stochastic
            )
            
            # Continue if: 
            # 1. Edge exists AND
            # 2. Child is not terminal AND
            # 3. Either:
            #    a. Child is not stochastic OR
            #    b. Child is stochastic but fully expanded
            continue_deterministic = jnp.logical_and(edge_exists, ~is_child_terminal)
            continue_through_stochastic = jnp.logical_and(
                is_child_stochastic,
                stochastic_fully_expanded
            )
            
            result = jnp.logical_and(
                continue_deterministic,
                jnp.logical_or(~is_child_stochastic, continue_through_stochastic)
            )
            
            debug_print(f"cond_fn result: {result} (edge_exists={edge_exists}, is_child_stochastic={is_child_stochastic}, " +
                       f"is_child_terminal={is_child_terminal}, stochastic_fully_expanded={stochastic_fully_expanded})")
            return result
        
        def body_fn(state: TraversalState) -> TraversalState:
            """Move to the child node and select the next action."""
            debug_print(f"In body_fn with parent={state.parent}, action={state.action}")
            # We know from cond_fn that the edge exists and the child is either:
            # 1. deterministic & not terminal OR
            # 2. stochastic & fully expanded & not terminal
            node_idx = tree.edge_map[state.parent, state.action]
            
            # === ADDED DEBUG: Log policy and mask for the current node ===
            current_node_data = tree.data_at(node_idx)
            current_policy = current_node_data.p
            current_embedding = current_node_data.embedding
            current_mask = current_embedding.legal_action_mask
            debug_print(f"  Node {node_idx} data for selection:", level=1)
            debug_print(f"    Policy (first 10): {current_policy[:10]}", level=1) # Show subset
            debug_print(f"    Legal Mask (first 10): {current_mask[:10]}", level=1) # Show subset
            debug_print(f"    Is No-Op (Action 0) legal? {current_mask[0]}", level=1)
            # === END DEBUG ===
            
            # Check if the node is stochastic
            is_node_stochastic = tree.node_is_stochastic[node_idx]
            
            # Define how to select the next action
            def select_from_deterministic():
                # Use the action selector for deterministic nodes
                return self.action_selector(tree, node_idx, self.discount)
            
            def select_from_stochastic():
                # For stochastic nodes, select among stochastic actions
                # We weight selection by the stochastic action probabilities
                # and also take into account visit counts if available
                
                # Get visit counts for each stochastic action's child
                def get_child_visit(action_idx):
                    child_exists = tree.is_edge(node_idx, action_idx)
                    child_idx = tree.edge_map[node_idx, action_idx]
                    return jnp.where(child_exists, tree.data_at(child_idx).n, 0)
                
                # Get visit counts for each stochastic action
                visit_counts = jnp.array([get_child_visit(i) for i in range(len(self.stochastic_action_probs))])
                
                # Combine probabilities and visit counts for selection
                # Formula: prob * (1 + visit_count)^(-0.5) (similar to PUCT)
                selection_weights = self.stochastic_action_probs * jnp.power(1.0 + visit_counts, -0.5)
                
                # Normalize weights
                normalized_weights = selection_weights / jnp.sum(selection_weights)
                
                # Select action based on weights
                # Use a deterministic approach to avoid random key management
                # Select the action with the highest weight
                return jnp.argmax(normalized_weights)
            
            # Use jax.lax.cond to select the appropriate action based on node type
            try:
                action = jax.lax.cond(
                    is_node_stochastic,
                    lambda _: select_from_stochastic(),
                    lambda _: select_from_deterministic(),
                    operand=None
                )
                debug_print(f"Selected action {action} from node {node_idx} (stochastic: {is_node_stochastic})")
                return TraversalState(parent=node_idx, action=action)
            except Exception as e:
                debug_print(f"Exception in action selection in body_fn: {e}")
                # Return the current state unchanged if selection fails
                return state
        
        # Traverse from root until cond_fn is false
        try:
            debug_print("About to enter while_loop in traverse")
            final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
            debug_print(f"while_loop completed, final state: parent={final_state.parent}, action={final_state.action}")
            return final_state
        except Exception as e:
            debug_print(f"Exception in while_loop: {e}")
            # Return the initial state if while_loop fails
            return initial_state


    def backpropagate(self, key: chex.PRNGKey, tree: StochasticMCTSTree, start_node_idx: int, value: float) -> StochasticMCTSTree: #pylint: disable=unused-argument
        """Backpropagate the value estimate from the start node up to the root node and update visit counts.
        Handles stochastic nodes by recalculating their value based on children.
        
        Args:
        - `key`: rng (unused, kept for potential future use)
        - `tree`: StochasticMCTSTree to evaluate
        - `start_node_idx`: index of the node from which backpropagation begins (leaf or expanded stochastic node).
        - `value`: value estimate to propagate upwards.

        Returns:
        - (StochasticMCTSTree): updated search tree
        """
        debug_print(f"Starting backpropagation from node {start_node_idx} with value {value}")

        def body_fn(state: BackpropState) -> BackpropState:
            """Updates one node in the backpropagation path."""
            # Note: loop_idx is not directly used, state carries the current node index
            node_idx, current_value, current_tree = state.node_idx, state.value, state.tree
            debug_print(f"Backpropagating through node {node_idx} with value {current_value}")
            
            # Stop if we reached the root's parent (-1)
            # Should not happen if start_node_idx is valid, but good safeguard.
            # The loop condition handles the primary termination.
            # Correct attribute name is 'parents'
            parent_idx = current_tree.parents[node_idx]
            # is_root = (node_idx == current_tree.ROOT_INDEX)

            # Get node data
            node_data = current_tree.data_at(node_idx)
            is_stochastic = current_tree.node_is_stochastic[node_idx]
            debug_print(f"Node {node_idx}: is_stochastic={is_stochastic}, n={node_data.n}, q={node_data.q}, parent={parent_idx}")

            # --- Calculate updated Q-value conditionally ---
            def update_deterministic_q():
                # Standard MCTS update: Q = (Q*N + V) / (N+1)
                # Ensure division by zero is handled if n starts at 0
                # We assume n represents visits *before* this backprop step.
                n_plus_1 = node_data.n + 1.0 # Use float for division
                new_q = (node_data.q * node_data.n + current_value) / n_plus_1
                debug_print(f"Deterministic update: old_q={node_data.q}, n={node_data.n}, value={current_value}, new_q={new_q}")
                return new_q

            def update_stochastic_q():
                debug_print(f"Updating stochastic node {node_idx}")
                # Value is the weighted average of children's current Q-values
                # Iterate through all possible actions/edges
                num_stochastic_outcomes = len(self.stochastic_action_probs)
                
                def sum_child_values(action_idx, running_value_sum):
                    child_exists = current_tree.is_edge(node_idx, action_idx)
                    child_node_idx = current_tree.edge_map[node_idx, action_idx]
                    # Get child Q-value if edge exists, otherwise 0
                    child_q = jnp.where(child_exists, current_tree.data_at(child_node_idx).q, 0.0)
                    # Get probability for this action
                    prob = self.stochastic_action_probs[action_idx]
                    # Add weighted value
                    new_sum = running_value_sum + prob * child_q
                    debug_print(f"  Child {action_idx}: exists={child_exists}, child_idx={child_node_idx if child_exists else 'N/A'}, q={child_q if child_exists else 0}, prob={prob}, running_sum={new_sum}")
                    return new_sum
                
                # Calculate sum using fori_loop over all stochastic outcomes
                weighted_q_sum = jax.lax.fori_loop(0, num_stochastic_outcomes, sum_child_values, 0.0)
                # The stochastic node's value IS this weighted sum.
                debug_print(f"Stochastic update final weighted_q_sum={weighted_q_sum}")
                return weighted_q_sum
            
            updated_q = jax.lax.cond(
                is_stochastic,
                update_stochastic_q,
                update_deterministic_q
            )

            # --- Update node data --- 
            updated_node_data = node_data.replace(
                n=node_data.n + 1, # Always increment visit count
                q=updated_q
            )
            # Debug print n *after* update but before returning
            #dprint("backprop body_fn node {idx}: updated_n={n}", idx=node_idx, n=updated_node_data.n)
            
            # Update tree
            new_tree = current_tree.update_node(index=node_idx, data=updated_node_data)
            
            # Prepare state for next iteration
            # Correct attribute name is 'parents'
            parent_idx = new_tree.parents[node_idx] # Use parent from the *updated* tree
            
            # The value propagated to the parent depends on the node type
            # For stochastic nodes, we always use the recalculated weighted sum value
            next_value = jax.lax.cond(
                is_stochastic,
                lambda: updated_q * self.discount, # Propagate the recalculated value
                lambda: current_value * self.discount # Propagate the original value discounted
            )
            
            debug_print(f"Node {node_idx} updated: n={updated_node_data.n}, q={updated_node_data.q}, next node will be {parent_idx}")
            # Return the *new_tree* in the state
            return BackpropState(node_idx=parent_idx, value=next_value, tree=new_tree) 

        def cond_fn(state: BackpropState) -> bool:
            """Continue backpropagation as long as we haven't processed the root node."""
            # We process the root node, then stop. The loop runs N times where N is path length.
            # state.node_idx becomes -1 after processing the root.
            result = state.node_idx != -1
            debug_print(f"backprop cond_fn with node_idx={state.node_idx}, result={result}")
            return result

        # Initial state for the loop
        # Start backprop from the node *parent* of the newly added/evaluated leaf 
        # OR from the stochastic node itself if that's what was expanded.
        # Correct attribute name is 'parents'
        initial_parent_idx = tree.parents[start_node_idx]
        initial_state = BackpropState(node_idx=start_node_idx, value=value, tree=tree)
        debug_print(f"Backprop initial state: node_idx={start_node_idx}, value={value}, parent={initial_parent_idx}")
        
        # Run the loop
        try:
            # Note: Using while_loop as path length isn't fixed
            debug_print("Entering backprop while_loop")
            final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)
            debug_print(f"Backprop while_loop completed, final tree size: {final_state.tree.next_free_idx}")
            return final_state.tree
        except Exception as e:
            debug_print(f"Exception in backprop while_loop: {e}")
            # Return the original tree if while_loop fails
            return tree


    def sample_root_action(self, key: chex.PRNGKey, tree: StochasticMCTSTree) -> Tuple[int, jnp.ndarray]:
        """Sample an action from the root node.
        
        Args:
        - `key`: The random key for sampling
        - `tree`: The StochasticMCTSTree
        
        Returns:
        - `action`: The selected action
        - `policy_weights`: The visit count-based policy weights
        """
        # Get the root node data
        root_node = tree.data_at(tree.ROOT_INDEX)
        
        # Get edge counts (N values) for all child nodes from the root
        # Note: We need to extract edge stats from the actual child nodes for each action
        n_values = jnp.zeros((self.branching_factor,), dtype=jnp.float32)
        
        # Loop through all possible actions to get N values for each child
        def update_n_values(i, n_vals):
            has_edge = tree.is_edge(tree.ROOT_INDEX, i)
            child_idx = jnp.where(has_edge, tree.edge_map[tree.ROOT_INDEX, i], 0)
            n_val = jnp.where(has_edge, tree.data_at(child_idx).n, 0)
            return n_vals.at[i].set(n_val)
        
        n_values = jax.lax.fori_loop(0, self.branching_factor, update_n_values, n_values)
        
        # Get the legal action mask directly from the root node's embedding
        legal_action_mask = root_node.embedding.legal_action_mask
        
        # Apply legal action mask to visit counts
        masked_n_values = jnp.where(legal_action_mask, n_values, 0.0)
        
        # Convert visit counts to a policy (add small epsilon to avoid log(0))
        total_n = jnp.sum(masked_n_values) + 1e-8
        policy_weights = masked_n_values / total_n
        
        # If no legal actions have been visited, use uniform distribution over legal actions
        has_legal_visits = jnp.sum(masked_n_values) > 0
        uniform_policy = legal_action_mask / (jnp.sum(legal_action_mask) + 1e-8)
        policy_weights = jnp.where(has_legal_visits, policy_weights, uniform_policy)
        
        # Select action based on policy and temperature
        if self.temperature == 0:
            # Greedy selection (highest visit count) - pick the largest legal value
            action = jnp.argmax(policy_weights)
        else:
            # Apply temperature: p_i ∝ n_i^(1/temperature)
            temperature_adjusted_weights = policy_weights ** (1.0 / self.temperature)
            normalized_weights = temperature_adjusted_weights / (jnp.sum(temperature_adjusted_weights) + 1e-8)
            
            # Ensure we only select from legal actions
            normalized_weights = jnp.where(legal_action_mask, normalized_weights, 0.0)
            normalized_weights = normalized_weights / (jnp.sum(normalized_weights) + 1e-8)
            
            action = jax.random.choice(key, jnp.arange(self.branching_factor), p=normalized_weights)
        
        # Safeguard: If we somehow selected an illegal action, pick the first legal one
        # Using JAX-friendly approach to find the first legal action
        def find_first_legal(legal_mask):
            # Initialize with default (0) in case no legal actions are found
            first_legal = jnp.array(0, dtype=jnp.int32)
            
            def body_fn(i, current_first):
                # If this action is legal and we haven't found a legal one yet
                # then update current_first
                is_legal = legal_mask[i]
                found_already = current_first < i
                return jnp.where(is_legal & ~found_already, i, current_first)
            
            return jax.lax.fori_loop(0, legal_mask.shape[0], body_fn, first_legal)
        
        # Check if the selected action is legal, if not, use the first legal action
        action = jax.lax.cond(
            legal_action_mask[action],
            lambda _: action,
            lambda _: find_first_legal(legal_action_mask),
            operand=None
        )

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
            # Make sure n is initialized as a proper JAX array
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
        return root_node.replace(
            p=root_policy,
            # keep old value estimate if the node has already been visited
            q=jnp.where(visited, root_node.q, root_value), 
            # keep old visit count if the node has already been visited
            n=jnp.where(visited, root_node.n, 1), 
            embedding=root_embedding
        )
    

    def reset(self, state: StochasticMCTSTree) -> StochasticMCTSTree:
        """Reset the search tree, clearing all nodes.
        
        Args:
        - `state`: current tree state
        
        Returns:
        - (StochasticMCTSTree): new empty tree initialized with the same capacity
        """
        debug_print(f"Performing StochasticMCTS reset on tree of size {state.next_free_idx}", level=1)
    
        # Create a brand new tree with the same capacity
        # reset_tree = init_tree(
        #     root_embedding=state.data_at(state.ROOT_INDEX).embedding,
        #     capacity=state.capacity,
        # )
        # Correctly call init_tree using parameters from self and a template node
        template_embedding = state.data_at(state.ROOT_INDEX).embedding
        template_node = MCTSNode(
            p=jnp.zeros(self.branching_factor),
            q=jnp.array(0.0, dtype=jnp.float32),
            n=jnp.array(0, dtype=jnp.int32),
            embedding=template_embedding, # Use embedding from old root as template
            terminated=jnp.array(False)
        )
        base_tree = init_tree(self.max_nodes, self.branching_factor, template_node)
        node_is_stochastic = jnp.zeros(self.max_nodes, dtype=jnp.bool_)
        reset_tree = StochasticMCTSTree(
            next_free_idx=base_tree.next_free_idx,
            parents=base_tree.parents,
            edge_map=base_tree.edge_map,
            data=base_tree.data,
            node_is_stochastic=node_is_stochastic
        )

        # Also need to initialize the root node data itself in the reset tree
        root_node = self.new_node( # Use new_node to create the initial root data
            policy=jnp.ones(self.branching_factor) / self.branching_factor, # Uniform policy
            value=0.0, # Initial value guess
            embedding=template_embedding,
            terminated=jnp.array(False) # Assume root is not terminal
        )
        reset_tree = reset_tree.set_root(root_node) # Use set_root to initialize index 0

        debug_print(f"Reset complete. New tree size: {reset_tree.next_free_idx}")
        return reset_tree
    
    
    def step(self, state: StochasticMCTSTree, action: int) -> StochasticMCTSTree:
        """Moves the root of the tree to the child node corresponding to the given action.
        This operation is important for maintaining and reusing the tree across agent steps.
        
        Args:
        - `state`: `StochasticMCTSTree` current tree state
        - `action`: action taken by agent, used to select new root node
        
        Returns:
        - (StochasticMCTSTree): updated tree with root moved to child node
        """
        debug_print(f"step() called with action {action}", level=1)
        child_index = state.edge_map[state.ROOT_INDEX, action]
        
        # Check if child exists
        child_exists = state.is_edge(state.ROOT_INDEX, action)
        
        def _move_root(operand):
            """Moves the root to the child node using get_subtree instead of move_root."""
            state, action = operand

            debug_print(f"Moving root to child node (action={action})", level=2)
            debug_print(f"Original tree: size={state.next_free_idx}, root_node at {state.ROOT_INDEX}", level=2)

            # Get the child index for the chosen action
            child_index = state.edge_map[state.ROOT_INDEX, action]
            debug_print(f"Child index: {child_index}", level=2)

            # Use jax.lax.cond instead of Python if/else to handle traced values
            def reset_fn(args):
                s, _ = args
                debug_print("Child index is NULL_INDEX, resetting tree", level=2)
                return self.reset(s)
            
            def extract_subtree_fn(args):
                s, a = args
                # <<< FIX: Get node_is_stochastic flags *before* subtree extraction >>>
                # Get mapping from old indices to new indices
                old_subtree_idxs, translation, erase_idxs = s._get_translation(a)
                new_size = translation.max() + 1
                
                # Create the new stochastic flags array
                # Initialize with False (or a default value)
                new_stochastic_flags = jnp.zeros(s.capacity, dtype=jnp.bool_)
                
                # Use the translation map to copy flags from old tree to new tree
                # Only copy flags for nodes that are retained (translation != -1)
                # Source indices are the indices in the original tree (old_subtree_idxs)
                # Target indices are the new indices in the pruned tree (translation[old_subtree_idxs])
                
                # Helper function to update flags using scatter (avoids loops)
                def update_flags(old_indices, trans, old_flags):
                    # Get the new indices corresponding to the old indices 
                    target_indices = trans[old_indices]
                    # Get the flags from the old array at the old indices
                    values_to_copy = old_flags[old_indices]
                    
                    # Initialize the new flags array
                    scattered_flags = jnp.zeros(s.capacity, dtype=jnp.bool_)
                    
                    # Iterate through the indices and copy if the target is valid
                    # Note: Using fori_loop for compatibility with JIT
                    def loop_body(i, current_flags):
                        target_idx = target_indices[i]
                        value_to_set = values_to_copy[i]
                        # Only update if the target index is valid (not NULL_INDEX)
                        return jnp.where(target_idx != s.NULL_INDEX, 
                                         current_flags.at[target_idx].set(value_to_set), 
                                         current_flags)

                    # Run the loop over all potential old indices
                    scattered_flags = jax.lax.fori_loop(0, old_indices.shape[0], loop_body, scattered_flags)
                    return scattered_flags

                # Populate the new stochastic flags array
                new_stochastic_flags = update_flags(old_subtree_idxs, translation, s.node_is_stochastic)
                # === DEBUG: Check new flags ===
                debug_print(f"extract_subtree_fn: New stochastic flags (first 10): {new_stochastic_flags[:10]}", level=1)
                # === END DEBUG ===
                # <<< END FIX >>>

                # Now get the base tree using the standard get_subtree
                new_base_tree = s.get_subtree(a)
                debug_print(f"Extracted subtree with new size={new_base_tree.next_free_idx}", level=2)
                
                # Construct the new StochasticMCTSTree using the base tree's structure
                # and the *correctly mapped* stochastic flags array
                final_tree = StochasticMCTSTree(
                    next_free_idx=new_base_tree.next_free_idx,
                    parents=new_base_tree.parents,
                    edge_map=new_base_tree.edge_map,
                    data=new_base_tree.data,
                    node_is_stochastic=new_stochastic_flags # Use the newly created flags
                )
                debug_print(f"Created new StochasticMCTSTree with size: {final_tree.next_free_idx}", level=2)
                return final_tree
            
            # Use jax.lax.cond for conditional logic with traced values
            return jax.lax.cond(
                child_index == state.NULL_INDEX,
                reset_fn,
                extract_subtree_fn,
                (state, action)
            )

        def _reset_tree(operand):
            """Resets the tree (e.g., if child doesn't exist or persist_tree is False)."""
            state, _ = operand
            # Call the actual reset method now (even if it's just a placeholder)
            return self.reset(state)
        
        # Choose whether to move root or reset
        updated_state = jax.lax.cond(
            jnp.logical_and(child_exists, self.persist_tree),
            _move_root, 
            _reset_tree,
            (state, action) # Pass action to _move_root
        )
        
        return updated_state


    def init(self, template_embedding: chex.ArrayTree, *args, **kwargs) -> StochasticMCTSTree: #pylint: disable=arguments-differ
        """Initializes the internal state of the MCTS evaluator.
        
        Args:
        - `template_embedding`: template environment state used for initializing the tree structure.

        Returns:
        - (StochasticMCTSTree): initialized StochasticMCTSTree
        """
        # Initialize the core Tree object using the provided function
        base_tree = init_tree(self.max_nodes, self.branching_factor, self.new_node(
            policy=jnp.zeros((self.branching_factor,)),
            value=0.0,
            embedding=template_embedding,
            terminated=False
        ))
        
        # Initialize the stochastic node tracker (all False initially)
        node_is_stochastic_array = jnp.zeros(self.max_nodes, dtype=jnp.bool_)
        
        # Construct StochasticMCTSTree using fields from base_tree and adding the new field.
        # Assumes StochasticMCTSTree (via MCTSTree) has corresponding fields.
        return StochasticMCTSTree(
            next_free_idx=base_tree.next_free_idx,
            parents=base_tree.parents,
            edge_map=base_tree.edge_map,
            data=base_tree.data,
            node_is_stochastic=node_is_stochastic_array  # Add our new field
            # ClassVars like ROOT_INDEX and properties like capacity/branching_factor are not passed here
        )

    # Implementation of the helper methods
    def _select_from_stochastic_root(self,
        root_key: chex.PRNGKey,
        action_key: chex.PRNGKey,
        eval_state: StochasticMCTSTree,
        stochastic_env_state: chex.ArrayTree,
        params: chex.ArrayTree,
        env_step_fn: EnvStepFn,
        root_metadata: StepMetadata
    ) -> Tuple[StochasticMCTSTree, int, chex.Array]:
        """Handles the case where the root node is stochastic.
        
        UPDATED DESIGN: Samples a stochastic action and returns it immediately.
        Does NOT perform MCTS search or step the environment.

        Args:
            root_key: JAX random key for sampling stochastic action.
            action_key: JAX random key (unused in this path).
            eval_state: The tree state passed to evaluate(). Returned unchanged.
            stochastic_env_state: The stochastic environment state (unused in this path).
            params: Network parameters (unused in this path).
            env_step_fn: Environment step function (unused in this path).
            root_metadata: Metadata for the stochastic root (unused in this path).

        Returns:
            Tuple[StochasticMCTSTree, int, chex.Array]:
                - Unchanged eval_state.
                - Sampled stochastic action index (as JAX array).
                - Dummy policy weights (e.g., uniform). 
        """
        debug_print("Selecting stochastic action from root without search", level=1)
        num_stochastic_outcomes = len(self.stochastic_action_probs)
        
        # --- Sample a stochastic outcome --- 
        sampled_stochastic_action_idx = jax.random.choice(
            key=root_key, 
            a=jnp.arange(num_stochastic_outcomes), 
            p=self.stochastic_action_probs
        )
        debug_print(f"_select_from_stochastic_root: Sampled stochastic action index: {sampled_stochastic_action_idx}", level=1)

        # Create dummy policy weights (not used for selection here)
        dummy_policy = jnp.zeros(self.branching_factor)

        # Return the original tree state, the sampled stochastic action, and dummy policy
        return eval_state, sampled_stochastic_action_idx, dummy_policy

    def _evaluate_deterministic_root(self,
        root_key: chex.PRNGKey,
        action_key: chex.PRNGKey,
        eval_state: StochasticMCTSTree,
        env_state: chex.ArrayTree,
        params: chex.ArrayTree,
        env_step_fn: EnvStepFn,
        root_metadata: StepMetadata
    ) -> Tuple[StochasticMCTSTree, int, chex.Array]:
        """ Handles evaluation starting from a deterministic root node.
        1. Updates the existing tree root to match the environment state.
        2. Runs MCTS iterations.

        Args:
            root_key: JAX random key for sampling deterministic action.
            action_key: JAX random key (unused in this path).
            eval_state: The tree state passed to evaluate(). Returned unchanged.
            env_state: The deterministic environment state.
            params: Network parameters (unused in this path).
            env_step_fn: Environment step function (unused in this path).
            root_metadata: Metadata for the deterministic root (unused in this path).

        Returns:
            Tuple[StochasticMCTSTree, int, chex.Array]:
                - Unchanged eval_state.
                - Sampled deterministic action index (as JAX array).
                - Deterministic action probabilities as policy weights (as JAX array).
        """
        debug_print("_evaluate_deterministic_root entered", level=1)

        # 1. Update the existing tree root to match the environment state.
        updated_tree = self.update_root(root_key, eval_state, env_state, params, root_metadata, env_step_fn)
        
        # 2. Sample an action from the updated tree
        action, policy_weights = self.sample_root_action(action_key, updated_tree)
        
        # 3. Return the updated tree, the sampled action, and the policy weights
        return updated_tree, action, policy_weights

