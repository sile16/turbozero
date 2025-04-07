import pytest
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

import pgx.backgammon as bg
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS, StochasticMCTSTree
from core.evaluators.mcts.action_selection import PUCTSelector
from core.types import StepMetadata

# --- Helper Functions (adapted from visualization script) ---

@jax.jit
def backgammon_eval_fn(state, params, key):
    """Simple evaluation function for backgammon (simple_doubles)."""
    num_actions = state.legal_action_mask.shape[0] # Env provides num_actions
    # Generate random policy logits for testing
    policy_key, value_key = jax.random.split(key)
    policy_logits = jax.random.normal(policy_key, shape=(num_actions,))

    # Simple value: +1 for player 0 winning, -1 for player 1 winning, 0 otherwise
    # More sophisticated eval could use pip count, but this is simpler for testing validity
    value = jnp.select(
        [state.terminated & (state.rewards[0] > 0), state.terminated & (state.rewards[1] > 0)],
        [1.0, -1.0],
        default=0.0
    )
    # Adjust value for player perspective
    value = jnp.where(state.current_player == 0, value, -value)

    # Ensure stochastic states are not evaluated directly by the network
    # MCTS internal logic handles stochastic nodes based on probabilities
    value = jnp.where(state.is_stochastic, jnp.nan, value)
    policy_logits = jnp.where(state.is_stochastic,
                              jnp.ones_like(policy_logits) * jnp.nan,
                              policy_logits)

    return policy_logits, value

def backgammon_step_fn(env, state, action, key):
    """Step function for backgammon that handles both deterministic and stochastic actions."""
    # Handle stochastic vs deterministic branches
    def stochastic_branch(s, a, k):
        # Stochastic steps don't use a key in PGX v1
        return env.stochastic_step(s, a)

    def deterministic_branch(s, a, k):
        # Deterministic steps require a key
        return env.step(s, a, k)

    # Use conditional to route to the appropriate branch
    new_state = jax.lax.cond(
        state.is_stochastic,
        stochastic_branch,
        deterministic_branch,
        state, action, key # Pass key only needed by deterministic branch
    )

    # Create standard metadata
    metadata = StepMetadata(
        rewards=new_state.rewards,
        action_mask=new_state.legal_action_mask,
        terminated=new_state.terminated,
        cur_player_id=new_state.current_player,
        step=new_state._step_count # Assuming _step_count exists
    )

    return new_state, metadata

# --- Test Case ---

def test_stochastic_mcts_backgammon_simple_doubles_valid_actions():
    """
    Tests a full backgammon game (simple_doubles) using two StochasticMCTS agents
    with different configurations, ensuring all selected actions are legal.
    Player 0: num_iterations=4, persist_tree=True
    Player 1: num_iterations=6, persist_tree=False
    """
    key = jax.random.PRNGKey(44) # Use a different seed
    env = bg.Backgammon(simple_doubles=True)

    # --- MCTS setup for two players ---
    mcts_p0 = StochasticMCTS(
        eval_fn=backgammon_eval_fn,
        action_selector=PUCTSelector(),
        branching_factor=env.num_actions,
        max_nodes=200, 
        num_iterations=2, # Player 0 iterations
        stochastic_action_probs=env.stochastic_action_probs,
        discount=-1.0,
        temperature=0.0, # Greedy action selection
        persist_tree=True, # Player 0 persists tree
        debug_level=0
    )

    mcts_p1 = StochasticMCTS(
        eval_fn=backgammon_eval_fn,
        action_selector=PUCTSelector(),
        branching_factor=env.num_actions,
        max_nodes=200, 
        num_iterations=4, # Player 1 iterations
        stochastic_action_probs=env.stochastic_action_probs,
        discount=-1.0,
        temperature=0.0, # Greedy action selection
        persist_tree=False, # Player 1 does NOT persist tree
        debug_level=0
    )

    # --- Initialization ---
    key, init_key = jax.random.split(key)
    state = env.init(init_key)
    
    # Initialize separate tree states for each player
    eval_state_p0 = mcts_p0.init(template_embedding=state)
    eval_state_p1 = mcts_p1.init(template_embedding=state)
    
    metadata = StepMetadata(
        rewards=state.rewards,
        action_mask=state.legal_action_mask,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state._step_count
    )

    max_steps = 500 # Safety limit
    step_count = 0

    while not bool(state.terminated) and step_count < max_steps:
        step_count += 1
        current_player = int(state.current_player) # Ensure Python int
        is_stochastic = bool(state.is_stochastic)
        print(f"Step {step_count}, Player: {current_player}, Stochastic: {is_stochastic}")

        key, eval_key, step_key = jax.random.split(key, 3)

        # Select evaluator and tree state based on current player
        if current_player == 0:
            mcts_evaluator = mcts_p0
            current_eval_state = eval_state_p0
            #print(f"  Using Player 0 MCTS (iters={mcts_evaluator.num_iterations}, persist={mcts_evaluator.persist_tree})")
        else:
            mcts_evaluator = mcts_p1
            current_eval_state = eval_state_p1
            #print(f"  Using Player 1 MCTS (iters={mcts_evaluator.num_iterations}, persist={mcts_evaluator.persist_tree})")

        # Evaluate with the selected MCTS evaluator
        mcts_output = mcts_evaluator.evaluate(
            key=eval_key,
            eval_state=current_eval_state, # Pass the correct player's tree state
            env_state=state,
            root_metadata=metadata,
            params={}, # No learned parameters needed for this test
            env_step_fn=partial(backgammon_step_fn, env)
        )

        action = mcts_output.action
        # IMPORTANT: Use the tree state *returned* by evaluate, 
        # as it might have been updated internally (especially if persist=True)
        updated_eval_state = mcts_output.eval_state 

        # --- CRITICAL CHECK: Verify action legality ---
        # is_stochastic = bool(state.is_stochastic) # Redundant, already defined
        if is_stochastic:
            # For stochastic states, check if action is within the valid range of stochastic outcomes
            num_stochastic_outcomes = len(env.stochastic_action_probs)
            is_legal = (0 <= action) & (action < num_stochastic_outcomes)
            action_type = "stochastic"
        else:
            # For deterministic states, check the legal action mask
            is_legal = state.legal_action_mask[action] == 1
            action_type = "deterministic"
        
        if not bool(is_legal): # Use bool() for clarity with JAX arrays
             print(f"ILLEGAL {action_type.upper()} ACTION selected at step {step_count}!")
             print(f"  Player: {current_player}")
             print(f"  Is Stochastic State: {is_stochastic}")
             print(f"  Selected Action: {action}")
             if is_stochastic:
                 print(f"  Action Str: {bg.stochastic_action_to_str(action)}")
                 print(f"  Expected Range: 0 to {num_stochastic_outcomes - 1}")
             else:
                 print(f"  Action Str: {bg.action_to_str(action)}")
                 print(f"  Legal Mask (len {len(state.legal_action_mask)}):")
                 print(f"  {state.legal_action_mask}")
             # Optionally print board state or other debug info here
             # import sys; sys.exit() # or raise error

        assert bool(is_legal), (
            f"Step {step_count}: MCTS selected illegal {action_type} action {action}. "
            f"Player: {current_player}, Stochastic: {is_stochastic}. "
            f"{('Legal mask: ' + str(state.legal_action_mask)) if not is_stochastic else ('Expected range: 0-' + str(num_stochastic_outcomes-1))}"
        )
        # --- End Check ---

        # Step environment and MCTS tree
        state, metadata = backgammon_step_fn(env, state, action, step_key)
        # Call step on the *correct* evaluator and its *updated* state from evaluate
        stepped_eval_state = mcts_evaluator.step(updated_eval_state, action)

        # Update the corresponding player's tree state for the next turn
        if current_player == 0:
            eval_state_p0 = stepped_eval_state
        else:
            eval_state_p1 = stepped_eval_state

        if bool(state.terminated):
            print(f"Game terminated at step {step_count}. Final rewards: {state.rewards}")

    assert bool(state.terminated), f"Game did not terminate within {max_steps} steps."
    print(f"Game completed successfully in {step_count} steps.")

def test_traverse_through_stochastic_nodes():
    """
    Test that MCTS can traverse through stochastic nodes and expand at least 2 levels 
    of deterministic nodes after a stochastic node.
    
    This verifies the fix for allowing MCTS to continue exploration through stochastic
    nodes in subsequent iterations.
    """
    # Initialize environment and MCTS
    key = jax.random.PRNGKey(42)
    env = bg.Backgammon(simple_doubles=True)
    
    # Set up MCTS with high iteration count for deep exploration
    mcts = StochasticMCTS(
        eval_fn=backgammon_eval_fn,
        action_selector=PUCTSelector(),
        branching_factor=env.num_actions,
        max_nodes=500,  # Large enough to store many nodes
        num_iterations=60,  # Enough iterations to explore deeply
        stochastic_action_probs=env.stochastic_action_probs,
        discount=-1.0,
        temperature=1.0,  # Some exploration temperature
        persist_tree=True,
        debug_level=1  # Enable some debugging output
    )
    
    # Initialize the game state
    key, init_key = jax.random.split(key)
    state = env.init(init_key)
    
    # Initialize evaluation state and metadata
    eval_state = mcts.init(template_embedding=state)
    metadata = StepMetadata(
        rewards=state.rewards,
        action_mask=state.legal_action_mask,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state._step_count
    )
    
    # First, get through the initial stochastic state (dice roll)
    key, eval_key = jax.random.split(key)
    print("Starting with initial stochastic state (dice roll)")
    mcts_output = mcts.evaluate(
        key=eval_key,
        eval_state=eval_state,
        env_state=state,
        root_metadata=metadata,
        params={},
        env_step_fn=partial(backgammon_step_fn, env)
    )
    
    # Apply stochastic action (dice roll)
    action = mcts_output.action
    eval_state = mcts_output.eval_state
    
    key, step_key = jax.random.split(key)
    state, metadata = backgammon_step_fn(env, state, action, step_key)
    eval_state = mcts.step(eval_state, action)
    
    print(f"Applied stochastic action (dice roll): {action}, {bg.stochastic_action_to_str(action)}")
    print(f"New state is_stochastic: {state.is_stochastic}")
    
    # Now we should be in a deterministic state. Make a move.
    key, eval_key = jax.random.split(key)
    print("Making first deterministic move")
    mcts_output = mcts.evaluate(
        key=eval_key,
        eval_state=eval_state,
        env_state=state,
        root_metadata=metadata,
        params={},
        env_step_fn=partial(backgammon_step_fn, env)
    )
    
    # Apply deterministic action
    action = mcts_output.action
    eval_state = mcts_output.eval_state
    
    key, step_key = jax.random.split(key)
    state, metadata = backgammon_step_fn(env, state, action, step_key)
    eval_state = mcts.step(eval_state, action)
    
    print(f"Applied deterministic action: {action}, {bg.action_to_str(action)}")
    print(f"New state is_stochastic: {state.is_stochastic}")
    
    # Force a stochastic state by manipulating the env if necessary
    # For Backgammon, a stochastic state will appear after a turn completes
    # Continue making moves until we get to a stochastic state
    
    # If not already in a stochastic state, continue until we reach one
    while not state.is_stochastic and not state.terminated:
        key, eval_key = jax.random.split(key)
        mcts_output = mcts.evaluate(
            key=eval_key,
            eval_state=eval_state,
            env_state=state,
            root_metadata=metadata,
            params={},
            env_step_fn=partial(backgammon_step_fn, env)
        )
        
        action = mcts_output.action
        eval_state = mcts_output.eval_state
        
        key, step_key = jax.random.split(key)
        state, metadata = backgammon_step_fn(env, state, action, step_key)
        eval_state = mcts.step(eval_state, action)
        
        print(f"Applied action to reach stochastic state: {action}")
        print(f"New state is_stochastic: {state.is_stochastic}")
    
    # Check that we now have a stochastic state
    assert state.is_stochastic, "Failed to reach a stochastic state for testing"
    print("Successfully reached a stochastic state")
    
    # Now run MCTS on this stochastic state with high iteration count
    # We expect it to expand all stochastic outcomes and continue exploration
    key, eval_key = jax.random.split(key)
    
    # Create a fresh MCTS instance with high iterations to ensure deep exploration
    explore_mcts = StochasticMCTS(
        eval_fn=backgammon_eval_fn,
        action_selector=PUCTSelector(),
        branching_factor=env.num_actions,
        max_nodes=1000,  # Very large to accommodate deep exploration
        num_iterations=60,  # Enough iterations to explore deeply
        stochastic_action_probs=env.stochastic_action_probs,
        discount=-1.0,
        temperature=1.0,
        persist_tree=True,
        debug_level=1
    )
    
    # Initialize a fresh tree
    explore_eval_state = explore_mcts.init(template_embedding=state)
    
    # Run MCTS with high iteration count
    print(f"Running MCTS with {explore_mcts.num_iterations} iterations on stochastic state")
    mcts_output = explore_mcts.evaluate(
        key=eval_key,
        eval_state=explore_eval_state,
        env_state=state,
        root_metadata=metadata,
        params={},
        env_step_fn=partial(backgammon_step_fn, env)
    )
    
    # Now examine the tree to verify it explored through stochastic nodes
    print(f"MCTS tree size after evaluation: {mcts_output.eval_state.next_free_idx} nodes")
    
    # Helper function to analyze the tree structure
    def find_deepest_path_through_stochastic(tree):
        """Find the deepest path that goes through at least one stochastic node."""
        # First, identify stochastic nodes
        stochastic_nodes = []
        for i in range(tree.next_free_idx):
            if tree.node_is_stochastic[i]:
                stochastic_nodes.append(i)
        
        print(f"Found {len(stochastic_nodes)} stochastic nodes in the tree")
        
        # For each stochastic node, find the maximum depth of deterministic nodes after it
        max_det_depth_after_stoch = 0
        max_depth_stoch_node = -1
        max_depth_path = []
        
        for stoch_node_idx in stochastic_nodes:
            # Get all children of the stochastic node
            stoch_children = []
            for a in range(tree.branching_factor):
                child_idx = tree.edge_map[stoch_node_idx, a]
                if child_idx != -1:  # Valid edge
                    stoch_children.append(child_idx)
            
            print(f"Stochastic node {stoch_node_idx} has {len(stoch_children)} children")
            
            # For each child of the stochastic node, find the maximum depth of deterministic nodes
            for child_idx in stoch_children:
                # Skip if child is itself stochastic
                if tree.node_is_stochastic[child_idx]:
                    continue
                
                # DFS to find maximum depth of deterministic nodes
                def dfs(node_idx, depth, path):
                    # Check if this is a leaf
                    has_children = False
                    max_depth = depth
                    best_path = path
                    
                    # Explore all children
                    for a in range(tree.branching_factor):
                        next_idx = tree.edge_map[node_idx, a]
                        if next_idx != -1:  # Valid edge
                            has_children = True
                            # Only continue through deterministic nodes
                            if not tree.node_is_stochastic[next_idx]:
                                child_depth, child_path = dfs(next_idx, depth + 1, path + [next_idx])
                                if child_depth > max_depth:
                                    max_depth = child_depth
                                    best_path = child_path
                    
                    return max_depth, best_path
                
                det_depth, det_path = dfs(child_idx, 1, [stoch_node_idx, child_idx])  # Start depth at 1 for this deterministic node
                
                if det_depth > max_det_depth_after_stoch:
                    max_det_depth_after_stoch = det_depth
                    max_depth_stoch_node = stoch_node_idx
                    max_depth_path = det_path
        
        return max_det_depth_after_stoch, max_depth_stoch_node, max_depth_path
    
    # Analyze the tree
    max_depth, stoch_node, path = find_deepest_path_through_stochastic(mcts_output.eval_state)
    
    print(f"Maximum depth of deterministic nodes after a stochastic node: {max_depth}")
    print(f"Stochastic node with deepest deterministic path: {stoch_node}")
    print(f"Path with maximum depth: {path}")
    
    # Verify that there are at least 2 levels of deterministic nodes after a stochastic node
    assert max_depth >= 2, f"Expected at least 2 levels of deterministic nodes after a stochastic node, got {max_depth}"
    
    print("Successfully verified that MCTS can traverse through stochastic nodes") 