import jax
import jax.numpy as jnp
from functools import partial

import pgx.backgammon as bg
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.evaluators.mcts.action_selection import PUCTSelector
from core.types import StepMetadata
import chex
from chex import ArrayTree, PRNGKey

env = bg.Backgammon(simple_doubles=True)

@jax.jit
def backgammon_eval_fn(state: chex.ArrayTree, params: chex.ArrayTree, key: chex.PRNGKey):
    """Calculates value based on pip count difference. Ignores params/key."""
    board = state._board
    pips = state._board[1:25]
    
    born_off_current = board[26] * 30
    born_off_opponent = board[27] * 30
    
    #ignore bar, basically 0 points per pip on bar
    
    point_map = jnp.arange(1, 25, dtype=jnp.int32)
    
    value = jnp.sum(pips * point_map) + born_off_current + born_off_opponent
    
    # Uniform policy over legal actions for greedy baseline
    policy_logits = jnp.where(state.legal_action_mask, 0.0, -jnp.inf)
    
    return policy_logits, jnp.array(value)



def backgammon_step_fn(state: bg.State, action: int, key: chex.PRNGKey):
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
        num_iterations=20, # Player 0 iterations
        stochastic_action_probs=env.stochastic_action_probs,
        discount=-1.0,
        temperature=0.0, # Greedy action selection
        persist_tree=True, # Player 0 persists tree
    )

    mcts_p1 = StochasticMCTS(
        eval_fn=backgammon_eval_fn,
        action_selector=PUCTSelector(),
        branching_factor=env.num_actions,
        max_nodes=200, 
        num_iterations=1, # Player 1 iterations
        stochastic_action_probs=env.stochastic_action_probs,
        discount=-1.0,
        temperature=0.0, # Greedy action selection
        persist_tree=False, # Player 1 does NOT persist tree
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
            env_step_fn=backgammon_step_fn
        )

        action = mcts_output.action
        
        if current_player == 0:
            eval_state_p0 = mcts_output.eval_state             
        else:
            eval_state_p1 = mcts_output.eval_state

        # --- CRITICAL CHECK: Verify action legality ---
        # is_stochastic = bool(state.is_stochastic) # Redundant, already defined
        if is_stochastic:
            # For stochastic states, check if action is within the valid range of stochastic outcomes
            num_stochastic_outcomes = len(env.stochastic_action_probs)
            is_legal = (0 <= action) & (action < num_stochastic_outcomes)
            action_type = "stochastic"
            action_str = bg.stochastic_action_to_str(action)
        else:
            # For deterministic states, check the legal action mask
            is_legal = state.legal_action_mask[action] == 1
            action_type = "deterministic"
            action_str = bg.action_to_str(action)


        print(f"Step {step_count}, Player: {current_player}, Stochastic: {is_stochastic}, Action: {action_str}, Action Str: {action_str}")

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


        assert bool(is_legal), (
            f"Step {step_count}: MCTS selected illegal {action_type} action {action}. "
            f"Player: {current_player}, Stochastic: {is_stochastic}. "
            f"Action Str: {action_str}"
            f"{('Legal mask: ' + str(state.legal_action_mask)) if not is_stochastic else ('Expected range: 0-' + str(num_stochastic_outcomes-1))}"
        )
        # --- End Check ---

        # Step environment and MCTS tree
        state, metadata = backgammon_step_fn(state, action, step_key)
        # Call step on both evaluators
        eval_state_p0 = mcts_evaluator.step(eval_state_p0, action)
        eval_state_p1 = mcts_evaluator.step(eval_state_p1, action)

        if bool(state.terminated):
            print(f"Game terminated at step {step_count}. Final rewards: {state.rewards}")

    assert bool(state.terminated), f"Game did not terminate within {max_steps} steps."
    print(f"Game completed successfully in {step_count} steps.")

def test_traverse_through_stochastic_nodes():
    """
    Test that MCTS can traverse through all node types during the evaluate
    
    This verifies the fix for allowing MCTS to continue exploration through stochastic
    nodes in subsequent iterations during evaluate.
    """
    # Initialize environment and MCTS
    key = jax.random.PRNGKey(42)

    # Initialize the game state
    key, init_key, eval_key, step_key = jax.random.split(key, 4)  # Add step_key
    

    state = env.init(init_key)
    
    state, metadata = backgammon_step_fn(state, 3, step_key) # dice roll of 4-4
    
    # Set up MCTS with high iteration count for deep exploration
    mcts = StochasticMCTS(
        eval_fn=backgammon_eval_fn,
        action_selector=PUCTSelector(),
        branching_factor=env.num_actions,
        max_nodes=500,  # Large enough to store many nodes
        num_iterations=400,  # Enough iterations to explore deeply
        stochastic_action_probs=env.stochastic_action_probs,
        discount=-1.0,
        temperature=1.0,  # Some exploration temperature
        persist_tree=True # Enable some debugging output
    )
    
    # Initialize evaluation state and metadata
    eval_state = mcts.init(template_embedding=state)

    metadata = StepMetadata(
        rewards=state.rewards,
        action_mask=state.legal_action_mask,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state._step_count
    )

    # do an initial eval.
    mcts_output = mcts.evaluate(
        key=eval_key,
        eval_state=eval_state,
        env_state=state,
        root_metadata=metadata,
        params={},
        env_step_fn=backgammon_step_fn
    )
    
    # Helper function to analyze the tree structure
    def find_deepest_path(tree):
        """Find the path to the deepest node in the tree.
        Returns only valid continuous parent-child paths from root to a leaf."""
        max_depth = 0
        deepest_node_idx = tree.ROOT_INDEX
        
        # Use DFS to find the deepest node
        def dfs(node_idx, current_depth, current_path):
            nonlocal max_depth, deepest_node_idx
            
            # Check if this node has any children
            has_children = False
            for a in range(tree.branching_factor):
                child_idx = tree.edge_map[node_idx, a]
                if child_idx != -1:  # Valid edge
                    has_children = True
                    # Verify this is a valid parent-child relationship
                    is_valid_edge = tree.parents[child_idx] == node_idx
                    if not is_valid_edge:
                        print(f"Warning: Invalid edge from {node_idx} to {child_idx}. Parent of {child_idx} is {tree.parents[child_idx]}")
                        continue
                    
                    # Continue DFS with the child added to path
                    dfs(child_idx, current_depth + 1, current_path + [child_idx])
            
            # If it's a leaf node and deeper than what we've seen before
            if not has_children and current_depth > max_depth:
                max_depth = current_depth
                deepest_node_idx = node_idx
                print(f"Found new deepest path with depth {current_depth}: {current_path}")
                
                # Validate the entire path to make sure it's proper parent-c
        
        # Start DFS from root node (depth 0)
        dfs(tree.ROOT_INDEX, 0, [tree.ROOT_INDEX])
        
        return max_depth, deepest_node_idx
    
    # Analyze the tree
    tree = mcts_output.eval_state
    max_depth, deepest_node_idx = find_deepest_path(tree)
    
    # Verify the path is actually valid
    current_node_idx = deepest_node_idx
    num_transitions = 0
    total_depth = 0
  
    while tree.parents[current_node_idx] != tree.ROOT_INDEX:
        if StochasticMCTS.is_node_idx_stochastic(tree, current_node_idx) != \
              StochasticMCTS.is_node_idx_stochastic(tree, tree.parents[current_node_idx]):
            num_transitions += 1
        current_node_idx = tree.parents[current_node_idx]
        total_depth += 1
        #print the node visit count
        print(f"Node {current_node_idx} visit count: {tree.data_at(current_node_idx).n}")
        assert tree.data_at(tree.parents[current_node_idx]).n >= tree.data_at(current_node_idx).n, \
            f"Parent node {tree.parents[current_node_idx]} visit count ({tree.data_at(tree.parents[current_node_idx]).n}) should be >= child node {current_node_idx} visit count ({tree.data_at(current_node_idx).n})"
        assert total_depth < 1000, "Infinite loop detected"  



    
    assert max_depth >= 4, f"Expected path length of at least 3, got {max_depth}"
    assert max_depth <= 105, f"Path length ({max_depth}) should not exceed iteration count"
    
 
    print(f"Number of transitions between stochastic and deterministic nodes: {num_transitions}")
    assert num_transitions >= 1, "Expected at least one transition between stochastic and deterministic nodes"
    
    print("Successfully verified that MCTS can traverse through stochastic nodes") 