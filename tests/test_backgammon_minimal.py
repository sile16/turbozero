"""
Test the basic functionality of the Backgammon environment and evaluator.
"""

from functools import partial
import jax
import jax.numpy as jnp
import pgx.backgammon as bg

from core.bgcommon import  bg_simple_step_fn, bg_pip_count_eval, bg_hit2_eval

from core.types import StepMetadata
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS


def test_backgammon_basics():
    """Test basic functionality of Backgammon environment."""
    print("Testing backgammon environment basic functionality...")
    
    # --- Environment Setup ---
    env = bg.Backgammon(simple_doubles=True)
    num_actions = env.num_actions
    print(f"NUM_ACTIONS: {num_actions}")
    assert num_actions == 156
    
    # --- Get test observation ---
    key = jax.random.PRNGKey(0)
    init_state = env.init(key)
    observation_shape = init_state.observation.shape
    print(f"Detected Observation Shape: {observation_shape}")
    assert observation_shape == (34,)
    
    stochastic_probs = env.stochastic_action_probs
    print(f"STOCHASTIC_PROBS: {stochastic_probs}")
    assert len(stochastic_probs) == 21  # should be 21 no matter what
    
    # --- Test basic environment operations ---
    # 1. Initial state should be stochastic (need to roll dice)
    assert init_state._is_stochastic
    assert not init_state.terminated
    
    # 2. Test stochastic step (dice roll)
    stochastic_action = 0  # choose first dice roll
    after_roll_state = env.stochastic_step(init_state, stochastic_action)
    
    # After rolling dice, state should be deterministic
    assert not after_roll_state._is_stochastic
    assert not after_roll_state.terminated
    
    # 3. Test a regular move
    # Find a legal action
    legal_actions = jnp.where(after_roll_state.legal_action_mask)[0]
    assert len(legal_actions) > 0, "No legal actions available after dice roll"
    
    print("Basic environment operations tested successfully!")

def test_backgammon_evaluator():
    """Test basic functionality of evaluator with Backgammon environment."""
    print("\nTesting backgammon evaluator basic functionality...")
    
    # --- Environment Setup ---
    env = bg.Backgammon(simple_doubles=True)
    num_actions = env.num_actions
    
    # --- Get test observation ---
    key = jax.random.PRNGKey(0)
    init_state = env.init(key)
    
   
    
    # --- Basic MCTS setup ---
    mcts = MCTS(
        eval_fn=bg_hit2_eval,
        num_iterations=2,  # Just 2 iterations for testing
        max_nodes=10,
        branching_factor=num_actions,
        action_selector=PUCTSelector(),
        temperature=0.0,
    )
    
    # --- Initialize evaluator ---
    eval_state = mcts.init(template_embedding=init_state)
    assert eval_state is not None
    
    # --- Roll dice ---
    stochastic_action = 0
    after_roll_state = env.stochastic_step(init_state, stochastic_action)
    assert not after_roll_state._is_stochastic
    
    # --- Create metadata ---
    metadata = StepMetadata(
        rewards=after_roll_state.rewards,
        action_mask=after_roll_state.legal_action_mask,
        terminated=after_roll_state.terminated,
        cur_player_id=after_roll_state.current_player,
        step=after_roll_state._step_count
    )
    
    # --- Test evaluator on deterministic state ---
    eval_key = jax.random.PRNGKey(1)

    step_fn = partial(bg_simple_step_fn, env)
    
    # Use empty params dict
    params = {}
    
    # Call evaluate method
    output = mcts.evaluate(
        key=eval_key,
        eval_state=eval_state,
        env_state=after_roll_state,
        root_metadata=metadata,
        params=params,
        env_step_fn=step_fn
    )
    
    # Verify we get valid outputs
    assert output is not None
    assert output.action is not None
    assert output.policy_weights is not None
    assert output.eval_state is not None
    
    # Ensure the selected action is legal
    assert after_roll_state.legal_action_mask[output.action]
    
    print("Evaluator test completed successfully!") 

def test_bg_pip_count_eval():
    """Test the pip count evaluator."""
    print("\nTesting pip count evaluator...")
    
    # --- Environment Setup ---
    env = bg.Backgammon(simple_doubles=True)
    num_actions = env.num_actions
    
    # Initialize game with fixed seed for reproducibility
    state = env.init(jax.random.PRNGKey(42))  # type: ignore

    dice = jnp.array([2, 3], dtype=jnp.int32)
    board = jnp.array([-1, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 12, -12], dtype=jnp.int32)  # type: ignore
    
    state = state.replace(
        _board=board,
    )
    state = env.set_dice(state, dice)

    count0 = bg_pip_count_eval(state, None, None)
    assert count0[1] == 0

    new_state = env.step(state, 122, jax.random.PRNGKey(0))
    count1 = bg_pip_count_eval(new_state, None, None)
    assert count1[1] > 0
    new_new_state = env.step(new_state, 135, jax.random.PRNGKey(0))

    # this changes player perspective, so we want to mkae sure count2 
    count2 = bg_pip_count_eval(new_new_state, None, None)
    
    print(f"Count1: {count1}, Count2: {count2}")
    assert (-1 * count2[1]) > count1[1]

    print("Pip count evaluator test completed successfully!")