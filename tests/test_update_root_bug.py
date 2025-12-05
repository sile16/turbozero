"""Tests for the update_root bug where Dirichlet noise isn't refreshed when persist_tree=False.

Engineer concern: _AlphaZero.update_root skips updating whenever root_n > 0, but MCTS.evaluate
deliberately calls update_root when persist_tree=False to refresh policy/value/embedding for
a new environment state. With the current guard, once the root has been visited, subsequent
non-persistent evaluations will reuse stale root data and skip the Dirichlet+policy/value
refresh—breaking the "reset each call" semantics.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from functools import partial

import pgx.backgammon as bg
from pgx.backgammon import State

from core.evaluators.mcts.action_selection import PUCTSelector
from core.types import StepMetadata

from core.bgcommon import bg_step_fn, bg_pip_count_eval


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def backgammon_env():
    return bg.Backgammon(simple_doubles=True)


@pytest.fixture
def mock_params():
    return {}


@jax.jit
def backgammon_eval_fn(state: State, params, key):
    return bg_pip_count_eval(state, params, key)


def backgammon_step_fn(env):
    return partial(bg_step_fn, env)


def test_alphazero_dirichlet_refresh_on_multiple_evaluates(backgammon_env, mock_params, key):
    """Test that AlphaZero refreshes Dirichlet noise on each evaluate call when persist_tree=False.

    This test validates the engineer's concern that update_root's root_n > 0 guard
    breaks the "reset each call" semantics for non-persistent trees.
    """
    # Import here to avoid circular import
    from core.evaluators.mcts.stochastic_mcts import StochasticMCTS

    key, init_key, eval_key1, eval_key2 = jax.random.split(key, 4)

    step_fn = backgammon_step_fn(backgammon_env)
    branching_factor = backgammon_env.num_actions
    stochastic_action_probs = backgammon_env.stochastic_action_probs

    # Create StochasticMCTS with persist_tree=False
    # StochasticMCTS already extends AlphaZero(MCTS), so it has Dirichlet noise
    alphazero = StochasticMCTS(
        eval_fn=backgammon_eval_fn,
        action_selector=PUCTSelector(),
        branching_factor=branching_factor,
        max_nodes=50,
        num_iterations=5,  # Few iterations to keep test fast
        discount=-1.0,
        temperature=0.0,
        persist_tree=False,  # Key: non-persistent tree
        stochastic_action_probs=stochastic_action_probs,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )

    # Initialize a deterministic state
    initial_state = backgammon_env.init(init_key)
    if initial_state._is_stochastic:
        initial_state = backgammon_env.stochastic_step(initial_state, 0)

    assert not initial_state._is_stochastic

    # Initialize tree
    eval_state = alphazero.init(template_embedding=initial_state)
    root_metadata = StepMetadata(
        rewards=initial_state.rewards,
        action_mask=initial_state.legal_action_mask,
        terminated=initial_state.terminated,
        cur_player_id=initial_state.current_player,
        step=initial_state._step_count
    )

    # First evaluate call
    output1 = alphazero.evaluate(
        key=eval_key1,
        eval_state=eval_state,
        env_state=initial_state,
        root_metadata=root_metadata,
        params=mock_params,
        env_step_fn=step_fn
    )

    tree1 = output1.eval_state
    root_policy1 = tree1.data.p[tree1.ROOT_INDEX]

    # Get the tree state after first evaluate (root should have visits now)
    root_n_after_first = tree1.data.n[tree1.ROOT_INDEX]
    assert root_n_after_first > 0, "Root should have visits after first evaluate"

    # Second evaluate call with SAME tree (simulates calling evaluate twice without step())
    # This is the bug scenario: persist_tree=False means each call should start fresh,
    # but if we don't reset the tree between calls, the root_n > 0 guard will skip
    # the Dirichlet refresh.
    output2 = alphazero.evaluate(
        key=eval_key2,  # Different key = different Dirichlet noise
        eval_state=tree1,  # Reuse tree from first call (bug scenario)
        env_state=initial_state,
        root_metadata=root_metadata,
        params=mock_params,
        env_step_fn=step_fn
    )

    tree2 = output2.eval_state
    root_policy2 = tree2.data.p[tree2.ROOT_INDEX]

    # The bug: if update_root skips the update because root_n > 0,
    # then root_policy1 == root_policy2 (stale policy with old Dirichlet noise)
    #
    # The fix: when persist_tree=False, update_root should ALWAYS refresh
    # the root node (policy with new Dirichlet noise, value, embedding)

    # With persist_tree=False, policies should be DIFFERENT due to different Dirichlet noise
    # (unless we get astronomically unlucky with the same random values)
    policies_are_identical = jnp.allclose(root_policy1, root_policy2, atol=1e-6)

    if policies_are_identical:
        print("BUG CONFIRMED: Root policies are identical even with different PRNG keys")
        print(f"Policy 1: {root_policy1[:10]}...")
        print(f"Policy 2: {root_policy2[:10]}...")
        print("This indicates Dirichlet noise was NOT refreshed on second evaluate")
    else:
        print("WORKING CORRECTLY: Root policies are different as expected")
        print(f"Policy 1: {root_policy1[:10]}...")
        print(f"Policy 2: {root_policy2[:10]}...")

    # This assertion will fail if the bug exists
    assert not policies_are_identical, (
        "BUG: With persist_tree=False, calling evaluate() twice with different keys "
        "should produce different root policies due to different Dirichlet noise. "
        "The root policies are identical, indicating the Dirichlet refresh was skipped."
    )


def test_alphazero_dirichlet_refresh_after_proper_reset(backgammon_env, mock_params, key):
    """Test that Dirichlet noise IS refreshed after properly resetting the tree.

    This shows the correct behavior: if we reset the tree between evaluate calls,
    the Dirichlet noise is properly refreshed.
    """
    # Import here to avoid circular import
    from core.evaluators.mcts.stochastic_mcts import StochasticMCTS

    key, init_key, eval_key1, eval_key2 = jax.random.split(key, 4)

    step_fn = backgammon_step_fn(backgammon_env)
    branching_factor = backgammon_env.num_actions
    stochastic_action_probs = backgammon_env.stochastic_action_probs

    # Create StochasticMCTS with persist_tree=False
    alphazero = StochasticMCTS(
        eval_fn=backgammon_eval_fn,
        action_selector=PUCTSelector(),
        branching_factor=branching_factor,
        max_nodes=50,
        num_iterations=5,
        discount=-1.0,
        temperature=0.0,
        persist_tree=False,
        stochastic_action_probs=stochastic_action_probs,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )

    # Initialize a deterministic state
    initial_state = backgammon_env.init(init_key)
    if initial_state._is_stochastic:
        initial_state = backgammon_env.stochastic_step(initial_state, 0)

    assert not initial_state._is_stochastic

    root_metadata = StepMetadata(
        rewards=initial_state.rewards,
        action_mask=initial_state.legal_action_mask,
        terminated=initial_state.terminated,
        cur_player_id=initial_state.current_player,
        step=initial_state._step_count
    )

    # First evaluate call
    eval_state1 = alphazero.init(template_embedding=initial_state)
    output1 = alphazero.evaluate(
        key=eval_key1,
        eval_state=eval_state1,
        env_state=initial_state,
        root_metadata=root_metadata,
        params=mock_params,
        env_step_fn=step_fn
    )

    root_policy1 = output1.eval_state.data.p[output1.eval_state.ROOT_INDEX]

    # Reset the tree before second evaluate (proper usage)
    eval_state2 = alphazero.init(template_embedding=initial_state)  # Fresh tree
    output2 = alphazero.evaluate(
        key=eval_key2,  # Different key = different Dirichlet noise
        eval_state=eval_state2,
        env_state=initial_state,
        root_metadata=root_metadata,
        params=mock_params,
        env_step_fn=step_fn
    )

    root_policy2 = output2.eval_state.data.p[output2.eval_state.ROOT_INDEX]

    # With fresh trees, policies should be different
    policies_are_different = not jnp.allclose(root_policy1, root_policy2, atol=1e-6)

    print("EXPECTED: With fresh trees, policies should differ")
    print(f"Policy 1: {root_policy1[:10]}...")
    print(f"Policy 2: {root_policy2[:10]}...")

    assert policies_are_different, (
        "With fresh trees, different PRNG keys should produce different Dirichlet noise"
    )


def test_update_root_node_preserves_visited_node(backgammon_env, key):
    """Test that update_root_node preserves data when node is already visited.

    This demonstrates the behavior of update_root_node() which preserves
    policy/value/embedding when visited=True, which is part of the issue.
    """
    from core.evaluators.mcts.mcts import MCTS
    from core.evaluators.mcts.state import MCTSNode

    key, init_key = jax.random.split(key, 2)

    initial_state = backgammon_env.init(init_key)
    if initial_state._is_stochastic:
        initial_state = backgammon_env.stochastic_step(initial_state, 0)

    branching_factor = backgammon_env.num_actions

    # Create a "visited" root node (n > 0)
    old_policy = jax.nn.softmax(jnp.ones(branching_factor))  # Uniform
    old_value_probs = jnp.ones(6, dtype=jnp.float32) / 6.0
    visited_root = MCTSNode(
        n=jnp.array(5, dtype=jnp.int32),  # Already visited!
        p=old_policy,
        q=jnp.array(0.5, dtype=jnp.float32),
        value_probs=old_value_probs,
        terminated=jnp.array(False, dtype=jnp.bool_),
        embedding=initial_state
    )

    # Create new policy/value that we want to update to
    new_policy = jax.nn.softmax(jnp.arange(branching_factor, dtype=jnp.float32))  # Non-uniform
    new_value = 0.9
    new_value_probs = jnp.array([0.7, 0.15, 0.05, 0.05, 0.03, 0.02], dtype=jnp.float32)
    new_embedding = initial_state  # Same for simplicity

    # Call update_root_node
    updated_root = MCTS.update_root_node(visited_root, new_policy, new_value, new_value_probs, new_embedding)

    # The issue: when n > 0 (visited), update_root_node preserves the OLD values
    # This is by design for persist_tree=True, but breaks persist_tree=False semantics

    print(f"Original visits: {visited_root.n}, Updated visits: {updated_root.n}")
    print(f"Policy preserved: {jnp.allclose(updated_root.p, old_policy)}")
    print(f"Value preserved: {updated_root.q == visited_root.q}")

    # These assertions verify the CURRENT behavior (which the engineer says is wrong
    # for persist_tree=False)
    assert jnp.allclose(updated_root.p, old_policy), (
        "update_root_node should preserve old policy when visited"
    )
    assert updated_root.q == visited_root.q, (
        "update_root_node should preserve old value when visited"
    )
    assert updated_root.n == visited_root.n, (
        "update_root_node should preserve old visit count when visited"
    )

    print("\nThis behavior is CORRECT for persist_tree=True")
    print("but INCORRECT for persist_tree=False (stale data not refreshed)")
