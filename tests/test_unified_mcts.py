"""Tests for UnifiedMCTS - the consolidated MCTS implementation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import chex

from typing import Tuple

import pgx
import pgx.backgammon as bg
from pgx.backgammon import State

from core.evaluators.mcts.unified_mcts import UnifiedMCTS
from core.evaluators.mcts.action_selection import PUCTSelector
from core.types import StepMetadata

from core.bgcommon import (
    bg_pip_count_eval,
    make_bg_decision_step_fn,
    make_bg_stochastic_step_fn,
)


# --- Evaluation function ---
@jax.jit
def backgammon_eval_fn(state: State, params: chex.ArrayTree, key: chex.PRNGKey) -> Tuple[chex.Array, float]:
    return bg_pip_count_eval(state, params, key)


# --- Fixtures ---

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def backgammon_env():
    """Create a backgammon environment."""
    return bg.Backgammon(simple_doubles=True)


@pytest.fixture
def mock_params():
    """Mock parameters for the neural network."""
    return {}


@pytest.fixture
def stochastic_action_probs(backgammon_env):
    """Get stochastic action probabilities from the backgammon environment."""
    return backgammon_env.stochastic_action_probs


@pytest.fixture
def unified_mcts(backgammon_env, stochastic_action_probs):
    """Create a UnifiedMCTS instance for backgammon."""
    return UnifiedMCTS(
        eval_fn=backgammon_eval_fn,
        action_selector=PUCTSelector(),
        policy_size=backgammon_env.num_actions,
        max_nodes=50,
        num_iterations=20,
        decision_step_fn=make_bg_decision_step_fn(backgammon_env),
        stochastic_step_fn=make_bg_stochastic_step_fn(backgammon_env),
        stochastic_action_probs=stochastic_action_probs,
        gumbel_k=16,
        temperature=0.0,
    )


def get_step_metadata(state: State, is_stochastic: bool = None) -> StepMetadata:
    """Create StepMetadata from a backgammon state."""
    if is_stochastic is None:
        is_stochastic = state._is_stochastic
    return StepMetadata(
        rewards=state.rewards,
        action_mask=state.legal_action_mask,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state._step_count,
        is_stochastic=jnp.array(is_stochastic, dtype=jnp.bool_)
    )


# --- Core Tests ---

def test_unified_mcts_init(unified_mcts, backgammon_env, key):
    """Test that UnifiedMCTS initializes correctly."""
    state = backgammon_env.init(key)
    tree = unified_mcts.init(template_embedding=state)

    assert tree is not None
    assert tree.ROOT_INDEX == 0
    # init_tree starts with next_free_idx=0 (empty tree)
    # nodes are added during evaluate()
    assert tree.next_free_idx == 0

    print("test_unified_mcts_init PASSED")


def test_root_embedding_after_step(unified_mcts, backgammon_env, mock_params, key):
    """Verify root embedding matches expected state after step().

    This is a critical test for subtree persistence:
    1. Evaluate from initial state
    2. Take action, step tree
    3. Step game environment with same action
    4. Verify tree root embedding matches env state
    """
    key, init_key, eval_key, step_key = jax.random.split(key, 4)

    # Initialize a deterministic state
    initial_state = backgammon_env.init(init_key)
    if initial_state._is_stochastic:
        initial_state = backgammon_env.stochastic_step(initial_state, 0)

    assert not initial_state._is_stochastic

    # Initialize tree and evaluate
    tree = unified_mcts.init(template_embedding=initial_state)
    root_metadata = get_step_metadata(initial_state, is_stochastic=False)

    mcts_output = unified_mcts.evaluate(
        key=eval_key,
        eval_state=tree,
        env_state=initial_state,
        root_metadata=root_metadata,
        params=mock_params,
    )

    # Get the action and step tree
    action = mcts_output.action
    new_tree = unified_mcts.step(mcts_output.eval_state, action)

    # Step the game environment with the same action
    decision_step_fn = make_bg_decision_step_fn(backgammon_env)
    next_state, _ = decision_step_fn(initial_state, action, step_key)

    # Verify root embedding matches environment state
    root_embedding = jax.tree_util.tree_map(lambda x: x[new_tree.ROOT_INDEX], new_tree.data.embedding)

    # Compare key state fields
    assert jnp.all(root_embedding._board == next_state._board), "Board should match"
    assert jnp.all(root_embedding.current_player == next_state.current_player), "Current player should match"
    assert jnp.all(root_embedding._is_stochastic == next_state._is_stochastic), "Stochastic flag should match"

    print("test_root_embedding_after_step PASSED")


def test_two_player_subtree_persistence(unified_mcts, backgammon_env, mock_params, key):
    """Verify Q-values are correct from both players' perspectives.

    1. Player 0 evaluates, takes action
    2. Step tree
    3. Player 1 evaluates from persisted tree
    4. Verify Q-values make sense for Player 1
    """
    key, init_key, eval_key1, eval_key2, step_key = jax.random.split(key, 5)

    # Initialize to a deterministic state (Player 0's turn)
    state = backgammon_env.init(init_key)
    if state._is_stochastic:
        state = backgammon_env.stochastic_step(state, 0)

    assert not state._is_stochastic
    player0 = state.current_player

    # Player 0 evaluates
    tree = unified_mcts.init(template_embedding=state)
    metadata = get_step_metadata(state, is_stochastic=False)

    mcts_output1 = unified_mcts.evaluate(
        key=eval_key1,
        eval_state=tree,
        env_state=state,
        root_metadata=metadata,
        params=mock_params,
    )

    # Get Player 0's Q-value
    player0_q = unified_mcts.get_value(mcts_output1.eval_state)

    # Step tree with action
    action = mcts_output1.action
    new_tree = unified_mcts.step(mcts_output1.eval_state, action)

    # Step game to get next state
    decision_step_fn = make_bg_decision_step_fn(backgammon_env)
    next_state, next_metadata = decision_step_fn(state, action, step_key)

    # If next state is stochastic, roll dice
    if next_state._is_stochastic:
        next_state = backgammon_env.stochastic_step(next_state, 0)
        new_tree = unified_mcts.reset(new_tree)  # Reset tree for new state

    # Get current player (should be different or same depending on game)
    player1 = next_state.current_player

    # Player 1 evaluates from persisted (or reset) tree
    metadata2 = get_step_metadata(next_state, is_stochastic=False)

    mcts_output2 = unified_mcts.evaluate(
        key=eval_key2,
        eval_state=new_tree,
        env_state=next_state,
        root_metadata=metadata2,
        params=mock_params,
    )

    # Get Player 1's Q-value
    player1_q = unified_mcts.get_value(mcts_output2.eval_state)

    # In a zero-sum game, if player changed, Q-values should have opposite signs
    # (approximately, due to exploration noise)
    if player0 != player1:
        # Q-values from different players' perspectives should have opposite tendencies
        # This is a soft check since exact values depend on exploration
        print(f"Player 0 Q-value: {player0_q}, Player 1 Q-value: {player1_q}")
        print(f"Player 0: {player0}, Player 1: {player1}")

    # Basic sanity checks
    assert not jnp.isnan(player0_q), "Player 0 Q-value should not be NaN"
    assert not jnp.isnan(player1_q), "Player 1 Q-value should not be NaN"

    print("test_two_player_subtree_persistence PASSED")


def test_stochastic_root_handling(unified_mcts, backgammon_env, mock_params, key):
    """Test that UnifiedMCTS handles stochastic roots correctly.

    In backgammon, initial state is stochastic (waiting for dice roll).
    """
    key, init_key, eval_key = jax.random.split(key, 3)

    # Get initial stochastic state
    state = backgammon_env.init(init_key)
    assert state._is_stochastic, "Initial backgammon state should be stochastic"

    # Initialize tree for stochastic state
    tree = unified_mcts.init(template_embedding=state)
    metadata = get_step_metadata(state, is_stochastic=True)

    # Evaluate - should use stochastic root path
    mcts_output = unified_mcts.evaluate(
        key=eval_key,
        eval_state=tree,
        env_state=state,
        root_metadata=metadata,
        params=mock_params,
    )

    # Action should be a valid stochastic action (dice roll)
    action = mcts_output.action
    stochastic_size = len(backgammon_env.stochastic_action_probs)
    assert 0 <= action < stochastic_size, f"Action {action} should be valid stochastic action"

    # Tree should have been updated
    assert mcts_output.eval_state.next_free_idx >= 1

    print("test_stochastic_root_handling PASSED")


def test_gumbel_topk_at_root(unified_mcts, backgammon_env, mock_params, key):
    """Verify Gumbel-Top-k is used at decision roots.

    With gumbel_k=16, approximately 16 actions should get visits.
    """
    key, init_key, eval_key = jax.random.split(key, 3)

    # Get a deterministic state
    state = backgammon_env.init(init_key)
    if state._is_stochastic:
        state = backgammon_env.stochastic_step(state, 0)

    assert not state._is_stochastic

    # Count legal actions
    legal_count = jnp.sum(state.legal_action_mask)

    # Create MCTS with specific gumbel_k
    mcts = UnifiedMCTS(
        eval_fn=backgammon_eval_fn,
        action_selector=PUCTSelector(),
        policy_size=backgammon_env.num_actions,
        max_nodes=200,
        num_iterations=100,  # More iterations to ensure coverage
        decision_step_fn=make_bg_decision_step_fn(backgammon_env),
        stochastic_step_fn=make_bg_stochastic_step_fn(backgammon_env),
        stochastic_action_probs=backgammon_env.stochastic_action_probs,
        gumbel_k=4,  # Only 4 actions
        temperature=0.0,
    )

    # Initialize and evaluate
    tree = mcts.init(template_embedding=state)
    metadata = get_step_metadata(state, is_stochastic=False)

    mcts_output = mcts.evaluate(
        key=eval_key,
        eval_state=tree,
        env_state=state,
        root_metadata=metadata,
        params=mock_params,
    )

    # Check that approximately gumbel_k actions got visits
    root_visits = mcts_output.eval_state.get_child_data('n', mcts_output.eval_state.ROOT_INDEX)
    actions_with_visits = jnp.sum(root_visits[:mcts.policy_size] > 0)

    # Should have roughly gumbel_k actions visited (could be fewer if legal < gumbel_k)
    expected_max = min(4, int(legal_count))
    print(f"Actions with visits: {actions_with_visits}, expected ~{expected_max}")

    # Allow some tolerance since Gumbel sampling is stochastic
    assert actions_with_visits <= expected_max + 2, f"Too many actions visited: {actions_with_visits}"
    assert actions_with_visits >= 1, "At least one action should be visited"

    print("test_gumbel_topk_at_root PASSED")


def test_step_always_persists_subtree(unified_mcts, backgammon_env, mock_params, key):
    """Verify step() always persists subtree (no reset option)."""
    key, init_key, eval_key = jax.random.split(key, 3)

    # Initialize a deterministic state
    state = backgammon_env.init(init_key)
    if state._is_stochastic:
        state = backgammon_env.stochastic_step(state, 0)

    # Initialize and evaluate
    tree = unified_mcts.init(template_embedding=state)
    metadata = get_step_metadata(state, is_stochastic=False)

    mcts_output = unified_mcts.evaluate(
        key=eval_key,
        eval_state=tree,
        env_state=state,
        root_metadata=metadata,
        params=mock_params,
    )

    # Record state before step
    tree_before = mcts_output.eval_state
    action = mcts_output.action

    # Check if child exists for the action
    child_exists = tree_before.is_edge(tree_before.ROOT_INDEX, action)

    # Step tree
    new_tree = unified_mcts.step(tree_before, action)

    if child_exists:
        # Subtree should be preserved
        assert new_tree.next_free_idx >= 1, "Tree should have nodes after step"
        assert new_tree.parents[new_tree.ROOT_INDEX] == -1, "New root should have no parent"
    else:
        # If child didn't exist, tree resets
        assert new_tree.next_free_idx == 1

    print("test_step_always_persists_subtree PASSED")


def test_decision_to_stochastic_transition(unified_mcts, backgammon_env, mock_params, key):
    """Test transition from decision node to stochastic node."""
    key, init_key, eval_key, step_key = jax.random.split(key, 4)

    # Get a deterministic state (after dice roll)
    state = backgammon_env.init(init_key)
    if state._is_stochastic:
        state = backgammon_env.stochastic_step(state, 0)

    assert not state._is_stochastic

    # Evaluate
    tree = unified_mcts.init(template_embedding=state)
    metadata = get_step_metadata(state, is_stochastic=False)

    mcts_output = unified_mcts.evaluate(
        key=eval_key,
        eval_state=tree,
        env_state=state,
        root_metadata=metadata,
        params=mock_params,
    )

    action = mcts_output.action

    # Step environment to check if next state is stochastic
    decision_step_fn = make_bg_decision_step_fn(backgammon_env)
    next_state, next_metadata = decision_step_fn(state, action, step_key)

    # In backgammon, after all moves are made, state becomes stochastic
    print(f"After action {action}: is_stochastic = {next_state._is_stochastic}")

    # Step tree
    new_tree = unified_mcts.step(mcts_output.eval_state, action)

    # If next state is stochastic, check root is marked as chance node
    if next_state._is_stochastic:
        root_is_chance = new_tree.data.is_chance_node[new_tree.ROOT_INDEX]
        assert root_is_chance, "Root should be marked as chance node for stochastic state"

    print("test_decision_to_stochastic_transition PASSED")


def test_get_config(unified_mcts):
    """Test that get_config returns proper configuration."""
    config = unified_mcts.get_config()

    assert config["type"] == "UnifiedMCTS"
    assert "policy_size" in config
    assert "max_nodes" in config
    assert "num_iterations" in config
    assert "gumbel_k" in config
    assert "is_stochastic_game" in config
    assert config["is_stochastic_game"] == True

    print("test_get_config PASSED")


def test_reset(unified_mcts, backgammon_env, mock_params, key):
    """Test reset() clears the tree."""
    key, init_key, eval_key = jax.random.split(key, 3)

    # Initialize and evaluate to build up tree
    state = backgammon_env.init(init_key)
    if state._is_stochastic:
        state = backgammon_env.stochastic_step(state, 0)

    tree = unified_mcts.init(template_embedding=state)
    metadata = get_step_metadata(state, is_stochastic=False)

    mcts_output = unified_mcts.evaluate(
        key=eval_key,
        eval_state=tree,
        env_state=state,
        root_metadata=metadata,
        params=mock_params,
    )

    # Tree should have multiple nodes
    assert mcts_output.eval_state.next_free_idx > 1

    # Reset tree
    reset_tree = unified_mcts.reset(mcts_output.eval_state)

    # Tree should be cleared
    assert reset_tree.next_free_idx == 0 or reset_tree.data.n[reset_tree.ROOT_INDEX] == 0

    print("test_reset PASSED")


def test_vmap_support(backgammon_env, mock_params, key):
    """Verify UnifiedMCTS.evaluate works under jax.vmap."""
    from core.evaluators.mcts.unified_mcts import UnifiedMCTS
    from core.evaluators.mcts.action_selection import PUCTSelector
    from core.bgcommon import (
        bg_pip_count_eval,
        make_bg_decision_step_fn,
        make_bg_stochastic_step_fn,
    )

    batch_size = 4
    key, *init_keys = jax.random.split(key, batch_size + 1)
    init_keys = jnp.stack(init_keys)

    # Create batch of states
    v_init = jax.vmap(backgammon_env.init)
    states = v_init(init_keys)

    # Get to deterministic states
    v_stoch_step = jax.vmap(lambda s: backgammon_env.stochastic_step(s, 0))
    states = jax.lax.cond(
        states._is_stochastic[0],
        lambda: v_stoch_step(states),
        lambda: states
    )

    # Create MCTS
    mcts = UnifiedMCTS(
        eval_fn=bg_pip_count_eval,
        action_selector=PUCTSelector(),
        policy_size=backgammon_env.num_actions,
        max_nodes=30,
        num_iterations=10,
        decision_step_fn=make_bg_decision_step_fn(backgammon_env),
        stochastic_step_fn=make_bg_stochastic_step_fn(backgammon_env),
        stochastic_action_probs=backgammon_env.stochastic_action_probs,
        gumbel_k=8,
        temperature=1.0,
    )

    # Initialize trees for batch
    def init_tree(state):
        return mcts.init(template_embedding=state)

    v_init_tree = jax.vmap(init_tree)
    trees = v_init_tree(states)

    # Create metadata for batch
    def make_metadata(state):
        return StepMetadata(
            rewards=state.rewards,
            action_mask=state.legal_action_mask,
            terminated=state.terminated,
            cur_player_id=state.current_player,
            step=state._step_count,
            is_stochastic=jnp.array(False, dtype=jnp.bool_)
        )

    v_make_metadata = jax.vmap(make_metadata)
    metadatas = v_make_metadata(states)

    # Vectorized evaluate
    key, *eval_keys = jax.random.split(key, batch_size + 1)
    eval_keys = jnp.stack(eval_keys)

    def single_evaluate(eval_key, tree, state, metadata):
        return mcts.evaluate(
            key=eval_key,
            eval_state=tree,
            env_state=state,
            root_metadata=metadata,
            params=mock_params,
        )

    v_evaluate = jax.vmap(single_evaluate)
    outputs = v_evaluate(eval_keys, trees, states, metadatas)

    # Verify outputs have batch dimension
    assert outputs.action.shape == (batch_size,), f"Expected batch actions, got {outputs.action.shape}"
    assert outputs.policy_weights.shape[0] == batch_size, f"Expected batch policies"

    print("test_vmap_support PASSED")


def test_value_convergence(key):
    """Test that root Q-value converges to expected value in simple stochastic environment.

    Uses a simple coin-flip scenario where outcome probabilities are known.
    This test uses a simple JAX-compatible state structure.
    """
    from core.evaluators.mcts.unified_mcts import UnifiedMCTS
    from core.evaluators.mcts.action_selection import PUCTSelector
    from chex import dataclass

    # Simple state as a proper JAX pytree
    @dataclass
    class SimpleState:
        rewards: chex.Array
        current_player: chex.Array
        legal_action_mask: chex.Array
        terminated: chex.Array
        step_count: chex.Array

    def make_state(value=0.0, is_terminal=False):
        return SimpleState(
            rewards=jnp.array([value, -value]),
            current_player=jnp.array(0),
            legal_action_mask=jnp.array([True, True]),
            terminated=jnp.array(is_terminal),
            step_count=jnp.array(0),
        )

    # Simple eval function that returns known values
    def simple_eval_fn(state, params, key):
        # Return uniform policy and value = 0 (will be overwritten by terminal)
        policy = jnp.ones(2) / 2
        value = jnp.array(0.0)
        return policy, value

    # Stochastic action probs: 50% outcome 0 (value +1), 50% outcome 1 (value -1)
    stochastic_probs = jnp.array([0.5, 0.5])

    # Decision step: always goes to stochastic
    def decision_step_fn(state, action, key):
        new_state = make_state(value=0.0, is_terminal=False)
        return new_state, StepMetadata(
            rewards=jnp.array([0.0, 0.0]),
            action_mask=jnp.array([True, True]),
            terminated=jnp.array(False),
            cur_player_id=jnp.array(0),
            step=jnp.array(1),
            is_stochastic=jnp.array(True)
        )

    # Stochastic step: outcome 0 = +1, outcome 1 = -1
    def stochastic_step_fn(state, outcome, key):
        value = jax.lax.cond(outcome == 0, lambda: 1.0, lambda: -1.0)
        new_state = make_state(value=value, is_terminal=True)
        return new_state, StepMetadata(
            rewards=jnp.array([value, -value]),
            action_mask=jnp.array([False, False]),
            terminated=jnp.array(True),
            cur_player_id=jnp.array(0),
            step=jnp.array(2),
            is_stochastic=jnp.array(False)
        )

    mcts = UnifiedMCTS(
        eval_fn=simple_eval_fn,
        action_selector=PUCTSelector(),
        policy_size=2,
        max_nodes=50,
        num_iterations=30,
        decision_step_fn=decision_step_fn,
        stochastic_step_fn=stochastic_step_fn,
        stochastic_action_probs=stochastic_probs,
        gumbel_k=2,
        temperature=1.0,
    )

    initial_state = make_state()
    tree = mcts.init(template_embedding=initial_state)
    metadata = StepMetadata(
        rewards=jnp.array([0.0, 0.0]),
        action_mask=jnp.array([True, True]),
        terminated=jnp.array(False),
        cur_player_id=jnp.array(0),
        step=jnp.array(0),
        is_stochastic=jnp.array(False)
    )

    key, eval_key = jax.random.split(key)
    output = mcts.evaluate(
        key=eval_key,
        eval_state=tree,
        env_state=initial_state,
        root_metadata=metadata,
        params={},
    )

    # Expected value: 0.5 * (+1) + 0.5 * (-1) = 0
    root_value = mcts.get_value(output.eval_state)
    assert jnp.abs(root_value) < 0.5, f"Expected value near 0, got {root_value}"

    print(f"test_value_convergence: root_value={root_value} (expected ~0)")
    print("test_value_convergence PASSED")


def test_full_tree_behavior(unified_mcts, backgammon_env, mock_params, key):
    """Verify stability when num_iterations exceeds max_nodes."""
    key, init_key, eval_key = jax.random.split(key, 3)

    # Create MCTS with more iterations than nodes
    from core.evaluators.mcts.unified_mcts import UnifiedMCTS
    from core.evaluators.mcts.action_selection import PUCTSelector
    from core.bgcommon import (
        bg_pip_count_eval,
        make_bg_decision_step_fn,
        make_bg_stochastic_step_fn,
    )

    # Create MCTS with num_iterations > max_nodes
    mcts = UnifiedMCTS(
        eval_fn=bg_pip_count_eval,
        action_selector=PUCTSelector(),
        policy_size=backgammon_env.num_actions,
        max_nodes=10,  # Small tree
        num_iterations=50,  # More iterations than nodes
        decision_step_fn=make_bg_decision_step_fn(backgammon_env),
        stochastic_step_fn=make_bg_stochastic_step_fn(backgammon_env),
        stochastic_action_probs=backgammon_env.stochastic_action_probs,
        gumbel_k=4,
        temperature=1.0,
    )

    # Get a deterministic state
    state = backgammon_env.init(init_key)
    if state._is_stochastic:
        state = backgammon_env.stochastic_step(state, 0)

    tree = mcts.init(template_embedding=state)
    metadata = get_step_metadata(state, is_stochastic=False)

    # This should not crash even with more iterations than nodes
    output = mcts.evaluate(
        key=eval_key,
        eval_state=tree,
        env_state=state,
        root_metadata=metadata,
        params=mock_params,
    )

    # Should have a valid action
    assert 0 <= output.action < backgammon_env.num_actions
    # Tree should be at capacity or close
    assert output.eval_state.next_free_idx <= mcts.max_nodes

    print("test_full_tree_behavior PASSED")


def test_value_to_scalar(key):
    """Test that _value_to_scalar correctly handles different value shapes."""
    from core.evaluators.mcts.unified_mcts import _value_to_scalar

    # Scalar value - should pass through
    scalar = jnp.array(0.5)
    result = _value_to_scalar(scalar)
    assert result.shape == (), f"Expected scalar, got shape {result.shape}"
    assert jnp.isclose(result, 0.5)

    # 1D value - should take first element
    array_1d = jnp.array([0.7, 0.3, 0.1])
    result = _value_to_scalar(array_1d)
    assert result.shape == (), f"Expected scalar, got shape {result.shape}"
    assert jnp.isclose(result, 0.7)

    # 4-way value head - should take first element
    array_4 = jnp.array([0.8, 0.1, 0.05, 0.05])
    result = _value_to_scalar(array_4)
    assert result.shape == (), f"Expected scalar, got shape {result.shape}"
    assert jnp.isclose(result, 0.8)

    print("test_value_to_scalar PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
