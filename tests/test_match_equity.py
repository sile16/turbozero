import jax.numpy as jnp

from core.evaluators.mcts.equity import (
    normalize_value_probs_4way,
    probs_to_equity_4way,
    terminal_value_probs_from_reward_4way,
    value_outputs_to_equity,
)


def test_probs_to_equity_4way_boundaries():
    win_probs = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.isclose(probs_to_equity_4way(win_probs, None, cube_value=2.0), 1.0)

    loss_probs = jnp.array([0.0, 0.0, 0.0, 0.0])
    assert jnp.isclose(probs_to_equity_4way(loss_probs, None, cube_value=1.0), 0.0)


def test_value_outputs_to_equity_mixed_distribution():
    mixed_probs = jnp.array([0.5, 0.5, 0.0, 0.0])
    equity = value_outputs_to_equity(mixed_probs, None)
    assert equity > 0.0


def test_normalize_value_probs_4way_sigmoids_logits():
    raw = jnp.array([2.0, 0.0, -1.0, 1.0])
    normalized = normalize_value_probs_4way(raw)
    assert jnp.all((normalized >= 0.0) & (normalized <= 1.0))
    assert normalized.shape == (4,)


def test_terminal_value_probs_from_reward_4way_mapping():
    win = terminal_value_probs_from_reward_4way(jnp.array(2.0))
    loss = terminal_value_probs_from_reward_4way(jnp.array(-3.0))
    assert jnp.allclose(win, jnp.array([1.0, 1.0, 0.0, 0.0]))
    assert jnp.allclose(loss, jnp.array([0.0, 0.0, 1.0, 1.0]))
