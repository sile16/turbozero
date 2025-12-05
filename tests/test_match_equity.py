import jax.numpy as jnp

from core.evaluators.mcts.equity import (
    BACKGAMMON_OUTCOME_POINTS,
    normalize_value_probs,
    probs_to_equity,
    terminal_value_probs_from_reward,
)


def test_probs_to_equity_boundaries():
    assert jnp.isclose(
        probs_to_equity(jnp.array([1, 0, 0, 0, 0, 0]), None, cube_value=2.0, outcome_points=BACKGAMMON_OUTCOME_POINTS),
        1.0,
    )
    assert jnp.isclose(
        probs_to_equity(jnp.array([0, 0, 0, 1, 0, 0]), None, cube_value=1.0, outcome_points=BACKGAMMON_OUTCOME_POINTS),
        0.0,
    )


def test_probs_to_equity_mixed_distribution():
    mixed_probs = jnp.array([0.4, 0.2, 0.0, 0.4, 0.0, 0.0])
    equity = probs_to_equity(mixed_probs, None, cube_value=1.0, outcome_points=BACKGAMMON_OUTCOME_POINTS)
    assert 0.0 < equity < 1.0
    assert jnp.isclose(equity, jnp.array(2.0 / 3.0))


def test_normalize_value_probs_softmaxes_logits():
    raw = jnp.array([2.0, 0.0, 0.0, -1.0, -1.0, -1.0])
    normalized = normalize_value_probs(raw)
    assert jnp.isclose(jnp.sum(normalized), 1.0)
    assert normalized.shape == (6,)


def test_terminal_value_probs_from_reward_mapping():
    win = terminal_value_probs_from_reward(jnp.array(2.0))
    loss = terminal_value_probs_from_reward(jnp.array(-3.0))
    assert jnp.argmax(win) == 1  # Gammon win bucket
    assert jnp.argmax(loss) == 5  # Backgammon loss bucket
