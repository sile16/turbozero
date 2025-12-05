import chex
import jax
import jax.numpy as jnp

# Outcome order: [win, gammon win, backgammon win, loss, gammon loss, backgammon loss]
BACKGAMMON_OUTCOME_POINTS = jnp.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0], dtype=jnp.float32)


def normalize_value_probs(value_head_output: chex.Array) -> chex.Array:
    """Ensure value head output is a proper probability distribution."""
    # Allow callers to pass logits or already-normalized probabilities.
    logits = value_head_output - jnp.max(value_head_output)
    probs = jax.nn.softmax(logits, axis=-1)
    return probs


def terminal_value_probs_from_reward(reward: chex.Array) -> chex.Array:
    """Map a signed reward (money play-style points) to a one-hot outcome vector."""
    magnitude = jnp.clip(jnp.rint(jnp.abs(reward)), 0, 3)
    outcome_bucket = jnp.minimum(magnitude - 1, 2).astype(jnp.int32)
    base = jax.nn.one_hot(outcome_bucket, 3, dtype=jnp.float32)
    zeros = jnp.zeros_like(base)
    win_vector = jnp.concatenate([base, zeros], axis=-1)
    loss_vector = jnp.concatenate([zeros, base], axis=-1)
    return jnp.where(reward[..., None] >= 0, win_vector, loss_vector)


def probs_to_equity(
    value_probs: chex.Array,
    match_score: chex.Array | None,
    cube_value: float = 1.0,
    outcome_points: chex.Array = BACKGAMMON_OUTCOME_POINTS,
) -> chex.Array:
    """Convert 6-way outcome probabilities to a scalar match equity in [0, 1].

    A Rockwell-Kazaross table can be injected via `outcome_points`/`match_score` upstream;
    here we compute money-play style equity as an expected points total scaled to [0, 1].
    """
    del match_score  # Placeholder for downstream table-driven implementations
    win_weights = jnp.abs(outcome_points[:3]) * cube_value
    loss_weights = jnp.abs(outcome_points[3:]) * cube_value

    win_expectation = jnp.sum(value_probs[:3] * win_weights)
    loss_expectation = jnp.sum(value_probs[3:] * loss_weights)

    denom = win_expectation + loss_expectation
    # If denom is zero (shouldn't happen with normalized probs), fall back to 0.5 neutrality
    normalized = jnp.where(
        denom > 0,
        (win_expectation - loss_expectation) / denom,
        jnp.array(0.0, dtype=jnp.float32),
    )

    return jnp.clip(0.5 + 0.5 * normalized, 0.0, 1.0)
