import chex
import jax
import jax.numpy as jnp
from typing import Tuple

# =============================================================================
# 4-way conditional value head functions
# =============================================================================

def reward_to_value_targets(reward: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Convert game reward to 4-way value targets with masks.

    Args:
        reward: Scalar reward from current player's perspective.
                Values: 1 (win), 2 (gammon win), 3 (bg win),
                       -1 (loss), -2 (gammon loss), -3 (bg loss)

    Returns:
        targets: Array of shape (4,) with target values:
            [win, gam_win_cond, gam_loss_cond, bg_rate]
        masks: Array of shape (4,) indicating which outputs to train:
            [1, 1 if win else 0, 1 if loss else 0, 1 if gammon else 0]
    """
    did_win = reward > 0
    was_gammon = jnp.abs(reward) >= 2
    was_bg = jnp.abs(reward) >= 3

    # Target values
    win_target = jnp.where(did_win, 1.0, 0.0)
    gam_win_cond_target = jnp.where(was_gammon & did_win, 1.0, 0.0)
    gam_loss_cond_target = jnp.where(was_gammon & ~did_win, 1.0, 0.0)
    bg_rate_target = jnp.where(was_bg, 1.0, 0.0)

    targets = jnp.stack([win_target, gam_win_cond_target, gam_loss_cond_target, bg_rate_target], axis=-1)

    # Masks: always train win, conditionally train gammons, train bg only when gammon
    win_mask = jnp.ones_like(reward)
    gam_win_mask = jnp.where(did_win, 1.0, 0.0)
    gam_loss_mask = jnp.where(~did_win, 1.0, 0.0)
    bg_mask = jnp.where(was_gammon, 1.0, 0.0)

    masks = jnp.stack([win_mask, gam_win_mask, gam_loss_mask, bg_mask], axis=-1)

    return targets, masks


def value_outputs_to_equity(value_probs: chex.Array, match_score: tuple | None = None) -> chex.Array:
    """Convert 4-way value outputs to equity.

    Args:
        value_probs: Array of shape (..., 4) with sigmoid outputs:
            [win, gam_win_cond, gam_loss_cond, bg_rate]
        match_score: Optional (my_score, opp_score, match_length) for match play.
                    If None, computes money game equity.

    Returns:
        Equity value(s) in [-3, +3] range for money game,
        or [0, 1] for match play.
    """
    win = value_probs[..., 0]
    gam_win_cond = value_probs[..., 1]
    gam_loss_cond = value_probs[..., 2]
    bg_rate = value_probs[..., 3]

    loss = 1.0 - win

    # Unconditional gammon probabilities
    gam_win = win * gam_win_cond
    gam_loss = loss * gam_loss_cond

    # Backgammon probabilities
    bg_win = gam_win * bg_rate
    bg_loss = gam_loss * bg_rate

    if match_score is None:
        # Money game equity: sum of point expectations
        # Single game: +1/-1, Gammon: +2/-2, Backgammon: +3/-3
        single_win = win - gam_win
        single_loss = loss - gam_loss
        gammon_win_only = gam_win - bg_win
        gammon_loss_only = gam_loss - bg_loss

        equity = (
            single_win * 1.0 - single_loss * 1.0 +
            gammon_win_only * 2.0 - gammon_loss_only * 2.0 +
            bg_win * 3.0 - bg_loss * 3.0
        )
        return equity
    else:
        # Match play equity using MET lookup
        # This requires a match equity table - implement separately
        raise NotImplementedError("Match play equity not yet implemented")


def normalize_value_probs_4way(value_head_output: chex.Array) -> chex.Array:
    """Convert 4-way value head logits to probabilities using sigmoid."""
    return jax.nn.sigmoid(value_head_output)


def terminal_value_probs_from_reward_4way(reward: chex.Array) -> chex.Array:
    """Map a signed reward to a 4-way conditional probability vector.

    Args:
        reward: Scalar reward from current player's perspective.
                Values: 1 (win), 2 (gammon win), 3 (bg win),
                       -1 (loss), -2 (gammon loss), -3 (bg loss)

    Returns:
        Array of shape (4,) with values:
            [win, gam_win_cond, gam_loss_cond, bg_rate]
    """
    reward = jnp.asarray(reward)
    did_win = reward > 0
    was_gammon = jnp.abs(reward) >= 2
    was_bg = jnp.abs(reward) >= 3

    win = jnp.where(did_win, 1.0, 0.0)
    gam_win_cond = jnp.where(was_gammon & did_win, 1.0, 0.0)
    gam_loss_cond = jnp.where(was_gammon & ~did_win, 1.0, 0.0)
    bg_rate = jnp.where(was_bg, 1.0, 0.0)

    return jnp.stack([win, gam_win_cond, gam_loss_cond, bg_rate], axis=-1)


def probs_to_equity_4way(
    value_probs: chex.Array,
    match_score: chex.Array | None = None,
    cube_value: float = 1.0,
) -> chex.Array:
    """Convert 4-way outcome probabilities to a scalar match equity in [0, 1].

    This is a wrapper around value_outputs_to_equity that normalizes the output
    to the [0, 1] range.

    Args:
        value_probs: Array of shape (4,) with sigmoid outputs:
            [win, gam_win_cond, gam_loss_cond, bg_rate]
        match_score: Optional match score tuple (unused for now)
        cube_value: Cube value multiplier (default 1.0)

    Returns:
        Equity value in [0, 1] range.
    """
    del match_score  # Placeholder for downstream table-driven implementations
    equity = value_outputs_to_equity(value_probs, None)
    # Scale by cube value
    equity = equity * cube_value
    # Normalize from [-3, +3] to [0, 1]
    return equity_to_normalized(equity)


def equity_to_normalized(equity: chex.Array) -> chex.Array:
    """Convert money game equity [-3, +3] to normalized [0, 1] range."""
    return (equity + 3.0) / 6.0


def normalized_to_equity(normalized: chex.Array) -> chex.Array:
    """Convert normalized [0, 1] to money game equity [-3, +3]."""
    return normalized * 6.0 - 3.0
