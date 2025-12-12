# TurboZero Update: 4-Way Conditional Value Head

This document describes the required changes to TurboZero to support the new 4-way conditional value head format for backgammon.

## Overview

We use a 4-way independent output format with conditional gammon probabilities.

### Value Head Format (4-way, independent with conditionals)
```
[win, gam_win_cond, gam_loss_cond, bg_rate]
```
- `win` [0,1]: P(current player wins)
- `gam_win_cond` [0,1]: P(gammon | win) - conditional probability
- `gam_loss_cond` [0,1]: P(gammon | loss) - conditional probability
- `bg_rate` [0,1]: P(backgammon | gammon) - combined rate

All outputs use sigmoid activation (independent probabilities, not mutually exclusive).

## Benefits

1. **No redundancy**: `opp_win = 1 - win` is derivable
2. **Better precision for gammons**: Conditional probabilities use full [0,1] range
3. **Matches bearoff table format**: Direct compatibility with precomputed endgame tables
4. **Cleaner training signal**: Conditional outputs only train when condition is met

## Required Changes in TurboZero

### 1. Update `core/evaluators/mcts/equity.py`

#### Functions:

```python
import jax.numpy as jnp


def reward_to_value_targets(reward: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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


def value_outputs_to_equity(value_probs: jnp.ndarray, match_score: tuple | None = None) -> jnp.ndarray:
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


def equity_to_normalized(equity: jnp.ndarray) -> jnp.ndarray:
    """Convert money game equity [-3, +3] to normalized [0, 1] range."""
    return (equity + 3.0) / 6.0


def normalized_to_equity(normalized: jnp.ndarray) -> jnp.ndarray:
    """Convert normalized [0, 1] to money game equity [-3, +3]."""
    return normalized * 6.0 - 3.0
```

### 2. Update Loss Functions in `core/training/loss_fns.py`

#### Add new 4-way value loss:

```python
def four_way_value_loss(
    pred_logits: jnp.ndarray,
    targets: jnp.ndarray,
    masks: jnp.ndarray,
) -> jnp.ndarray:
    """Compute masked binary cross-entropy loss for 4-way value head.

    Args:
        pred_logits: Raw logits of shape (batch, 4), will be passed through sigmoid
        targets: Target values of shape (batch, 4), each in [0, 1]
        masks: Mask of shape (batch, 4), 1.0 to include in loss, 0.0 to exclude

    Returns:
        Scalar loss value (mean over batch and masked outputs)
    """
    # Numerically stable sigmoid cross-entropy
    # BCE = max(x, 0) - x*t + log(1 + exp(-|x|))
    pred_probs = jax.nn.sigmoid(pred_logits)

    # Per-element BCE
    bce = jnp.maximum(pred_logits, 0) - pred_logits * targets + jnp.log1p(jnp.exp(-jnp.abs(pred_logits)))

    # Apply mask
    masked_bce = bce * masks

    # Average over non-zero mask entries
    total_loss = jnp.sum(masked_bce)
    total_mask = jnp.sum(masks)

    return total_loss / jnp.maximum(total_mask, 1.0)
```

#### Update `az_default_loss_fn` or create new variant:

```python
def az_loss_fn_4way(
    params: chex.ArrayTree,
    train_state: TrainState,
    experience: BaseExperience,
    l2_reg_lambda: float = 0.0001,
) -> Tuple[chex.Array, Tuple[chex.ArrayTree, optax.OptState]]:
    """AlphaZero loss with 4-way conditional value head."""

    # Forward pass
    variables = {'params': params}
    if hasattr(train_state, 'batch_stats'):
        variables['batch_stats'] = train_state.batch_stats

    mutables = ['batch_stats'] if hasattr(train_state, 'batch_stats') else []

    (pred_policy_logits, pred_value_logits), updates = train_state.apply_fn(
        variables,
        x=experience.observation_nn,
        mutable=mutables
    )

    # Policy loss (unchanged)
    pred_policy_logits = jnp.where(
        experience.policy_mask,
        pred_policy_logits,
        jnp.finfo(jnp.float32).min
    )
    policy_loss = optax.softmax_cross_entropy(
        pred_policy_logits, experience.policy_weights
    ).mean()

    # Value loss: 4-way with masking
    current_player = experience.cur_player_id
    batch_indices = jnp.arange(experience.reward.shape[0])
    target_reward = experience.reward[batch_indices, current_player]

    targets, masks = jax.vmap(reward_to_value_targets)(target_reward)
    value_loss = four_way_value_loss(pred_value_logits, targets, masks)

    # L2 regularization
    l2_reg = l2_reg_lambda * jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree.map(lambda x: (x ** 2).sum(), params)
    )

    loss = policy_loss + value_loss + l2_reg

    # Metrics
    pred_value_probs = jax.nn.sigmoid(pred_value_logits)

    aux_metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'l2_reg': l2_reg,
        # Per-output metrics
        'pred_win': pred_value_probs[:, 0].mean(),
        'pred_gam_win_cond': pred_value_probs[:, 1].mean(),
        'pred_gam_loss_cond': pred_value_probs[:, 2].mean(),
        'pred_bg_rate': pred_value_probs[:, 3].mean(),
        'target_win': targets[:, 0].mean(),
    }

    return loss, (aux_metrics, updates)
```

### 3. Update MCTS Value Handling

In the MCTS evaluator, update how value is extracted:

```python
# 4-way sigmoid
value_probs = jax.nn.sigmoid(value_logits)
equity = value_outputs_to_equity(value_probs)
```

### 4. Update Network Architecture Support

Ensure networks specify `value_head_out_size=4` and treat outputs as conditional logits.

## Testing

After implementing changes:

1. **Unit test `reward_to_value_targets`**:
   - reward=1 → targets=[1,0,0,0], masks=[1,1,0,0]
   - reward=2 → targets=[1,1,0,0], masks=[1,1,0,1]
   - reward=3 → targets=[1,1,0,1], masks=[1,1,0,1]
   - reward=-1 → targets=[0,0,0,0], masks=[1,0,1,0]
   - reward=-2 → targets=[0,0,1,0], masks=[1,0,1,1]
   - reward=-3 → targets=[0,0,1,1], masks=[1,0,1,1]

2. **Unit test `value_outputs_to_equity`**:
   - [1,0,0,0] → equity=1.0 (single win)
   - [1,1,0,0] → equity=2.0 (gammon win, no bg)
   - [1,1,0,1] → equity=3.0 (backgammon win)
   - [0,0,0,0] → equity=-1.0 (single loss)
   - [0.5,0.2,0.3,0.1] → verify intermediate equity calculation

3. **Integration test**: Run a short training loop and verify loss decreases.

## Notes

- 4-way conditional value head is the only supported format.
