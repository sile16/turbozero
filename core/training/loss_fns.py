
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from core.memory.replay_memory import BaseExperience
from core.evaluators.mcts.equity import reward_to_value_targets


def entropy(probs: jnp.ndarray) -> jnp.ndarray:
    """Calculate the entropy of a probability distribution.
    
    Args:
        probs: Probabilities, should sum to 1 along the specified axis.
        
    Returns:
        Entropy values.
    """
    # Add small epsilon to avoid log(0)
    safe_probs = jnp.clip(probs, 1e-12, 1.0)
    return -jnp.sum(safe_probs * jnp.log(safe_probs), axis=-1)


def az_default_loss_fn(params: chex.ArrayTree, train_state: TrainState, experience: BaseExperience,
                       l2_reg_lambda: float = 0.0001) -> Tuple[chex.Array, Tuple[chex.ArrayTree, optax.OptState]]:
    """ Implements the default AlphaZero loss function.

    = Policy Loss + Value Loss + L2 Regularization
    Policy Loss: Cross-entropy loss between predicted policy and target policy
    Value Loss: L2 loss between predicted value and target value

    Args:
    - `params`: the parameters of the neural network
    - `train_state`: flax TrainState (holds optimizer and other state)
    - `experience`: experience sampled from replay buffer
        - stores the observation, target policy, target value
    - `l2_reg_lambda`: L2 regularization weight (default = 1e-4)

    Returns:
    - (loss, (aux_metrics, updates))
        - `loss`: total loss
        - `aux_metrics`: auxiliary metrics (policy_loss, value_loss)
        - `updates`: optimizer updates
    """

    # get batch_stats if using batch_norm
    variables = {'params': params, 'batch_stats': train_state.batch_stats} \
        if hasattr(train_state, 'batch_stats') else {'params': params}
    mutables = ['batch_stats'] if hasattr(train_state, 'batch_stats') else []

    # get predictions
    (pred_policy_logits, pred_value_logits), updates = train_state.apply_fn(
        variables,
        x=experience.observation_nn,
        train=True,
        mutable=mutables
    )

    # set invalid actions in policy to -inf
    pred_policy_logits = jnp.where(
        experience.policy_mask,
        pred_policy_logits,
        jnp.finfo(jnp.float32).min
    )

    # compute policy loss
    policy_loss = optax.softmax_cross_entropy(pred_policy_logits, experience.policy_weights).mean()

    # select appropriate value from experience.reward
    current_player = experience.cur_player_id
    batch_indices = jnp.arange(experience.reward.shape[0])
    target_reward = experience.reward[batch_indices, current_player]

    # Value loss:
    # - scalar head (shape (batch,) or (batch,1)): MSE on target_reward
    # - 4-way conditional head (shape (batch,4)): masked BCE via reward_to_value_targets
    if pred_value_logits.ndim == 1 or pred_value_logits.shape[-1] == 1:
        pred_scalar = pred_value_logits if pred_value_logits.ndim == 1 else jnp.squeeze(pred_value_logits, axis=-1)
        value_loss = jnp.mean((pred_scalar - target_reward) ** 2)
        pred_value_probs = None
        targets = None
        masks = None
    elif pred_value_logits.shape[-1] == 4:
        targets, masks = jax.vmap(reward_to_value_targets)(target_reward)
        value_loss = four_way_value_loss(pred_value_logits, targets, masks)
        pred_value_probs = jax.nn.sigmoid(pred_value_logits)
    else:
        raise ValueError(f"Unsupported value head size: {pred_value_logits.shape}")

    # compute L2 regularization
    l2_reg = l2_reg_lambda * jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree.map(
            lambda x: (x ** 2).sum(),
            params
        )
    )

    # total loss
    loss = policy_loss + value_loss + l2_reg

    # Compute predicted and target policy distributions for metrics
    pred_policy_probs = jax.nn.softmax(pred_policy_logits, axis=-1)

    # Policy accuracy: top-1 match between predicted and target
    pred_top1 = jnp.argmax(pred_policy_probs, axis=-1)
    target_top1 = jnp.argmax(experience.policy_weights, axis=-1)
    policy_accuracy = jnp.mean(pred_top1 == target_top1)

    # Policy KL divergence: KL(target || predicted)
    # Add small epsilon for numerical stability
    eps = 1e-8
    target_probs_safe = jnp.clip(experience.policy_weights, eps, 1.0)
    pred_probs_safe = jnp.clip(pred_policy_probs, eps, 1.0)
    # Only compute KL for valid actions (where target > 0)
    kl_per_sample = jnp.sum(
        jnp.where(experience.policy_weights > eps,
                  experience.policy_weights * (jnp.log(target_probs_safe) - jnp.log(pred_probs_safe)),
                  0.0),
        axis=-1
    )
    policy_kl = jnp.mean(kl_per_sample)

    aux_metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'l2_reg': l2_reg,
        'policy_entropy': entropy(pred_policy_probs).mean(),
        'policy_accuracy': policy_accuracy,
        'policy_kl': policy_kl,
    }

    if pred_value_probs is None:
        aux_metrics.update({
            'value_mse': value_loss,
            'value_mae': jnp.mean(jnp.abs(pred_scalar - target_reward)),
        })
    else:
        aux_metrics.update({
            'pred_win': pred_value_probs[:, 0].mean(),
            'pred_gam_win_cond': pred_value_probs[:, 1].mean(),
            'pred_gam_loss_cond': pred_value_probs[:, 2].mean(),
            'pred_bg_rate': pred_value_probs[:, 3].mean(),
            'target_win': targets[:, 0].mean(),
            'target_gammon_rate': masks[:, 3].mean(),
        })
    return loss, (aux_metrics, updates)


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
    bce = jnp.maximum(pred_logits, 0) - pred_logits * targets + jnp.log1p(jnp.exp(-jnp.abs(pred_logits)))

    # Apply mask
    masked_bce = bce * masks

    # Average over non-zero mask entries
    total_loss = jnp.sum(masked_bce)
    total_mask = jnp.sum(masks)

    return total_loss / jnp.maximum(total_mask, 1.0)


def az_loss_fn_4way(params: chex.ArrayTree, train_state: TrainState, experience: BaseExperience,
                    l2_reg_lambda: float = 0.0001) -> Tuple[chex.Array, Tuple[chex.ArrayTree, optax.OptState]]:
    """AlphaZero loss with 4-way conditional value head.

    = Policy Loss + Value Loss + L2 Regularization
    Policy Loss: Cross-entropy loss between predicted policy and target policy
    Value Loss: Masked binary cross-entropy for 4-way conditional outputs

    Args:
    - `params`: the parameters of the neural network
    - `train_state`: flax TrainState (holds optimizer and other state)
    - `experience`: experience sampled from replay buffer
        - stores the observation, target policy, target value
    - `l2_reg_lambda`: L2 regularization weight (default = 1e-4)

    Returns:
    - (loss, (aux_metrics, updates))
        - `loss`: total loss
        - `aux_metrics`: auxiliary metrics (policy_loss, value_loss, per-output metrics)
        - `updates`: optimizer updates
    """

    # get batch_stats if using batch_norm
    variables = {'params': params, 'batch_stats': train_state.batch_stats} \
        if hasattr(train_state, 'batch_stats') else {'params': params}
    mutables = ['batch_stats'] if hasattr(train_state, 'batch_stats') else []

    # get predictions
    (pred_policy_logits, pred_value_logits), updates = train_state.apply_fn(
        variables,
        x=experience.observation_nn,
        train=True,
        mutable=mutables
    )

    # set invalid actions in policy to -inf
    pred_policy_logits = jnp.where(
        experience.policy_mask,
        pred_policy_logits,
        jnp.finfo(jnp.float32).min
    )

    # compute policy loss
    policy_loss = optax.softmax_cross_entropy(pred_policy_logits, experience.policy_weights).mean()

    # Value loss: 4-way with masking
    current_player = experience.cur_player_id
    batch_indices = jnp.arange(experience.reward.shape[0])
    target_reward = experience.reward[batch_indices, current_player]

    targets, masks = jax.vmap(reward_to_value_targets)(target_reward)
    value_loss = four_way_value_loss(pred_value_logits, targets, masks)

    # compute L2 regularization
    l2_reg = l2_reg_lambda * jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree.map(
            lambda x: (x ** 2).sum(),
            params
        )
    )

    # total loss
    loss = policy_loss + value_loss + l2_reg

    # Compute predicted and target policy distributions for metrics
    pred_policy_probs = jax.nn.softmax(pred_policy_logits, axis=-1)

    # Policy accuracy: top-1 match between predicted and target
    pred_top1 = jnp.argmax(pred_policy_probs, axis=-1)
    target_top1 = jnp.argmax(experience.policy_weights, axis=-1)
    policy_accuracy = jnp.mean(pred_top1 == target_top1)

    # Policy KL divergence: KL(target || predicted)
    eps = 1e-8
    target_probs_safe = jnp.clip(experience.policy_weights, eps, 1.0)
    pred_probs_safe = jnp.clip(pred_policy_probs, eps, 1.0)
    kl_per_sample = jnp.sum(
        jnp.where(experience.policy_weights > eps,
                  experience.policy_weights * (jnp.log(target_probs_safe) - jnp.log(pred_probs_safe)),
                  0.0),
        axis=-1
    )
    policy_kl = jnp.mean(kl_per_sample)

    # Value metrics for 4-way head
    pred_value_probs = jax.nn.sigmoid(pred_value_logits)

    aux_metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'l2_reg': l2_reg,
        'policy_entropy': entropy(pred_policy_probs).mean(),
        'policy_accuracy': policy_accuracy,
        'policy_kl': policy_kl,
        # Per-output metrics for 4-way value head
        'pred_win': pred_value_probs[:, 0].mean(),
        'pred_gam_win_cond': pred_value_probs[:, 1].mean(),
        'pred_gam_loss_cond': pred_value_probs[:, 2].mean(),
        'pred_bg_rate': pred_value_probs[:, 3].mean(),
        'target_win': targets[:, 0].mean(),
    }
    return loss, (aux_metrics, updates)
