
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from core.memory.replay_memory import BaseExperience


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
                       l2_reg_lambda: float = 0.0001,
                       value_loss_weight: float = 1.0,
                       illegal_action_penalty: float = 1.0) -> Tuple[chex.Array, Tuple[chex.ArrayTree, optax.OptState]]:
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
    (pred_policy_logits_raw, pred_value_logits), updates = train_state.apply_fn(
        variables,
        x=experience.observation_nn,
        train=True,
        mutable=mutables
    )

    # Illegal action loss: penalize high logits for illegal actions
    # This trains the network to output low probabilities for illegal moves
    # We use softmax to get probabilities, then penalize probability mass on illegal actions
    illegal_mask = ~experience.policy_mask  # True where action is illegal
    pred_probs_raw = jax.nn.softmax(pred_policy_logits_raw, axis=-1)
    # Sum of probability on illegal actions (should be 0 ideally)
    illegal_prob_mass = jnp.sum(pred_probs_raw * illegal_mask, axis=-1)
    illegal_action_loss = jnp.mean(illegal_prob_mass)

    # set invalid actions in policy to -inf for cross-entropy computation
    pred_policy_logits = jnp.where(
        experience.policy_mask,
        pred_policy_logits_raw,
        jnp.finfo(jnp.float32).min
    )

    # compute policy loss (skip for chance nodes if flag is set)
    policy_loss_per_sample = optax.softmax_cross_entropy(pred_policy_logits, experience.policy_weights)

    # CRITICAL FIX: For stochastic games, some samples may be chance nodes where:
    # 1. policy_mask is all False (no valid player actions)
    # 2. policy_weights may be all zeros or not sum to 1
    # 3. log_softmax of all-masked logits produces -inf or NaN
    # This can produce very large (but finite) or non-finite loss values.
    #
    # We apply multiple safeguards:
    # 1. Replace non-finite values with 0
    # 2. Clamp to reasonable range (max cross-entropy should be ~log(num_classes) ≈ 10)
    # 3. Zero out samples with no valid actions
    policy_loss_per_sample = jnp.where(
        jnp.isfinite(policy_loss_per_sample),
        policy_loss_per_sample,
        0.0
    )
    # Clamp to reasonable range - cross-entropy should never exceed ~10 for any game
    policy_loss_per_sample = jnp.clip(policy_loss_per_sample, 0.0, 100.0)

    # Detect samples with no valid actions (all policy_mask False) and zero them out
    has_valid_actions = jnp.any(experience.policy_mask, axis=-1)
    policy_loss_per_sample = jnp.where(has_valid_actions, policy_loss_per_sample, 0.0)

    # Also detect samples where policy_weights don't sum to ~1 (invalid targets)
    policy_sum = experience.policy_weights.sum(axis=-1)
    valid_policy = jnp.logical_and(policy_sum > 0.5, policy_sum < 1.5)  # Allow some tolerance
    policy_loss_per_sample = jnp.where(valid_policy, policy_loss_per_sample, 0.0)
    has_valid_actions = jnp.logical_and(has_valid_actions, valid_policy)

    # Mask out policy loss for chance nodes (where we don't have meaningful policy targets)
    if experience.is_chance_node is not None:
        # is_chance_node is True for chance nodes, False for decision nodes
        # We want to keep loss only for decision nodes (is_chance_node = False)
        is_decision_node = ~experience.is_chance_node
        num_decision_samples = jnp.sum(is_decision_node)

        # Avoid division by zero if all samples are chance nodes
        policy_loss = jnp.where(
            num_decision_samples > 0,
            jnp.sum(policy_loss_per_sample * is_decision_node) / num_decision_samples,
            0.0
        )
    else:
        # Even without is_chance_node tracking, we now skip samples with no valid actions
        num_valid_samples = jnp.sum(has_valid_actions)
        policy_loss = jnp.where(
            num_valid_samples > 0,
            jnp.sum(policy_loss_per_sample) / num_valid_samples,
            0.0
        )

    # select appropriate value from experience.reward
    current_player = experience.cur_player_id
    batch_indices = jnp.arange(experience.reward.shape[0])
    target_reward = experience.reward[batch_indices, current_player]

    # Value loss:
    # - scalar head (shape (batch,) or (batch,1)): MSE on target_reward
    # NOTE: We train the value head on ALL samples including chance nodes.
    # The value at a chance node should be the expected value over outcomes,
    # which the NN can learn even from individual sampled outcomes (with more noise).
    # Training value on chance nodes helps the NN understand state values.
    if pred_value_logits.ndim == 1 or pred_value_logits.shape[-1] == 1:
        pred_scalar = pred_value_logits if pred_value_logits.ndim == 1 else jnp.squeeze(pred_value_logits, axis=-1)
        value_loss_per_sample = (pred_scalar - target_reward) ** 2
        value_loss = value_loss_per_sample.mean()
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

    # total loss (with configurable value loss weight and illegal action penalty)
    loss = policy_loss + value_loss_weight * value_loss + illegal_action_penalty * illegal_action_loss + l2_reg

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

    # Debug: Check step rewards and reward targets in the experience
    step_reward_sum = experience.step_reward.sum() if hasattr(experience, 'step_reward') else 0.0
    reward_sum = experience.reward.sum() if hasattr(experience, 'reward') else 0.0

    aux_metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'illegal_action_loss': illegal_action_loss,
        'l2_reg': l2_reg,
        'policy_entropy': entropy(pred_policy_probs).mean(),
        'policy_accuracy': policy_accuracy,
        'policy_kl': policy_kl,
        'value_mse': value_loss,
        'value_mae': jnp.mean(jnp.abs((pred_scalar if 'pred_scalar' in locals() else 0.0) - target_reward)),
        # Debug metrics for value targets
        'value_target_mean': jnp.mean(target_reward),
        'value_target_std': jnp.std(target_reward),
        'value_target_max': jnp.max(target_reward),
        'value_pred_mean': jnp.mean(pred_scalar) if 'pred_scalar' in locals() else 0.0,
        # Debug step reward and reward target
        'step_reward_sum': step_reward_sum,
        'reward_sum': reward_sum,
    }

    return loss, (aux_metrics, updates)
