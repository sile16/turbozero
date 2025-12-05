
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from core.memory.replay_memory import BaseExperience
from core.evaluators.mcts.equity import terminal_value_probs_from_reward


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
    target_value_probs = terminal_value_probs_from_reward(target_reward)
    pred_value_log_probs = jax.nn.log_softmax(pred_value_logits, axis=-1)
    value_loss = -(target_value_probs * pred_value_log_probs).sum(axis=-1).mean()

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

    # Value calibration metrics
    pred_value_probs = jax.nn.softmax(pred_value_logits, axis=-1)
    value_top1 = jnp.argmax(pred_value_probs, axis=-1)
    target_top1 = jnp.argmax(target_value_probs, axis=-1)
    value_accuracy = jnp.mean(value_top1 == target_top1)
    value_entropy = entropy(pred_value_probs).mean()

    aux_metrics = {
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'l2_reg': l2_reg,
        'policy_entropy': entropy(pred_policy_probs).mean(),
        'policy_accuracy': policy_accuracy,
        'policy_kl': policy_kl,
        'value_accuracy': value_accuracy,
        'value_entropy': value_entropy,
    }
    return loss, (aux_metrics, updates)
