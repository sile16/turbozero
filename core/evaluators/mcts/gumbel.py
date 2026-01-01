"""Gumbel AlphaZero / Gumbel MuZero action selection.

Full implementation of "Policy improvement by planning with Gumbel" (ICLR 2022)
https://openreview.net/forum?id=bERaNdoegnO

Key innovations:
1. Gumbel-Top-k sampling replaces Dirichlet noise for exploration
2. Sequential Halving progressively eliminates actions
3. σ(q̂) scaling normalizes Q-values for action selection
4. Improved policy computed from g(a) + logits(a) + σ(q̂(a))

This guarantees policy improvement even with very few simulations (2-16).
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import chex


def gumbel_top_k(
    key: chex.PRNGKey,
    logits: chex.Array,
    k: int,
    mask: Optional[chex.Array] = None
) -> Tuple[chex.Array, chex.Array]:
    """Sample k actions without replacement using Gumbel-Top-k trick.

    Args:
        key: JAX random key
        logits: Log probabilities (unnormalized) for each action [num_actions]
        k: Number of actions to sample (must be static for JIT)
        mask: Optional boolean mask of legal actions (True=legal)

    Returns:
        Tuple of:
        - selected_actions: Indices of k selected actions [k]
        - gumbel_logits: Gumbel + logits values for all actions [num_actions]
    """
    # Sample Gumbel noise: g ~ Gumbel(0, 1) = -log(-log(u)), u ~ Uniform(0, 1)
    u = jax.random.uniform(key, shape=logits.shape, minval=1e-10, maxval=1.0 - 1e-10)
    gumbels = -jnp.log(-jnp.log(u))

    # Perturbed log-probs: g(a) + log π(a)
    gumbel_logits = logits + gumbels

    # Mask out illegal actions
    if mask is not None:
        gumbel_logits = jnp.where(mask, gumbel_logits, -jnp.inf)

    # Get top-k indices
    _, selected_actions = jax.lax.top_k(gumbel_logits, k)

    return selected_actions, gumbel_logits


def sigma_q(
    q_values: chex.Array,
    max_visit_count: chex.Array,
    c_visit: float = 50.0,
    c_scale: float = 1.0,
) -> chex.Array:
    """Compute σ(q̂) - the scaled Q-values for Gumbel policy improvement.

    From the paper:
        σ(q̂(a)) = c_scale · q̂(a) / (c_visit + max_b N(b))

    This scaling ensures Q-values are on a comparable scale to the Gumbel + logits.

    Args:
        q_values: Q-values from MCTS search [num_actions] or [k]
        max_visit_count: Maximum visit count among all actions
        c_visit: Visit count offset (default 50 from paper)
        c_scale: Scale factor (default 1.0 from paper)

    Returns:
        Scaled Q-values [num_actions] or [k]
    """
    return c_scale * q_values / (c_visit + max_visit_count)


def compute_gumbel_score(
    gumbel_logits: chex.Array,
    q_values: chex.Array,
    max_visit_count: chex.Array,
    c_visit: float = 50.0,
    c_scale: float = 1.0,
) -> chex.Array:
    """Compute the Gumbel score: g(a) + logits(a) + σ(q̂(a)).

    This is the key quantity for action selection and policy improvement.

    Args:
        gumbel_logits: g(a) + log π(a) from Gumbel sampling [num_actions]
        q_values: Q-values from MCTS [num_actions]
        max_visit_count: Maximum visit count among all actions
        c_visit: Visit count offset for σ function
        c_scale: Scale factor for σ function

    Returns:
        Gumbel scores for all actions [num_actions]
    """
    scaled_q = sigma_q(q_values, max_visit_count, c_visit, c_scale)
    return gumbel_logits + scaled_q


def sequential_halving_phases(num_simulations: int, num_actions: int) -> int:
    """Compute number of phases for Sequential Halving.

    The budget is divided into ceil(log2(m)) phases where m is initial action count.

    Args:
        num_simulations: Total simulation budget (n)
        num_actions: Number of actions to consider (m)

    Returns:
        Number of phases
    """
    import math
    # log2(m) phases, but at least 1
    # Use Python math to avoid JAX tracing issues
    return max(1, math.ceil(math.log2(max(num_actions, 1))))


def sequential_halving_schedule(
    num_simulations: int,
    num_actions: int,
) -> Tuple[chex.Array, chex.Array]:
    """Compute the Sequential Halving schedule.

    Divides n simulations into log2(m) phases. Each phase has equal budget
    distributed among remaining actions. After each phase, half the actions
    are eliminated.

    Args:
        num_simulations: Total simulation budget (n)
        num_actions: Initial number of actions to consider (m)

    Returns:
        Tuple of:
        - phase_budgets: Simulations per action in each phase [num_phases]
        - actions_per_phase: Number of actions remaining in each phase [num_phases]
    """
    num_phases = sequential_halving_phases(num_simulations, num_actions)

    # Actions remaining at start of each phase: m, m/2, m/4, ...
    actions_per_phase = jnp.array([
        max(1, num_actions // (2 ** i)) for i in range(num_phases)
    ])

    # Budget per action per phase = n / (sum of actions across phases)
    total_action_visits = jnp.sum(actions_per_phase)
    budget_per_action = num_simulations / jnp.maximum(total_action_visits, 1)

    # Each phase visits all remaining actions equally
    phase_budgets = jnp.full(num_phases, budget_per_action)

    return phase_budgets, actions_per_phase


def select_action_after_halving(
    gumbel_logits: chex.Array,
    q_values: chex.Array,
    visit_counts: chex.Array,
    selected_actions: chex.Array,
    legal_mask: chex.Array,
    c_visit: float = 50.0,
    c_scale: float = 1.0,
) -> int:
    """Select the final action after Sequential Halving completes.

    The action with the highest g(a) + logits(a) + σ(q̂(a)) among the
    most-visited actions is selected.

    Args:
        gumbel_logits: g(a) + log π(a) values [num_actions]
        q_values: Q-values from MCTS [num_actions]
        visit_counts: Visit counts from MCTS [num_actions]
        selected_actions: Actions that were sampled [k]
        legal_mask: Boolean mask of legal actions [num_actions]
        c_visit: Visit count offset for σ function
        c_scale: Scale factor for σ function

    Returns:
        Selected action index
    """
    max_visits = jnp.max(visit_counts)

    # Compute Gumbel scores
    scores = compute_gumbel_score(gumbel_logits, q_values, max_visits, c_visit, c_scale)

    # Only consider visited actions (those that survived halving)
    # An action is "surviving" if it has visits close to max
    # Use threshold of max_visits / 2 to identify survivors
    min_survivor_visits = jnp.maximum(max_visits / 2, 1)
    is_survivor = visit_counts >= min_survivor_visits

    # Mask out non-survivors and illegal actions
    masked_scores = jnp.where(
        jnp.logical_and(is_survivor, legal_mask),
        scores,
        -jnp.inf
    )

    return jnp.argmax(masked_scores)


def compute_improved_policy(
    logits: chex.Array,
    q_values: chex.Array,
    visit_counts: chex.Array,
    legal_mask: chex.Array,
    root_value: chex.Array,
    c_visit: float = 50.0,
    c_scale: float = 1.0,
) -> chex.Array:
    """Compute the improved policy for training.

    From the paper "Policy improvement by planning with Gumbel" (ICLR 2022):
        π'(a) ∝ exp(logits(a) + σ(q̂(a)))

    where q̂(a) is the "completed" Q-value:
    - For visited actions: the MCTS Q-value
    - For unvisited actions: root_value (bootstrap from NN)

    This preserves information about unvisited actions through the prior logits,
    ensuring the network doesn't "forget" them even with small k.

    Args:
        logits: log π(a) from neural network [num_actions]
        q_values: Q-values from MCTS [num_actions]
        visit_counts: Visit counts from MCTS [num_actions]
        legal_mask: Boolean mask of legal actions [num_actions]
        root_value: Value estimate at root (for completing unvisited Q-values)
        c_visit: Visit count offset for σ function (default 50 from paper)
        c_scale: Scale factor for σ function (default 1.0 from paper)

    Returns:
        Improved policy weights [num_actions]
    """
    max_visits = jnp.max(visit_counts)

    # Complete Q-values: use root_value for unvisited actions
    # This is crucial for preserving information about unexplored actions
    is_visited = visit_counts > 0
    completed_q = jnp.where(is_visited, q_values, root_value)

    # Compute σ(q̂) scaling
    scaled_q = sigma_q(completed_q, max_visits, c_visit, c_scale)

    # Improved policy: π'(a) ∝ exp(logits(a) + σ(q̂(a)))
    # Note: We use logits (log π(a)), NOT gumbel_logits (g(a) + log π(a))
    # The Gumbel noise g(a) is for action selection, not for policy target
    improved_logits = logits + scaled_q

    # Mask illegal actions
    improved_logits = jnp.where(legal_mask, improved_logits, -jnp.inf)

    # Softmax to get probabilities
    policy = jax.nn.softmax(improved_logits)

    # Ensure proper normalization (handle edge cases)
    policy_sum = jnp.sum(policy)
    policy = jnp.where(
        policy_sum > 0,
        policy / policy_sum,
        jnp.where(legal_mask, 1.0 / jnp.maximum(jnp.sum(legal_mask), 1), 0.0)
    )

    return policy


def get_action_to_simulate(
    iteration: int,
    selected_actions: chex.Array,
    visit_counts: chex.Array,
    active_mask: chex.Array,
    key: chex.PRNGKey,
    legal_mask: Optional[chex.Array] = None,
) -> int:
    """Get the action to simulate on this iteration during Sequential Halving.

    Within each phase, we visit all active actions equally. The action with
    the minimum visits among active actions is selected.

    Args:
        iteration: Current iteration number
        selected_actions: Actions selected by Gumbel-Top-k [k]
        visit_counts: Current visit counts for all actions [num_actions]
        active_mask: Boolean mask of actions still active in halving [k]
        key: Random key for tiebreaking
        legal_mask: Optional boolean mask of legal actions [num_actions]

    Returns:
        Action index to simulate
    """
    # Get visits for selected actions
    selected_visits = visit_counts[selected_actions]

    # Combine active mask with legal mask if provided
    if legal_mask is not None:
        selected_legal = legal_mask[selected_actions]
        effective_mask = jnp.logical_and(active_mask, selected_legal)
    else:
        effective_mask = active_mask

    # Mask out eliminated/illegal actions (set visits to infinity)
    masked_visits = jnp.where(effective_mask, selected_visits, jnp.inf)

    # Select action with minimum visits (ensures equal distribution within phase)
    # Add small noise for tiebreaking
    noise = jax.random.uniform(key, shape=masked_visits.shape, maxval=1e-6)
    action_idx = jnp.argmin(masked_visits + noise)

    return selected_actions[action_idx]


def eliminate_half(
    selected_actions: chex.Array,
    active_mask: chex.Array,
    gumbel_logits: chex.Array,
    q_values: chex.Array,
    visit_counts: chex.Array,
    c_visit: float = 50.0,
    c_scale: float = 1.0,
) -> chex.Array:
    """Eliminate half of the active actions based on Gumbel scores.

    Actions with lower g(a) + logits(a) + σ(q̂(a)) are eliminated.

    Args:
        selected_actions: Actions selected by Gumbel-Top-k [k]
        active_mask: Boolean mask of currently active actions [k]
        gumbel_logits: g(a) + log π(a) values [num_actions]
        q_values: Q-values from MCTS [num_actions]
        visit_counts: Visit counts from MCTS [num_actions]
        c_visit: Visit count offset for σ function
        c_scale: Scale factor for σ function

    Returns:
        Updated active_mask with half the actions eliminated [k]
    """
    max_visits = jnp.max(visit_counts)

    # Compute Gumbel scores for selected actions
    selected_gumbel = gumbel_logits[selected_actions]
    selected_q = q_values[selected_actions]
    scores = selected_gumbel + sigma_q(selected_q, max_visits, c_visit, c_scale)

    # Mask out already-eliminated actions
    masked_scores = jnp.where(active_mask, scores, -jnp.inf)

    # Find median score among active actions
    num_active = jnp.sum(active_mask)
    num_to_keep = jnp.maximum(num_active // 2, 1)

    # Keep top half by score
    # Sort scores descending and get threshold
    sorted_scores = jnp.sort(masked_scores)[::-1]
    threshold = sorted_scores[num_to_keep - 1]

    # Keep actions above threshold (with tiebreaking for equal scores)
    new_active = jnp.logical_and(active_mask, scores >= threshold)

    # Ensure we keep at least one action
    any_active = jnp.any(new_active)
    new_active = jnp.where(
        any_active,
        new_active,
        active_mask.at[jnp.argmax(masked_scores)].set(True)
    )

    return new_active
