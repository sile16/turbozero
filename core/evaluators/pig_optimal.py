"""
Optimal (Perfect) Play Strategy for the Game of Pig
====================================================

Computes the optimal strategy using dynamic programming / value iteration.
The result is a lookup table mapping (my_score, opp_score, turn_total) -> action.

State space: 100 x 100 x 100 = 1,000,000 states
- my_score: 0-99 (first to 100 wins)
- opp_score: 0-99
- turn_total: 0-99 (practical limit)

Actions:
- 0: Roll
- 1: Hold

References:
- Neller & Presser, "Optimal Play of the Dice Game Pig" (2004)
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple
import os
import chex

from core.evaluators.evaluator import Evaluator, EvalOutput


@chex.dataclass(frozen=True)
class PigOptimalEvalState:
    """Eval state for PigPerfectPlayEvaluator - stores value estimate."""
    value: chex.Array  # Win probability converted to [-1, 1]

# Constants
GOAL = 100
MAX_TURN_TOTAL = 100  # Practical limit


def compute_optimal_strategy(
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optimal Pig strategy using value iteration.

    Uses a simple loop-based approach for correctness (vectorization was buggy).

    Returns:
        win_prob: Array of shape (100, 100, 100) with win probabilities
        optimal_action: Array of shape (100, 100, 100) with optimal actions (0=roll, 1=hold)
    """
    # V[i, j, k] = P(win | my_score=i, opp_score=j, turn_total=k)
    V = np.zeros((GOAL, GOAL, MAX_TURN_TOTAL), dtype=np.float64)

    if verbose:
        print("Computing optimal Pig strategy...")

    for iteration in range(max_iterations):
        V_old = V.copy()

        for i in range(GOAL):
            for j in range(GOAL):
                for k in range(MAX_TURN_TOTAL):
                    # If I can win by holding, win prob = 1
                    if i + k >= GOAL:
                        V[i, j, k] = 1.0
                        continue

                    # Value of holding: bank k points, opponent plays from (j, i+k, 0)
                    # My win prob = 1 - opponent's win prob
                    v_hold = 1.0 - V[j, i + k, 0]

                    # Value of rolling:
                    # 1/6 chance: roll 1, bust, opponent plays from (j, i, 0)
                    v_bust = 1.0 - V[j, i, 0]

                    # 5/6 chance: roll 2-6 (uniform)
                    v_continue = 0.0
                    for die in range(2, 7):
                        new_k = min(k + die, MAX_TURN_TOTAL - 1)
                        if i + new_k >= GOAL:
                            v_continue += 1.0  # Can win
                        else:
                            v_continue += V[i, j, new_k]
                    v_continue /= 5.0

                    v_roll = (1.0/6.0) * v_bust + (5.0/6.0) * v_continue

                    V[i, j, k] = max(v_hold, v_roll)

        delta = np.max(np.abs(V - V_old))
        if verbose and iteration % 10 == 0:
            print(f"  Iteration {iteration}: max delta = {delta:.2e}")

        if delta < tolerance:
            if verbose:
                print(f"  Converged after {iteration + 1} iterations")
            break

    # Extract optimal actions
    optimal_action = np.zeros((GOAL, GOAL, MAX_TURN_TOTAL), dtype=np.int8)

    for i in range(GOAL):
        for j in range(GOAL):
            for k in range(MAX_TURN_TOTAL):
                if i + k >= GOAL:
                    optimal_action[i, j, k] = 1  # Hold to win
                    continue

                v_hold = 1.0 - V[j, i + k, 0]
                v_bust = 1.0 - V[j, i, 0]

                v_continue = 0.0
                for die in range(2, 7):
                    new_k = min(k + die, MAX_TURN_TOTAL - 1)
                    if i + new_k >= GOAL:
                        v_continue += 1.0
                    else:
                        v_continue += V[i, j, new_k]
                v_continue /= 5.0

                v_roll = (1.0/6.0) * v_bust + (5.0/6.0) * v_continue

                optimal_action[i, j, k] = 1 if v_hold >= v_roll else 0

    return V, optimal_action


def get_optimal_threshold_approximation() -> np.ndarray:
    """
    Get a simplified threshold-based approximation of optimal play.

    This returns for each (my_score, opp_score) the minimum turn_total at which
    to hold. This is a compact representation often used in practice.

    Returns:
        threshold: Array of shape (100, 100) where threshold[i,j] is the
                   minimum turn_total at which to hold when my_score=i, opp_score=j
    """
    _, optimal_action = compute_optimal_strategy(verbose=False)

    threshold = np.zeros((GOAL, GOAL), dtype=np.int8)

    for i in range(GOAL):
        for j in range(GOAL):
            # Find first k where optimal action is hold
            for k in range(MAX_TURN_TOTAL):
                if optimal_action[i, j, k] == 1:
                    threshold[i, j] = k
                    break
            else:
                threshold[i, j] = MAX_TURN_TOTAL  # Always roll (shouldn't happen)

    return threshold


class PigOptimalLookup:
    """
    Optimal Pig strategy lookup table.

    Precomputes and caches the optimal strategy for fast lookup.
    """

    _instance = None
    _win_prob = None
    _optimal_action = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._load_or_compute()
        return cls._instance

    @classmethod
    def _load_or_compute(cls):
        """Load from cache or compute the strategy."""
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cache')
        cache_file = os.path.join(cache_dir, 'pig_optimal.npz')

        if os.path.exists(cache_file):
            print(f"Loading cached Pig optimal strategy from {cache_file}")
            data = np.load(cache_file)
            cls._win_prob = data['win_prob']
            cls._optimal_action = data['optimal_action']
        else:
            print("Computing Pig optimal strategy (this takes ~30 seconds)...")
            cls._win_prob, cls._optimal_action = compute_optimal_strategy()

            # Cache for future use
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cache_file,
                              win_prob=cls._win_prob,
                              optimal_action=cls._optimal_action)
            print(f"Cached to {cache_file}")

    def get_action(self, my_score: int, opp_score: int, turn_total: int) -> int:
        """Get optimal action (0=roll, 1=hold) for given state."""
        i = min(my_score, GOAL - 1)
        j = min(opp_score, GOAL - 1)
        k = min(turn_total, MAX_TURN_TOTAL - 1)
        return int(self._optimal_action[i, j, k])

    def get_win_prob(self, my_score: int, opp_score: int, turn_total: int) -> float:
        """Get win probability for given state under optimal play."""
        i = min(my_score, GOAL - 1)
        j = min(opp_score, GOAL - 1)
        k = min(turn_total, MAX_TURN_TOTAL - 1)
        return float(self._win_prob[i, j, k])

    @property
    def optimal_action_table(self) -> np.ndarray:
        """Return the full optimal action lookup table."""
        return self._optimal_action

    @property
    def win_prob_table(self) -> np.ndarray:
        """Return the full win probability table."""
        return self._win_prob


# JAX-compatible functions for use in evaluators
def create_jax_lookup_fn():
    """
    Create a JAX-compatible lookup function.

    Returns a function that can be jitted and vmapped.
    """
    lookup = PigOptimalLookup()
    action_table = jnp.array(lookup.optimal_action_table)

    def lookup_action(my_score: jnp.ndarray, opp_score: jnp.ndarray,
                      turn_total: jnp.ndarray) -> jnp.ndarray:
        """JAX-compatible optimal action lookup."""
        i = jnp.clip(my_score, 0, GOAL - 1).astype(jnp.int32)
        j = jnp.clip(opp_score, 0, GOAL - 1).astype(jnp.int32)
        k = jnp.clip(turn_total, 0, MAX_TURN_TOTAL - 1).astype(jnp.int32)
        return action_table[i, j, k]

    return lookup_action


class PigPerfectPlayEvaluator(Evaluator):
    """
    Perfect play evaluator for Pig using precomputed optimal strategy.

    Uses dynamic programming lookup table to return the optimal action
    for any game state.
    """

    def __init__(self):
        super().__init__()
        self._lookup = PigOptimalLookup()
        # Store JAX arrays for fast lookup
        self._action_table = jnp.array(self._lookup.optimal_action_table)
        self._win_prob_table = jnp.array(self._lookup.win_prob_table)

    def init(self, template_state=None, template_embedding=None):
        """Initialize evaluator state with default value."""
        return PigOptimalEvalState(value=jnp.array(0.0))

    def reset(self, state):
        """Reset evaluator state."""
        return PigOptimalEvalState(value=jnp.array(0.0))

    def evaluate(self, key, eval_state, env_state, root_metadata, params=None, env_step_fn=None, **kwargs):
        """
        Return optimal action based on lookup table.

        Args:
            key: Random key (unused)
            eval_state: Evaluator state (unused)
            env_state: PGX Pig game state
            root_metadata: Step metadata
            params: NN params (unused)
            env_step_fn: Step function (unused)

        Returns:
            EvalOutput with optimal action and policy weights
        """
        # Extract scores from observation
        # Pig observation: [my_score/100, opp_score/100, turn_total/100]
        obs = env_state.observation

        # Works for both batched (..., 3) and unbatched (3,) observations
        my_score = (obs[..., 0] * 100).astype(jnp.int32)
        opp_score = (obs[..., 1] * 100).astype(jnp.int32)
        turn_total = (obs[..., 2] * 100).astype(jnp.int32)

        # Clip to valid range
        my_score = jnp.clip(my_score, 0, GOAL - 1)
        opp_score = jnp.clip(opp_score, 0, GOAL - 1)
        turn_total = jnp.clip(turn_total, 0, MAX_TURN_TOTAL - 1)

        # Lookup optimal action
        action = self._action_table[my_score, opp_score, turn_total]

        # Create policy weights (one-hot for the optimal action)
        # Pig has 6 actions: 0=roll, 1=hold, 2-5=dice outcomes
        policy = jax.nn.one_hot(action, 6)

        # Compute value estimate (win probability in [-1, 1])
        win_prob = self._win_prob_table[my_score, opp_score, turn_total]
        value = 2.0 * win_prob - 1.0  # Map [0,1] -> [-1,1]

        # Store value in new eval_state
        new_eval_state = PigOptimalEvalState(value=value)

        return EvalOutput(
            eval_state=new_eval_state,
            action=action,
            policy_weights=policy
        )

    def get_value(self, eval_state: PigOptimalEvalState) -> jnp.ndarray:
        """Get win probability for current state under optimal play.

        The value was computed and stored in eval_state during evaluate().
        """
        return eval_state.value

    def get_config(self):
        return {"strategy": "optimal_dp"}


if __name__ == '__main__':
    # Compute and display some statistics
    print("=" * 60)
    print("Pig Optimal Strategy Computation")
    print("=" * 60)

    V, A = compute_optimal_strategy(verbose=True)

    print()
    print("Sample optimal actions (my_score, opp_score, turn_total -> action):")
    test_cases = [
        (0, 0, 0),    # Game start
        (0, 0, 20),   # Hold20 threshold
        (0, 0, 25),   # Above Hold20
        (50, 50, 0),  # Mid-game start of turn
        (50, 50, 15), # Mid-game with some points
        (90, 50, 5),  # Close to winning
        (50, 90, 5),  # Opponent close to winning
        (95, 95, 0),  # Both close
    ]

    for i, j, k in test_cases:
        action = "HOLD" if A[i, j, k] == 1 else "ROLL"
        win_p = V[i, j, k]
        print(f"  ({i:2d}, {j:2d}, {k:2d}) -> {action:4s}  (win prob: {win_p:.3f})")

    print()
    print("Win probability from (0, 0, 0):", V[0, 0, 0])
    print("This is the expected win rate for player 1 under optimal play by both.")

    # Compute average hold threshold
    thresholds = get_optimal_threshold_approximation()
    avg_threshold = np.mean(thresholds[thresholds < MAX_TURN_TOTAL])
    print(f"Average hold threshold: {avg_threshold:.1f}")
