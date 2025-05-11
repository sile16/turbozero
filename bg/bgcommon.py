"""
Common functions for the Backgammon environment.
"""
from typing import Tuple

import jax
import jax.numpy as jnp
import chex

from pgx import backgammon as bg
from pgx.core import Env

from core.types import StepMetadata
from core.evaluators.evaluator import Evaluator, EvalOutput
from core.types import EnvStepFn


def bg_simple_step_fn(env: Env, state, action):
    """Simple step function for testing does not know about stochastic nodes."""
    # MCTS step_fn doesn't take a key parameter
    step_key = jax.random.PRNGKey(23)  # Use a fixed key for determinism
    new_state = env.step(state, action, step_key)
    return new_state, StepMetadata(
        rewards=new_state.rewards,
        action_mask=new_state.legal_action_mask,
        terminated=new_state.terminated,
        cur_player_id=new_state.current_player,
        step=new_state._step_count
    )

def bg_step_fn(env: Env, state: bg.State, action: int, key: chex.PRNGKey) -> Tuple[bg.State, StepMetadata]:
    """Combined step function for backgammon environment that handles both deterministic and stochastic actions."""
    # print(f"[DEBUG-BG_STEP-{time.time()}] Called with state (stochastic={state.is_stochastic}), action={action}") # Optional debug

    # Handle stochastic vs deterministic branches
    def stochastic_branch(operand):
        s, a, _ = operand # state, action, key (key ignored for stochastic step)
        # Use env instance captured by closure (assuming env is accessible in this scope)
        return env.stochastic_step(s, a)

    def deterministic_branch(operand):
        s, a, k = operand # state, action, key
        # Use env instance captured by closure
        return env.step(s, a, k)

    # Use conditional to route to the appropriate branch
    # The key is only needed for the deterministic branch
    new_state = jax.lax.cond(
        state.is_stochastic,
        stochastic_branch,
        deterministic_branch,
        (state, action, key) # Pass all required operands
    )

    # Create standard metadata
    metadata = StepMetadata(
        rewards=new_state.rewards,
        action_mask=new_state.legal_action_mask,
        terminated=new_state.terminated,
        cur_player_id=new_state.current_player,
        step=new_state._step_count
    )

    return new_state, metadata


# --- Pip Count Eval Fn (for test evaluator) ---
@jax.jit
def bg_pip_count_eval(state: chex.ArrayTree, params: chex.ArrayTree, key: chex.PRNGKey):
    """Calculates value based on pip count difference. Ignores params/key.
    The board is always from the current players perspective, 
    current player is positive numbers opponent is negative."""
    board = state._board
    pips = state._board[1:25]

    jax.debug.print(f"[DEBUG-BG_PIP_COUNT_EVAL] Board: {board}")
    
    # Calculate pip counts for current player and opponent
    current_pips = jnp.sum(jnp.maximum(0, pips) * jnp.arange(1, 25, dtype=jnp.int32))
    opponent_pips = jnp.sum(jnp.maximum(0, -pips) * jnp.arange(1, 25, dtype=jnp.int32))

    jax.debug.print(f"[DEBUG-BG_PIP_COUNT_EVAL] Current pips: {current_pips}, Opponent pips: {opponent_pips}")
    jax.debug.print(f"[DEBUG-BG_PIP_COUNT_EVAL] Current born-off: {board[26]}, Opponent born-off: {board[27]}")
    
    # Add born-off checkers with appropriate weights
    # Using 25 points for born-off checkers (standard backgammon pip count)
    current_born_off = board[26] * 25  # Current player's born-off checkers
    opponent_born_off = board[27] * 25  # Opponent's born-off checkers

    #jax.debug.print(f"[DEBUG-BG_PIP_COUNT_EVAL] Current born-off: {current_born_off}, Opponent born-off: {opponent_born_off}")
    #jax.debug.print(f"[DEBUG-BG_PIP_COUNT_EVAL] Total pips: {total_pips}")
    
    # Calculate total pips for normalization
    total_pips = current_pips + opponent_pips + current_born_off + opponent_born_off + 1e-6

    #jax.debug.print(f"[DEBUG-BG_PIP_COUNT_EVAL] Total pips: {total_pips}")
    
    # Calculate normalized value between -1 and 1
    # Positive value means current player is ahead
    value = (opponent_pips + opponent_born_off - current_pips - current_born_off) / total_pips

    #jax.debug.print(f"[DEBUG-BG_PIP_COUNT_EVAL] Value: {value}")
    #jax.debug.print(f"[DEBUG-BG_PIP_COUNT_EVAL] State: {state}")
    
    # Uniform policy over legal actions for greedy baseline
    policy_logits = jnp.where(state.legal_action_mask, 0.0, -jnp.inf)
    
    return policy_logits, jnp.array(value)


# --- Random Evaluator ---
class BGRandomEvaluator(Evaluator):
    """An evaluator that selects actions randomly from legal moves."""

    def __init__(self, discount: float = -1.0):
        """Initializes the RandomEvaluator."""
        super().__init__(discount=discount)

    def evaluate(self, key: chex.PRNGKey, eval_state: chex.ArrayTree, env_state: chex.ArrayTree,
                 root_metadata: StepMetadata, params: chex.ArrayTree, env_step_fn: EnvStepFn, **kwargs) -> EvalOutput:
        """Chooses a random legal action."""
        action_mask = root_metadata.action_mask
        num_actions = action_mask.shape[-1]
        
        # Create uniform policy over legal actions
        legal_actions_count = jnp.sum(action_mask)
        uniform_prob = jnp.where(legal_actions_count > 0, 1.0 / legal_actions_count, 0.0)
        policy_weights = jnp.where(action_mask, uniform_prob, 0.0)
        
        # Sample a random action from the legal ones
        action = jax.random.choice(key, jnp.arange(num_actions), p=policy_weights)
        
        return EvalOutput(
            eval_state=eval_state, 
            action=action,
            policy_weights=policy_weights
        )

    def get_value(self, state: chex.ArrayTree) -> chex.Array:
        """Returns a zero value estimate."""
        return jnp.array(0.0)

    def init(self, template_embedding: chex.ArrayTree, *args, **kwargs) -> chex.ArrayTree:
        """Initializes the dummy state (can be empty)."""
        return jnp.array(0) # Return a minimal placeholder state

    def reset(self, state: chex.ArrayTree) -> chex.ArrayTree:
        """Resets the dummy state."""
        return state # Stateless, nothing to reset

    def step(self, state: chex.ArrayTree, action: int) -> chex.ArrayTree:
        """Updates the dummy state (no change needed)."""
        return state # Stateless, nothing to update based on action