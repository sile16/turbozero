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
from pgx._src.types import Array
import flax.linen as nn

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

def bg_pip_count_eval(state: chex.ArrayTree, params: chex.ArrayTree, key: chex.PRNGKey) -> Tuple[Array, Array]:
    """Calculates value based on pip count difference. Ignores params/key.
    The board is always from the current players perspective, 
    current player is positive numbers opponent is negative."""
    board = state._board
    pips = state._board[0:24]
    opponent_pips = -pips[::-1]


    #jax.debug.print(f"[DEBUG-BG_PIP_COUNT_EVAL] Board: {board}")
    
    # Calculate pip counts for current player and opponent
    # make an array of 1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19, 20.20, 21.21, 22.22, 23.23
    pip_weights = jnp.arange(0, 24, dtype=jnp.float32) + jnp.arange(0, 24, dtype=jnp.float32) / 10.0

    # Calculate pip counts for current player and opponent
    current_pips = jnp.sum(jnp.maximum(0, pips) * pip_weights)
    opponent_pips = jnp.sum(jnp.maximum(0, opponent_pips) * pip_weights)

    #jax.debug.print(f"[DEBUG-BG_PIP_COUNT_EVAL] Current pips: {current_pips}, Opponent pips: {opponent_pips}")
    #jax.debug.print(f"[DEBUG-BG_PIP_COUNT_EVAL] Current born-off: {board[26]}, Opponent born-off: {board[27]}")
    
    # Add born-off checkers with appropriate weights
    # Using 25 points for born-off checkers (standard backgammon pip count)
    current_born_off = board[26] * 25  # Current player's born-off checkers
    opponent_born_off = board[27] * 25  # Opponent's born-off checkers
    current_bar = board[bg._bar_idx()] * -5
    opponent_bar = board[bg._bar_idx() + 1] * -5

    #jax.debug.print(f"[DEBUG-BG_PIP_COUNT_EVAL] Current born-off: {current_born_off}, Opponent born-off: {opponent_born_off}")
    # Calculate normalized value between -1 and 1
    # Positive value means current player is ahead
    value = (current_pips + current_born_off + current_bar- opponent_pips - opponent_born_off - opponent_bar) / 400
    value = jnp.clip(value, -1.0, 1.0) # Clip to [-1, 1]
    #jax.debug.print(f"[DEBUG-BG_PIP_COUNT_EVAL] Value: {value}")
    
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
    

def bg_hit2_eval(state: bg.State, params: chex.ArrayTree, key: chex.PRNGKey) -> Tuple[Array, Array]:
    """
    Implements the Hit2 strategy evaluator for Backgammon.

    Value: Based on the difference between opponent's checkers on the bar
           and the player's checkers on the bar.
           e(s) = 0.055 * (opponent_bar_checkers - own_bar_checkers)

    Policy: Prioritizes actions that hit opponent's blots.
            Hitting moves get a logit of 1.0, other legal moves get 0.0.
            Illegal moves get -inf.
    
    Args:
        state: The current backgammon State object.
        params: Network parameters (ignored for this evaluator).
        key: PRNGKey (ignored for this evaluator).

    Returns:
        A tuple containing:
        - policy_logits: An array of logits for each possible action.
        - value: A scalar value estimate for the current state.
    """
    board = state._board
    legal_action_mask = state.legal_action_mask
    # --- Calculate Policy Logits ---

    # Define a function to check if a single action is a hitting move
    def is_hitting_move(action: Array, current_board: Array) -> bool:
        _, _, tgt = bg._decompose_action(action)
        
        # Check if target is a valid board point (0-23)
        is_board_target = (tgt >= 0) & (tgt <= 23)
        
        # Check if the target point has exactly one opponent checker (-1)
        is_hit = is_board_target & (current_board[tgt] == -1)
        
        return is_hit

    # Vectorize the check over all possible actions (0 to 26*6 - 1)
    all_actions = jnp.arange(26 * 6, dtype=jnp.int32)
    # Check which actions *would* result in a hit, regardless of legality for now
    potential_hits = jax.vmap(is_hitting_move, in_axes=(0, None))(all_actions, board)

    # Assign scores: 1.0 for potential hits, 0.0 otherwise
    hit_scores = potential_hits.astype(jnp.float32) # 1.0 if hit, 0.0 otherwise

    # Combine scores with the legal action mask
    # - Legal hitting moves get logit 1.0
    # - Legal non-hitting moves get logit 0.0
    # - Illegal moves get logit -inf
    policy_logits = jnp.where(legal_action_mask, hit_scores, -jnp.inf)

    # Final check: if ONLY no-op actions are legal, ensure they have logit 0.0
    # This should already be handled correctly by the jnp.where above, as no-op
    # cannot be a hit (hit_score=0.0) and will be selected if legal_action_mask allows it.
    # If no moves are legal at all, all logits will be -inf, which is also correct.

    _, value = bg_pip_count_eval(state, params, key)

    return policy_logits, value


# Pre‑activation ResNet‑V2 block
class ResBlockV2(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        r = x
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.features, use_bias=False)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.features, use_bias=False)(x)
        return x + r

class ResNetTurboZero(nn.Module):
    num_actions: int            # 156 here
    hidden_dim: int = 256
    num_blocks: int = 10

    @nn.compact
    def __call__(self, x, train: bool = False):
        # 1) ResNet tower
        x = nn.Dense(self.hidden_dim, use_bias=False)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        for _ in range(self.num_blocks):
            x = ResBlockV2(self.hidden_dim)(x)

        # 2) Policy head: single Dense into 156 logits
        policy_logits = nn.Dense(self.num_actions)(x)

        # 3) Value head
        v = nn.LayerNorm()(x)
        v = nn.relu(v)
        v = nn.Dense(1)(v)
        v = jnp.squeeze(v, -1)

        return policy_logits, v
    