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

def scalar_value_to_probs(value: chex.Array) -> chex.Array:
    """Project a scalar value in [-1, 1] to a 4-way conditional distribution.

    Returns [win, gam_win_cond, gam_loss_cond, bg_rate] where:
    - win: P(win) based on value
    - gam_win_cond: P(gammon | win) = 0 for simple heuristic
    - gam_loss_cond: P(gammon | loss) = 0 for simple heuristic
    - bg_rate: P(backgammon | gammon) = 0 for simple heuristic
    """
    win_prob = jnp.clip(0.5 + 0.5 * value, 0.0, 1.0)
    return jnp.array([win_prob, 0.0, 0.0, 0.0], dtype=jnp.float32)


def probs_to_equity(value_probs: chex.Array) -> chex.Array:
    """Convert 4-way value probabilities to scalar equity for money games.

    Args:
        value_probs: [win, gam_win_cond, gam_loss_cond, bg_rate]
            - win: P(win)
            - gam_win_cond: P(gammon | win)
            - gam_loss_cond: P(gammon | loss)
            - bg_rate: P(backgammon | gammon)

    Returns:
        Scalar equity in [-3, 3] for money games where:
        - Normal win/loss: ±1
        - Gammon win/loss: ±2
        - Backgammon win/loss: ±3

    For match play, this can be replaced with match equity table lookups.
    """
    win = value_probs[0]
    gam_win_cond = value_probs[1]
    gam_loss_cond = value_probs[2]
    bg_rate = value_probs[3]

    # Expected points when winning: 1 + P(gammon|win) * (1 + P(bg|gammon))
    expected_win_points = 1.0 + gam_win_cond * (1.0 + bg_rate)
    # Expected points when losing
    expected_loss_points = 1.0 + gam_loss_cond * (1.0 + bg_rate)

    # Equity = P(win) * E[points|win] - P(loss) * E[points|loss]
    equity = win * expected_win_points - (1.0 - win) * expected_loss_points

    return equity

def bg_simple_step_fn(env: Env, state, action, key=None):
    """Simple step function for testing does not know about stochastic nodes."""
    # MCTS step_fn expects a key parameter, use it if provided, otherwise use fixed key
    step_key = key if key is not None else jax.random.PRNGKey(23)
    new_state = env.step(state, action, step_key)
    return new_state, StepMetadata(
        rewards=new_state.rewards,
        action_mask=new_state.legal_action_mask,
        terminated=new_state.terminated,
        cur_player_id=new_state.current_player,
        step=new_state._step_count
    )

def bg_step_fn(env: Env, state: bg.State, action: int, key: chex.PRNGKey) -> Tuple[bg.State, StepMetadata]:
    """Combined step function for backgammon environment that handles both deterministic and stochastic actions.

    This function preserves stochastic states at turn boundaries so that MCTS can properly
    explore dice roll outcomes. When a turn ends, instead of auto-rolling dice (which env.step does),
    we return a stochastic state requiring a dice roll.
    """

    # Handle stochastic vs deterministic branches
    def stochastic_branch(operand):
        s, a, _ = operand  # state, action, key (key ignored for stochastic step)
        return env.stochastic_step(s, a)

    def deterministic_branch(operand):
        s, a, k = operand  # state, action, key
        old_turn = s._turn
        new_state = env.step(s, a, k)
        new_turn = new_state._turn

        # Detect turn change: when turn increments, a stochastic dice roll is needed.
        # env.step() auto-rolls dice, but for proper MCTS we need to preserve the
        # stochastic state so the tree can explore different dice outcomes.
        turn_changed = new_turn != old_turn

        # Create legal action mask for stochastic state (dice roll actions 0-20 are valid)
        # There are 21 dice outcomes: 6 doubles (0-5) + 15 non-doubles (6-20)
        num_dice_outcomes = 21
        stochastic_action_mask = jnp.concatenate([
            jnp.ones(num_dice_outcomes, dtype=jnp.bool_),
            jnp.zeros(new_state.legal_action_mask.shape[0] - num_dice_outcomes, dtype=jnp.bool_)
        ])

        # When turn changes, restore stochastic state for dice roll
        def make_stochastic(ns):
            return ns.replace(
                _is_stochastic=jnp.array(True),
                _dice=jnp.array([0, 0]),  # Reset dice (not rolled yet)
                _playable_dice=jnp.array([-1, -1, -1, -1]),
                _played_dice_num=jnp.array(0),
                legal_action_mask=stochastic_action_mask
            )

        return jax.lax.cond(
            turn_changed,
            make_stochastic,
            lambda ns: ns,
            new_state
        )

    # Use conditional to route to the appropriate branch
    new_state = jax.lax.cond(
        state._is_stochastic,
        stochastic_branch,
        deterministic_branch,
        (state, action, key)
    )

    # Create standard metadata
    metadata = StepMetadata(
        rewards=new_state.rewards,
        action_mask=new_state.legal_action_mask,
        terminated=new_state.terminated,
        cur_player_id=new_state.current_player,
        step=new_state._step_count,
        is_stochastic=new_state._is_stochastic
    )

    return new_state, metadata


def make_bg_decision_step_fn(env: Env):
    """Create a decision step function for backgammon (for deterministic/player actions).

    This function preserves stochastic states at turn boundaries so that MCTS can properly
    explore dice roll outcomes. When a turn ends, instead of auto-rolling dice (which env.step does),
    we return a stochastic state requiring a dice roll.
    """
    def step_fn(state, action, key=None):
        action = jnp.asarray(action, dtype=jnp.int32)
        step_key = key if key is not None else jax.random.PRNGKey(0)
        old_turn = state._turn
        new_state = env.step(state, action, step_key)
        new_turn = new_state._turn

        # Detect turn change: when turn increments, a stochastic dice roll is needed.
        # env.step() auto-rolls dice, but for proper MCTS we need to preserve the
        # stochastic state so the tree can explore different dice outcomes.
        turn_changed = new_turn != old_turn

        # Create legal action mask for stochastic state (dice roll actions 0-20 are valid)
        num_dice_outcomes = 21
        stochastic_action_mask = jnp.concatenate([
            jnp.ones(num_dice_outcomes, dtype=jnp.bool_),
            jnp.zeros(new_state.legal_action_mask.shape[0] - num_dice_outcomes, dtype=jnp.bool_)
        ])

        # When turn changes, restore stochastic state for dice roll
        def make_stochastic(ns):
            return ns.replace(
                _is_stochastic=jnp.array(True),
                _dice=jnp.array([0, 0]),
                _playable_dice=jnp.array([-1, -1, -1, -1]),
                _played_dice_num=jnp.array(0),
                legal_action_mask=stochastic_action_mask
            )

        final_state = jax.lax.cond(
            turn_changed,
            make_stochastic,
            lambda ns: ns,
            new_state
        )

        return final_state, StepMetadata(
            rewards=final_state.rewards,
            action_mask=final_state.legal_action_mask,
            terminated=final_state.terminated,
            cur_player_id=final_state.current_player,
            step=final_state._step_count,
            is_stochastic=final_state._is_stochastic
        )
    return step_fn


def make_bg_stochastic_step_fn(env: Env):
    """Create a stochastic step function for backgammon (for dice roll outcomes)."""
    def step_fn(state, outcome, key=None):
        outcome = jnp.asarray(outcome, dtype=jnp.int32)
        # Use env.step_stochastic for chance node outcomes (dice rolls)
        new_state = env.step_stochastic(state, outcome)
        return new_state, StepMetadata(
            rewards=new_state.rewards,
            action_mask=new_state.legal_action_mask,
            terminated=new_state.terminated,
            cur_player_id=new_state.current_player,
            step=new_state._step_count,
            is_stochastic=new_state._is_stochastic
        )
    return step_fn


def make_bg_stochastic_aware_step_fn(env: Env):
    """Create a step function that uses the appropriate sub-function based on state type.

    This is a convenience function for tests and callers that need to step through
    states without knowing in advance if they're stochastic or deterministic.

    For production MCTS code, prefer using the separate decision_step_fn and
    stochastic_step_fn directly for clarity and type safety.
    """
    decision_step = make_bg_decision_step_fn(env)
    stochastic_step = make_bg_stochastic_step_fn(env)

    def step_fn(state, action, key=None):
        return jax.lax.cond(
            state._is_stochastic,
            lambda args: stochastic_step(args[0], args[1], args[2]),
            lambda args: decision_step(args[0], args[1], args[2]),
            (state, action, key)
        )
    return step_fn


# --- Pip Count Eval Fn (for test evaluator) ---
def bg_pip_count_eval(state: chex.ArrayTree, params: chex.ArrayTree, key: chex.PRNGKey) -> Tuple[Array, Array]:
    """Calculates value based on pip count difference. Ignores params/key.
    The board is always from the current players perspective, 
    current player is positive numbers opponent is negative."""
    board = state._board
    pips = state._board[0:24] # this trucates the off and bar
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
    opponent_born_off = board[27] * 25 * -1  # Opponent's born-off checkers
    current_bar = board[24] * -5
    opponent_bar = board[25] * -5 * -1

    #jax.debug.print(f"[DEBUG-BG_PIP_COUNT_EVAL] Current born-off: {current_born_off}, Opponent born-off: {opponent_born_off}")
    # Calculate normalized value between -1 and 1
    # Positive value means current player is ahead
    value = (current_pips + current_born_off + current_bar - opponent_pips - opponent_born_off - opponent_bar) / 200
    value = jnp.clip(value, -1.0, 1.0) # Clip to [-1, 1]
    value_probs = scalar_value_to_probs(value)
    #jax.debug.print(f"[DEBUG-BG_PIP_COUNT_EVAL] Value: {value}")
    
    # Uniform policy over legal actions for greedy baseline
    policy_logits = jnp.where(state.legal_action_mask, 0.0, -jnp.inf)
    
    return policy_logits, value_probs


# --- Random Evaluator ---
class BGRandomEvaluator(Evaluator):
    """An evaluator that selects actions randomly from legal moves."""

    def __init__(self):
        """Initializes the RandomEvaluator."""
        super().__init__()

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

    _, value_probs = bg_pip_count_eval(state, params, key)

    return policy_logits, value_probs


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

        # 3) Value head (4-way conditional logits)
        v = nn.LayerNorm()(x)
        v = nn.relu(v)
        v = nn.Dense(4)(v)

        return policy_logits, v
    
