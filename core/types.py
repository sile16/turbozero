
from typing import Callable, Tuple

import chex
from flax.training.train_state import TrainState
import jax
import optax

from core.memory.replay_memory import BaseExperience

@chex.dataclass(frozen=True)
class StepMetadata:
    """Metadata for a step in the environment.
    - `rewards`: rewards received by the players
    - `action_mask`: mask of valid actions
    - `terminated`: whether the environment is terminated
    - `cur_player_id`: current player id
    - `step`: step number
    - `match_score`: optional match score context (e.g., (player_score, opp_score, match_length))
    - `cube_value`: current cube value for backgammon-style games
    - `is_stochastic`: whether current state is a chance node (for StochasticMCTS)
    - `stochastic_action_mask`: mask of valid stochastic actions (for 2048: empty positions)
    """
    rewards: chex.Array
    action_mask: chex.Array
    terminated: bool
    cur_player_id: int
    step: int
    match_score: chex.ArrayTree | None = None
    cube_value: float = 1.0
    is_stochastic: bool = False
    stochastic_action_mask: chex.Array | None = None
    

EnvStepFn = Callable[[chex.ArrayTree, int], Tuple[chex.ArrayTree, StepMetadata]]
EnvInitFn = Callable[[jax.random.PRNGKey], Tuple[chex.ArrayTree, StepMetadata]]  
DataTransformFn = Callable[[chex.Array, chex.Array, chex.ArrayTree], Tuple[chex.Array, chex.Array, chex.ArrayTree]]
Params = chex.ArrayTree
EvalFn = Callable[[chex.ArrayTree, Params, jax.random.PRNGKey], Tuple[chex.Array, chex.Array]]
LossFn = Callable[[chex.ArrayTree, TrainState, BaseExperience], Tuple[chex.Array, Tuple[chex.ArrayTree, optax.OptState]]]
ExtractModelParamsFn = Callable[[TrainState], chex.ArrayTree]
StateToNNInputFn = Callable[[chex.ArrayTree], chex.Array]
