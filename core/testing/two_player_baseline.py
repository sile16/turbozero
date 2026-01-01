"""Two-player baseline tester for evaluating agents against fixed opponents."""

from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from core.common import GameFrame, two_player_game
from core.evaluators.evaluator import Evaluator
from core.testing.tester import BaseTester, TestState
from core.types import EnvInitFn, EnvStepFn


class TwoPlayerBaseline(BaseTester):
    """Evaluates an agent against a baseline evaluator in a two-player game.

    Tracks win/loss/draw rates overall and by who went first.
    """

    def __init__(self, num_episodes: int, baseline_evaluator: Evaluator, baseline_params: Optional[chex.ArrayTree] = None,
                 *args, **kwargs):
        """
        Args:
            num_episodes: Number of episodes to evaluate against the baseline.
            baseline_evaluator: The baseline evaluator to evaluate against.
            baseline_params: (optional) The parameters of the baseline evaluator.
        """
        super().__init__(num_keys=num_episodes, *args, **kwargs)
        self.num_episodes = num_episodes
        self.baseline_evaluator = baseline_evaluator
        if baseline_params is None:
            baseline_params = jnp.array([])
        self.baseline_params = baseline_params

    def check_size_compatibilities(self, num_devices: int) -> None:
        """Check if tester configuration is compatible with number of devices."""
        if self.num_episodes % num_devices != 0:
            raise ValueError(f"{self.__class__.__name__}: number of episodes ({self.num_episodes}) must be divisible by number of devices ({num_devices})")

    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0, 1, 2, 3, 4))
    def test(self, max_steps: int, env_step_fn: EnvStepFn, env_init_fn: EnvInitFn, evaluator: Evaluator,
        keys: chex.PRNGKey, state: TestState, params: chex.ArrayTree) -> Tuple[TestState, Dict, GameFrame, chex.Array]:
        """Test the agent against the baseline evaluator in a two-player game.

        Args:
            max_steps: Maximum number of steps per episode.
            env_step_fn: Environment step function.
            env_init_fn: Environment initialization function.
            evaluator: The agent evaluator.
            keys: RNG keys.
            state: Internal state of the tester.
            params: NN parameters used by agent.

        Returns:
            Tuple of (state, metrics, frames, player_ids).
        """
        game_fn = partial(two_player_game,
            evaluator_1=evaluator,
            evaluator_2=self.baseline_evaluator,
            params_1=params,
            params_2=self.baseline_params,
            env_step_fn=env_step_fn,
            env_init_fn=env_init_fn,
            max_steps=max_steps
        )

        results, frames, p_ids_all = jax.vmap(game_fn)(keys)
        frames = jax.tree.map(lambda x: x[0], frames)
        p_ids = p_ids_all[0]

        # results[:, 0] contains outcome for our agent (evaluator_1)
        # Outcome is typically: 1 = win, 0 = draw, -1 = loss
        player_outcomes = results[:, 0]
        num_games = player_outcomes.shape[0]

        # Overall stats
        avg = player_outcomes.mean()
        wins = jnp.sum(player_outcomes > 0.5)
        losses = jnp.sum(player_outcomes < -0.5)
        draws = num_games - wins - losses

        # Track by who went first
        # p_ids_all[:, 0] is our agent's player ID for each game
        # If player ID is 0, our agent went first
        agent_went_first = p_ids_all[:, 0] == 0

        # Wins/losses when agent went first
        first_mask = agent_went_first
        first_count = jnp.sum(first_mask)
        first_wins = jnp.sum(jnp.where(first_mask, player_outcomes > 0.5, False))
        first_losses = jnp.sum(jnp.where(first_mask, player_outcomes < -0.5, False))

        # Wins/losses when agent went second
        second_mask = ~agent_went_first
        second_count = jnp.sum(second_mask)
        second_wins = jnp.sum(jnp.where(second_mask, player_outcomes > 0.5, False))
        second_losses = jnp.sum(jnp.where(second_mask, player_outcomes < -0.5, False))

        metrics = {
            f"{self.name}_avg_outcome": avg,
            f"{self.name}_win_rate": wins / num_games,
            f"{self.name}_draw_rate": draws / num_games,
            # First player stats (when agent went first)
            f"{self.name}_first_win_rate": jnp.where(first_count > 0, first_wins / first_count, 0.0),
            f"{self.name}_first_count": first_count,
            # Second player stats (when agent went second)
            f"{self.name}_second_win_rate": jnp.where(second_count > 0, second_wins / second_count, 0.0),
            f"{self.name}_second_count": second_count,
        }

        return state, metrics, frames, p_ids
