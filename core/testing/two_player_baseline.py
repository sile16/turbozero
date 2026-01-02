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

        # Overall stats (min/avg/max for WandB grouping)
        overall_avg = player_outcomes.mean()
        overall_min = player_outcomes.min()
        overall_max = player_outcomes.max()

        # Track by who went first
        # p_ids_all[:, 0] is our agent's player ID for each game
        # If player ID is 0, our agent went first
        agent_went_first = p_ids_all[:, 0] == 0

        # First player stats (when agent went first)
        first_mask = agent_went_first
        first_count = jnp.sum(first_mask)
        first_outcomes = jnp.where(first_mask, player_outcomes, 0.0)
        first_avg = jnp.where(first_count > 0, jnp.sum(first_outcomes) / first_count, 0.0)
        first_min = jnp.where(first_count > 0, jnp.min(jnp.where(first_mask, player_outcomes, 1.0)), 0.0)
        first_max = jnp.where(first_count > 0, jnp.max(jnp.where(first_mask, player_outcomes, -1.0)), 0.0)

        # Second player stats (when agent went second)
        second_mask = ~agent_went_first
        second_count = jnp.sum(second_mask)
        second_outcomes = jnp.where(second_mask, player_outcomes, 0.0)
        second_avg = jnp.where(second_count > 0, jnp.sum(second_outcomes) / second_count, 0.0)
        second_min = jnp.where(second_count > 0, jnp.min(jnp.where(second_mask, player_outcomes, 1.0)), 0.0)
        second_max = jnp.where(second_count > 0, jnp.max(jnp.where(second_mask, player_outcomes, -1.0)), 0.0)

        # Metrics use slash notation for WandB grouping (min/avg/max on same graph)
        metrics = {
            # Overall reward (all games)
            f"{self.name}_reward/min": overall_min,
            f"{self.name}_reward/avg": overall_avg,
            f"{self.name}_reward/max": overall_max,
            # First player reward (when agent went first)
            f"{self.name}_first_reward/min": first_min,
            f"{self.name}_first_reward/avg": first_avg,
            f"{self.name}_first_reward/max": first_max,
            f"{self.name}_first_reward/count": first_count,
            # Second player reward (when agent went second)
            f"{self.name}_second_reward/min": second_min,
            f"{self.name}_second_reward/avg": second_avg,
            f"{self.name}_second_reward/max": second_max,
            f"{self.name}_second_reward/count": second_count,
        }

        return state, metrics, frames, p_ids
