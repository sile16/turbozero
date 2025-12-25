"""Single-player game tester for evaluating agents on games like 2048."""

from functools import partial
from typing import Callable, Dict, Optional, Tuple

import chex
from chex import dataclass
import jax
import jax.numpy as jnp

from core.common import step_env_and_evaluator
from core.evaluators.evaluator import Evaluator
from core.testing.tester import BaseTester, TestState
from core.types import EnvInitFn, EnvStepFn


@dataclass(frozen=True)
class SinglePlayerTestState(TestState):
    """State for single-player tester."""
    pass


class SinglePlayerTester(BaseTester):
    """Tester for single-player games (e.g., 2048, Tetris).

    Runs evaluation episodes and reports:
    - Average total reward
    - Average episode length
    - Max reward achieved
    """

    def __init__(self,
                 num_episodes: int = 16,
                 epochs_per_test: int = 1,
                 render_fn: Optional[Callable] = None,
                 render_dir: str = '/tmp/turbozero/',
                 name: str = 'single_player'):
        """
        Args:
            num_episodes: Number of evaluation episodes to run
            epochs_per_test: Run evaluation every N epochs
            render_fn: Optional rendering function
            render_dir: Directory for rendered outputs
            name: Name for logging
        """
        super().__init__(
            num_keys=num_episodes,
            epochs_per_test=epochs_per_test,
            render_fn=render_fn,
            render_dir=render_dir,
            name=name
        )
        self.num_episodes = num_episodes

    def init(self, **kwargs) -> SinglePlayerTestState:
        return SinglePlayerTestState()

    def check_size_compatibilities(self, num_devices: int) -> None:
        if self.num_episodes % num_devices != 0:
            raise ValueError(
                f"num_episodes ({self.num_episodes}) must be divisible by "
                f"num_devices ({num_devices})"
            )

    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0, 1, 2, 3, 4))
    def test(self, max_steps: int, env_step_fn: EnvStepFn, env_init_fn: EnvInitFn,
             evaluator: Evaluator, keys: chex.PRNGKey, state: SinglePlayerTestState,
             params: chex.ArrayTree) -> Tuple[SinglePlayerTestState, Dict, chex.ArrayTree, chex.Array]:
        """Run evaluation episodes.

        Returns:
            - Updated state
            - Metrics dict with avg_reward, avg_length, max_reward
            - Frames for rendering (first episode)
            - Player IDs (zeros for single player)
        """
        # Create step function
        step_fn = partial(
            step_env_and_evaluator,
            evaluator=evaluator,
            env_step_fn=env_step_fn,
            env_init_fn=env_init_fn,
            max_steps=max_steps
        )

        def run_episode(key):
            """Run a single episode and return (total_reward, length)."""
            # Initialize
            init_key, step_key = jax.random.split(key)
            env_state, metadata = env_init_fn(init_key)
            eval_state = evaluator.init(template_embedding=env_state)

            # Episode loop
            def step_body(carry, _):
                env_state, eval_state, metadata, total_reward, length, done, key = carry

                key, step_key = jax.random.split(key)
                eval_output, new_env_state, new_metadata, terminated, truncated, rewards = step_fn(
                    key=step_key,
                    env_state=env_state,
                    env_state_metadata=metadata,
                    eval_state=eval_state,
                    params=params
                )

                # Accumulate reward (player 0 for single-player)
                step_reward = jnp.where(done, 0.0, rewards[0])
                new_total_reward = total_reward + step_reward
                new_length = jnp.where(done, length, length + 1)
                new_done = done | terminated | truncated

                return (new_env_state, eval_output.eval_state, new_metadata,
                        new_total_reward, new_length, new_done, key), env_state

            # Run episode
            init_carry = (env_state, eval_state, metadata, 0.0, 0, False, step_key)
            (_, _, _, total_reward, length, _, _), frames = jax.lax.scan(
                step_body, init_carry, None, length=max_steps
            )

            return total_reward, length, frames

        # Run all episodes (vmapped across keys on this device)
        rewards, lengths, all_frames = jax.vmap(run_episode)(keys)

        # Compute metrics
        # Use _avg_outcome suffix so it shows in console output
        metrics = {
            f'{self.name}_avg_outcome': jnp.mean(rewards),
            f'{self.name}_max_outcome': jnp.max(rewards),
            f'{self.name}_avg_length': jnp.mean(lengths),
        }

        # Return first episode's frames for rendering
        first_frames = jax.tree.map(lambda x: x[0], all_frames)
        player_ids = jnp.zeros(max_steps, dtype=jnp.int32)

        return state, metrics, first_frames, player_ids
