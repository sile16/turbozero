"""
StochasticTrainer is a variant of the Trainer for stochastic environments.
during self-play data collection, by not adding stochastnic nodes to replay buffer.
"""
import jax
import jax.numpy as jnp
import chex


from core.training.train import Trainer, CollectionState, ReplayBufferState
from core.types import BaseExperience

class StochasticTrainer(Trainer):
    """Trainer variant that explicitly handles stochastic environment steps 
    during self-play data collection."""

    def collect(self,
        key: jax.random.PRNGKey,
        state: CollectionState,
        params: chex.ArrayTree
    ) -> CollectionState:
        """
        - Collects self-play data for a single step. 
        - Handles stochastic states by not storing their policy data for training.
        - Stores experience in replay buffer for deterministic states only.
        - Resets environment/evaluator if episode is terminated.
        
        Args:
        - `key`: rng
        - `state`: current collection state (environment, evaluator, replay buffer)
        - `params`: model parameters
        
        Returns:
        - (CollectionState): updated collection state
        """
        # Step environment and evaluator
        eval_output, new_env_state, new_metadata, terminated, truncated, rewards = \
            self.step_train(
                key=key,
                env_state=state.env_state,
                env_state_metadata=state.metadata,
                eval_state=state.eval_state,
                params=params
            )

        # Define functions for handling deterministic vs stochastic states
        def handle_deterministic_state(state: CollectionState) -> ReplayBufferState:
            
            # Store experience in replay buffer for deterministic states
            updated_buffer = self.memory_buffer.add_experience(
                state=state.buffer_state,
                experience=BaseExperience(
                    observation_nn=self.state_to_nn_input_fn(state.env_state),
                    policy_mask=state.metadata.action_mask,
                    policy_weights=eval_output.policy_weights,
                    reward=jnp.empty_like(state.metadata.rewards),
                    cur_player_id=state.metadata.cur_player_id
                )
            )
            return updated_buffer

        def handle_stochastic_state(state: CollectionState) -> ReplayBufferState:
            #jax.debug.print('stochastic step: {step}', step=new_metadata.step)
            # For stochastic states, don't add to replay buffer
            # these states are waiting for the dice roll, 
            # and after the stochastic action, the next state will be added to the buffer
            return state.buffer_state

        # Use jax.lax.cond to choose between handling deterministic or stochastic state
        # We need to check if the state has is_stochastic attribute and what its value is
        is_stochastic = state.env_state.is_stochastic
        buffer_state = jax.lax.cond(
            is_stochastic,
            handle_stochastic_state,
            handle_deterministic_state,
            state
        )

        def terminated_fn(s: ReplayBufferState) -> ReplayBufferState:
            return self.memory_buffer.assign_rewards(s, rewards)
        
        def not_terminated_fn(s: ReplayBufferState) -> ReplayBufferState:
            return s

        # Assign rewards to buffer if episode is terminated
        buffer_state = jax.lax.cond(
            terminated,
            terminated_fn,
            not_terminated_fn,
            buffer_state
        )

        def truncated_fn(s: ReplayBufferState) -> ReplayBufferState:
            return self.memory_buffer.truncate(s)
    
        def not_truncated_fn(s: ReplayBufferState) -> ReplayBufferState:
            return s

        # Truncate episode experiences in buffer if episode is too long
        buffer_state = jax.lax.cond(
            truncated,
            truncated_fn,
            not_truncated_fn,
            buffer_state
        )

        # Return new collection state
        return state.replace(
            eval_state=eval_output.eval_state,
            env_state=new_env_state,
            buffer_state=buffer_state,
            metadata=new_metadata
        )
