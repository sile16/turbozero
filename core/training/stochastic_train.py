import jax
import jax.numpy as jnp
import chex
from typing import Tuple

from core.training.train import Trainer, CollectionState
from core.types import StepMetadata, BaseExperience
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS

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
        def handle_deterministic_state(buffer_state):
            # Store experience in replay buffer for deterministic states
            updated_buffer = self.memory_buffer.add_experience(
                state=buffer_state,
                experience=BaseExperience(
                    observation_nn=self.state_to_nn_input_fn(state.env_state),
                    policy_mask=state.metadata.action_mask,
                    policy_weights=eval_output.policy_weights,
                    reward=jnp.empty_like(state.metadata.rewards),
                    cur_player_id=state.metadata.cur_player_id
                )
            )
            
            # We can't use a for loop with transforms in JIT compiled code,
            # so we need to handle any transforms separately
            if len(self.transform_fns) > 0:
                # For simplicity in this example, just handle the first transform if any exists
                transform_fn = self.transform_fns[0]
                t_policy_mask, t_policy_weights, t_env_state = transform_fn(
                    state.metadata.action_mask,
                    eval_output.policy_weights,
                    state.env_state
                )
                updated_buffer = self.memory_buffer.add_experience(
                    state=updated_buffer,
                    experience=BaseExperience(
                        observation_nn=self.state_to_nn_input_fn(t_env_state),
                        policy_mask=t_policy_mask,
                        policy_weights=t_policy_weights,
                        reward=jnp.empty_like(state.metadata.rewards),
                        cur_player_id=state.metadata.cur_player_id
                    )
                )
            return updated_buffer
            
        def handle_stochastic_state(buffer_state):
            # For stochastic states, don't add to replay buffer
            return buffer_state
        
        # Use jax.lax.cond to choose between handling deterministic or stochastic state
        # We need to check if the state has is_stochastic attribute and what its value is
        is_stochastic = getattr(state.env_state, 'is_stochastic', jnp.array(False))
        buffer_state = jax.lax.cond(
            is_stochastic,
            handle_stochastic_state,
            handle_deterministic_state,
            state.buffer_state
        )
        
        # Assign rewards to buffer if episode is terminated
        buffer_state = jax.lax.cond(
            terminated,
            lambda s: self.memory_buffer.assign_rewards(s, rewards),
            lambda s: s,
            buffer_state
        )
        
        # Truncate episode experiences in buffer if episode is too long
        buffer_state = jax.lax.cond(
            truncated,
            self.memory_buffer.truncate,
            lambda s: s,
            buffer_state
        )
        
        # Return new collection state
        return state.replace(
            eval_state=eval_output.eval_state,
            env_state=new_env_state,
            buffer_state=buffer_state,
            metadata=new_metadata
        )
