"""
StochasticTrainer is a variant of the Trainer for stochastic environments.
during self-play data collection, by not adding stochastnic nodes to replay buffer.
"""
import jax
import jax.numpy as jnp
import chex
import time
from typing import Optional, Tuple
import wandb

from flax.training.train_state import TrainState
from core.training.train import Trainer, CollectionState, ReplayBufferState
from core.types import BaseExperience


from core.training.train import Trainer, CollectionState, ReplayBufferState
from core.types import BaseExperience

class StochasticTrainer(Trainer):
    """Trainer variant that explicitly handles stochastic environment steps 
    during self-play data collection."""

    def set_temp_fn(self, temp_func: Callable[int, float]) -> None:
        self.temp_func = temp_func


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
    
    def train_loop(self,
        seed: int,
        num_epochs: int,
        eval_every: int = 1,
        initial_state: Optional[TrainLoopOutput] = None
        ) -> Tuple[CollectionState, TrainState]:
        """Runs the training loop for `num_epochs` epochs. Mostly configured by the Trainer's attributes.
        - Collects self-play episdoes across a batch of environments.
        - Trains the neural network on the collected experiences.
        - Tests the agent on a set of Testers, which evaluate the agent's performance.

        Args:
        - `seed`: rng seed (int)
        - `num_epochs`: number of epochs to run the training loop for
        - `eval_every`: number of epochs between evaluations
        - `initial_state`: (optional) TrainLoopOutput, used to continue training from a previous state

        Returns:
        - (TrainLoopOutput): contains train_state, collection_state, test_states, cur_epoch after training loop
        """
        # init rng
        key = jax.random.PRNGKey(seed)

        # initialize states
        if initial_state:
            collection_state = initial_state.collection_state
            train_state = initial_state.train_state
            tester_states = initial_state.test_states
            cur_epoch = initial_state.cur_epoch
        else:
            cur_epoch = 0
            # initialize collection state
            init_key, key = jax.random.split(key)
            collection_state = partition(self.init_collection_state(init_key, self.batch_size), self.num_devices)
            # initialize train state
            init_key, key = jax.random.split(key)
            init_keys = jnp.tile(init_key[None], (self.num_devices, 1))
            train_state = self.init_train_state(init_keys)
            params = self.extract_model_params_fn(train_state)
            # initialize tester states
            tester_states = []
            for tester in self.testers:
                state = jax.pmap(tester.init, axis_name='d')(params=params)
                tester_states.append(state)
        
        # Define collect function before warmup

        collect = jax.vmap(self.collect_steps, in_axes=(1, 1, None, None), out_axes=1)
        params = self.extract_model_params_fn(train_state)
        
        # warmup
        # populate replay buffer with initial self-play games
        # Only do warmup if warmup_steps > 0
        # I think this is why inside of collect_steps, we have a conditional for num_steps > 0
        # However, I think it's probably better to do this in the training loop, 
        # Rather than inside of collect_steps, todo: Performance improvement
        if self.warmup_steps > 0:
            collect_key, key = jax.random.split(key)
            collect_keys = partition(jax.random.split(collect_key, self.batch_size), self.num_devices)
            collection_state = collect(collect_keys, collection_state, params, self.warmup_steps)

        # training loop
        while cur_epoch < num_epochs:
            # Track epoch start time for performance metrics

            # get the temperature
            current_temp = self.temp_func(cur_epoch * self.collection_steps_per_epoch)
            metrics["collect/temperature"] = current_temp
            self.evaluator_train.temperature = current_temp



            epoch_start_time = time.time()
            
            # Collect self-play games
            print("Collecting self-play games")
            collect_start_time = time.time()
            collect_key, key = jax.random.split(key)
            collect_keys = partition(jax.random.split(collect_key, self.batch_size), self.num_devices)
            collection_state = collect(collect_keys, collection_state, params, self.collection_steps_per_epoch)
            collect_duration = time.time() - collect_start_time

            # Train
            print("Training")
            train_start_time = time.time()
            train_key, key = jax.random.split(key)
            collection_state, train_state, metrics = self.train_steps(train_key, collection_state, train_state, self.train_steps_per_epoch)
            print("Training Done")
            train_duration = time.time() - train_start_time
            metrics["perf/train_time_sec"] = train_duration
            metrics["perf/train_steps_per_sec"] = self.train_steps_per_epoch * self.train_batch_size / jnp.maximum(train_duration, 1e-6)
            
           # Add performance metrics
            collection_steps = self.batch_size * (cur_epoch+1) * self.collection_steps_per_epoch
            metrics["perf/collect_time_sec"] = collect_duration
            metrics["perf/collect_steps_per_sec"] = self.collection_steps_per_epoch / max(collect_duration, 1e-6)
            # we don't know how many games, 
            #metrics["perf/collect_game_steps_per_sec"] = (self.collection_steps_per_epoch * self.batch_size) / max(collect_duration, 1e-6)

            
           # Add replay buffer statistics
            buffer_state = collection_state.buffer_state
            populated = jnp.sum(buffer_state.populated)
            metrics["buffer/populated"] = populated
            metrics["buffer/has_reward"] = jnp.sum(buffer_state.has_reward)
            metrics["buffer/fullness_pct"] = 100.0 * populated / buffer_state.populated.size
            
            metrics["perf/epoch_time_sec"] = time.time() - epoch_start_time
            # Log metrics
            self.log_metrics(metrics, cur_epoch, step=collection_steps)
            

            # test 
            
            if cur_epoch % eval_every == 0:
                print("Testing")
                params = self.extract_model_params_fn(train_state)
                
                for i, test_state in enumerate(tester_states):
                    test_start_time = time.time()
                    run_key, key = jax.random.split(key)
                    new_test_state, metrics, rendered = self.testers[i].run(
                        key=run_key, epoch_num=cur_epoch, max_steps=self.max_episode_steps, num_devices=self.num_devices,
                        env_step_fn=self.env_step_fn, env_init_fn=self.env_init_fn, evaluator=self.evaluator_test,
                        state=test_state, params=params)
                        
                    metrics = {k: v.mean() for k, v in metrics.items()}
                    metrics[f"perf/test_{self.testers[i].name}_time_sec"] = time.time() - test_start_time
                    self.log_metrics(metrics, cur_epoch, step=collection_steps)
                    
                    if rendered and self.run is not None:
                        self.run.log({f'{self.testers[i].name}_game': wandb.Video(rendered)}, step=collection_steps)
                    tester_states[i] = new_test_state

                
            # save checkpoint
            # make sure previous save task has finished 
            self.checkpoint_manager.wait_until_finished()
            self.save_checkpoint(train_state, cur_epoch)
            # next epoch
            cur_epoch += 1
            
        # make sure last save task has finished
        self.checkpoint_manager.wait_until_finished() #
        # return state so that training can be continued!
        return TrainLoopOutput(
            collection_state=collection_state,
            train_state=train_state,
            test_states=tester_states,
            cur_epoch=cur_epoch
        )
