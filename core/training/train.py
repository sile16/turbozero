from functools import partial
import os
import shutil
import time
from typing import Any, List, Optional, Tuple

import chex
from chex import dataclass
import flax
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
import wandb
# Orbax may be incompatible with newer JAX layouts; guard import so tests still run.
try:  # pragma: no cover
    from flax.training import orbax_utils
    import orbax.checkpoint as ocp
except Exception:  # pragma: no cover
    orbax_utils = None
    ocp = None

from core.common import partition, step_env_and_evaluator
from core.evaluators.evaluator import Evaluator
from core.memory.replay_memory import BaseExperience, EpisodeReplayBuffer, ReplayBufferState
from core.testing.tester import BaseTester, TestState
from core.types import DataTransformFn, EnvInitFn, EnvStepFn, ExtractModelParamsFn, LossFn, StateToNNInputFn, StepMetadata


@dataclass(frozen=True)
class CollectionState:
    """Stores state of self-play episode collection. Persists across generations.
    - `eval_state`: state of the evaluator
    - `env_state`: state of the environment
    - `buffer_state`: state of the replay buffer
    - `metadata`: metadata of the current environment state
    """
    eval_state: chex.ArrayTree
    env_state: chex.ArrayTree
    buffer_state: ReplayBufferState
    metadata: StepMetadata

@dataclass(frozen=True)
class TrainLoopOutput:
    """
    Stores the state of the training loop.
    collection_state is included to access replay memory.
    - `collection_state`: state of self-play episode collection.
    - `train_state`: flax TrainState, holds optimizer state, model params
    - `test_states`: states of testers
    - `cur_epoch`: current epoch num
    """
    collection_state: CollectionState
    train_state: TrainState
    test_states: List[TestState]
    cur_epoch: int


class TrainStateWithBS(TrainState):
    """Custom flax TrainState to handle BatchNorm"""
    batch_stats: chex.ArrayTree
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

def extract_params(state: TrainState) -> chex.ArrayTree:
    """Extracts model parameters from TrainState.
    
    Args:
    - `state`: TrainState containing model parameters

    Returns:
    - (chex.ArrayTree): model parameters
    """
    if hasattr(state, 'batch_stats'):
        return {'params': state.params, 'batch_stats': state.batch_stats}
    return {'params': state.params}


class Trainer:
    """Implements a training loop for AlphaZero.
    Maintains state across self-play game collection, training, and testing."""

    def __init__(self,
        batch_size: int,
        train_batch_size: int,
        warmup_steps: int,
        collection_steps_per_epoch: int,
        train_steps_per_epoch: int,
        nn: flax.linen.Module,
        loss_fn: LossFn,
        optimizer: optax.GradientTransformation,
        evaluator: Evaluator,        
        memory_buffer: EpisodeReplayBuffer,
        max_episode_steps: int,
        env_step_fn: EnvStepFn,
        env_init_fn: EnvInitFn,
        state_to_nn_input_fn: StateToNNInputFn,
        testers: List[BaseTester],
        evaluator_test: Optional[Evaluator] = None,
        data_transform_fns: List[DataTransformFn] = [],
        extract_model_params_fn: Optional[ExtractModelParamsFn] = extract_params,
        wandb_project_name: str = "",
        ckpt_dir: str = "/tmp/turbozero_checkpoints",
        max_checkpoints: int = 2,
        num_devices: Optional[int] = None,
        wandb_run: Optional[Any] = None,
        extra_wandb_config: Optional[dict] = None
    ):
        """
        Args:
        - `batch_size`: batch size for self-play games
        - `train_batch_size`: minibatch size for training steps
        - `warmup_steps`: # of steps (per batch) to collect via self-play prior to entering the training loop. 
            - This is used to populate the replay memory with some initial samples
        - `collection_steps_per_epoch`: # of steps (per batch) to collect via self-play in each epoch
        - `train_steps_per_epoch`: # of training steps to take in each epoch
        - `nn`: flax.linen.Module containing configured neural network
        - `loss_fn`: loss function for training (see core.training.loss_fns)
        - `optimizer`: optax optimizer
        - `evaluator`: the `Evaluator` to use during self-play
        - `memory_buffer`: replay memory buffer class, used to store self-play experiences
        - `max_episode_steps`: maximum number of steps in an episode
        - `env_step_fn`: environment step function (env_state, action) -> (new_env_state, metadata)
        - `env_init_fn`: environment initialization function (key) -> (env_state, metadata)
        - `state_to_nn_input_fn`: function to convert environment state to neural network input
        - `testers`: list of testers to evaluate the agent against (see core.testing.tester)
        - `evaluator_test`: (optional) evaluator to use during testing. If not provided, `evaluator` is used.
        - `data_transform_fns`: (optional) list of data transform functions to apply to self-play experiences (e.g. rotation, reflection, etc.)
        - `extract_model_params_fn`: (optional) function to extract model parameters from TrainState
        - `wandb_project_name`: (optional) name of wandb project to log to
        - `ckpt_dir`: directory to save checkpoints
        - `max_checkpoints`: maximum number of checkpoints to keep
        - `num_devices`: (optional) number of devices to use, defaults to jax.local_device_count()
        - `wandb_run`: (optional) wandb run object, will continue logging to this run if passed, else a new run is initialized
        - `extra_wandb_config`: (optional) extra config to pass to wandb
        """
        self.num_devices = num_devices if num_devices is not None else jax.local_device_count()
        # environment
        self.env_step_fn = env_step_fn
        self.env_init_fn = env_init_fn
        self.max_episode_steps = max_episode_steps
        self.template_env_state = self.make_template_env_state()
        # nn
        self.state_to_nn_input_fn = state_to_nn_input_fn
        self.nn = nn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.extract_model_params_fn = extract_model_params_fn
        # selfplay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.collection_steps_per_epoch = collection_steps_per_epoch
        self.memory_buffer = memory_buffer
        self.evaluator_train = evaluator
        self.transform_fns = data_transform_fns
        self.step_train = partial(step_env_and_evaluator,
            evaluator=self.evaluator_train,
            env_step_fn=self.env_step_fn,
            env_init_fn=self.env_init_fn,
            max_steps=self.max_episode_steps
        )
        # training
        self.train_steps_per_epoch = train_steps_per_epoch
        self.train_batch_size = train_batch_size
        # testing
        self.testers = testers
        self.evaluator_test = evaluator_test if evaluator_test is not None else evaluator
        self.step_test = partial(step_env_and_evaluator,
            evaluator=self.evaluator_test,
            env_step_fn=self.env_step_fn,
            env_init_fn=self.env_init_fn,
            max_steps=self.max_episode_steps
        )
        # checkpoints
        self.ckpt_dir = ckpt_dir
        if ocp is not None and orbax_utils is not None:
            options = ocp.CheckpointManagerOptions(max_to_keep=max_checkpoints, create=True)
            self.checkpoint_manager = ocp.CheckpointManager(
                ocp.test_utils.erase_and_create_empty(ckpt_dir), options=options)
        else:
            self.checkpoint_manager = None
        # wandb
        self.wandb_project_name = wandb_project_name
        self.use_wandb = wandb_project_name != ""
        if self.use_wandb:
            if wandb_run is not None:
                self.run = wandb_run
            else:
                self.run = self.init_wandb(wandb_project_name, extra_wandb_config)
        else:
            self.run = None
        # check batch sizes, etc. are compatible with number of devices
        self.check_size_compatibilities()


    def init_wandb(self, project_name: str, extra_wandb_config: Optional[dict]):
        """Initializes wandb run.
        Args: 
        - `project_name`: name of wandb project
        - `extra_wandb_config`: (optional) extra config to pass to wandb
        
        Returns:
        - (wandb.Run): wandb run
        """
        if extra_wandb_config is None:
            extra_wandb_config = {} 
        return wandb.init(
            project=project_name,
            config={**self.get_config(), **extra_wandb_config}
        )
    

    def check_size_compatibilities(self):
        """Checks if batch sizes, etc. are compatible with number of devices.
        Calls check_size_compatibilities on each tester."""

        err_fmt = "Batch size must be divisible by the number of devices. Got {b} batch size and {d} devices."
        # check train batch size
        if self.train_batch_size % self.num_devices != 0:
            raise ValueError(err_fmt.format(b=self.train_batch_size, d=self.num_devices))
        # check collection batch size
        if self.batch_size % self.num_devices != 0:
            raise ValueError(err_fmt.format(b=self.batch_size, d=self.num_devices))
        # check testers 
        for tester in self.testers:
            tester.check_size_compatibilities(self.num_devices)


    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0,))
    def init_train_state(self, key: jax.random.PRNGKey) -> TrainState:
        """Initializes the training state (params, optimizer, etc.) partitions across devices.
        
        Args:
        - `key`: rng

        Returns:
        - (TrainState): initialized training state
        """
        # get template env state
        sample_env_state = self.make_template_env_state()
        # get sample nn input
        sample_obs = self.state_to_nn_input_fn(sample_env_state)
        # initialize nn parameters
        variables = self.nn.init(key, sample_obs[None, ...], train=False)
        params = variables['params']
        # handle batchnorm
        if 'batch_stats' in variables:
            return TrainStateWithBS.create(
                apply_fn=self.nn.apply,
                params=params,
                tx=self.optimizer,
                batch_stats=variables['batch_stats']
            )
        # init TrrainState
        return TrainState.create(
            apply_fn=self.nn.apply,
            params=params,
            tx=self.optimizer,
        )

        
    def get_config(self):
        """Returns a dictionary of the configuration of the trainer. Used for logging/wand."""
        return {
            'batch_size': self.batch_size,
            'train_batch_size': self.train_batch_size,
            'warmup_steps': self.warmup_steps,
            'collection_steps_per_epoch': self.collection_steps_per_epoch,
            'train_steps_per_epoch': self.train_steps_per_epoch,
            'num_devices': self.num_devices,
            'evaluator_train': self.evaluator_train.__class__.__name__,
            'evaluator_train_config': self.evaluator_train.get_config(),
            'evaluator_test': self.evaluator_test.__class__.__name__,
            'evaluator_test_config': self.evaluator_test.get_config(),
            'memory_buffer': self.memory_buffer.__class__.__name__,
            'memory_buffer_config': self.memory_buffer.get_config(),
        }
    
    
    def collect(self,
        key: jax.random.PRNGKey,
        state: CollectionState,
        params: chex.ArrayTree
    ) -> CollectionState:
        """
        - Collects self-play data for a single step. 
        - Stores experience in replay buffer.
        - Resets environment/evaluator if episode is terminated.
        
        Args:
        - `key`: rng
        - `state`: current collection state (environment, evaluator, replay buffer)
        - `params`: model parameters
        
        Returns:
        - (CollectionState): updated collection state
        """
        # step environment and evaluator
        eval_output, new_env_state, new_metadata, terminated, truncated, rewards = \
            self.step_train(
                key = key,
                env_state = state.env_state,
                env_state_metadata = state.metadata,
                eval_state = state.eval_state,
                params = params
            )
        
        # store experience in replay buffer
        buffer_state = self.memory_buffer.add_experience(
            state = state.buffer_state,
            experience = BaseExperience(
                observation_nn=self.state_to_nn_input_fn(state.env_state),
                policy_mask=state.metadata.action_mask,
                policy_weights=eval_output.policy_weights,
                reward=jnp.empty_like(state.metadata.rewards),
                cur_player_id=state.metadata.cur_player_id
            )
        )
        # apply transforms 
        for transform_fn in self.transform_fns:
            t_policy_mask, t_policy_weights, t_env_state = transform_fn(
                state.metadata.action_mask,
                eval_output.policy_weights,
                state.env_state
            )
            buffer_state = self.memory_buffer.add_experience(
                state = buffer_state,
                experience = BaseExperience(
                    observation_nn=self.state_to_nn_input_fn(t_env_state),
                    policy_mask=t_policy_mask,
                    policy_weights=t_policy_weights,
                    reward=jnp.empty_like(state.metadata.rewards),
                    cur_player_id=state.metadata.cur_player_id
                )
            )
        # assign rewards to buffer if episode is terminated
        buffer_state = jax.lax.cond(
            terminated,
            lambda s: self.memory_buffer.assign_rewards(s, rewards),
            lambda s: s,
            buffer_state
        )
        # truncate episode experiences in buffer if episode is too long
        buffer_state = jax.lax.cond(
            truncated,
            self.memory_buffer.truncate,
            lambda s: s,
            buffer_state
        )
        # return new collection state
        return state.replace(
            eval_state=eval_output.eval_state,
            env_state=new_env_state,
            buffer_state=buffer_state,
            metadata=new_metadata
        )

    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0, 4))
    def collect_steps(self,
        key: chex.PRNGKey,
        state: CollectionState,
        params: chex.ArrayTree,
        num_steps: int
    ) -> CollectionState:
        """Collects self-play data for `num_steps` steps. Mapped across devices.
        
        Args:
        - `key`: rng
        - `state`: current collection state
        - `params`: model parameters
        - `num_steps`: number of self-play steps to collect

        Returns:
        - (CollectionState): updated collection state
        """
        # Handle the case where num_steps=0 by immediately returning the state
        # This prevents trying to split the key into 0 parts
        def do_collect(s):
            collect = partial(self.collect, params=params)
            keys = jax.random.split(key, num_steps)
            return jax.lax.fori_loop(0, num_steps, lambda i, state: collect(keys[i], state), s)
            
        # Use jax.lax.cond instead of if statement for JAX compatibility
        return jax.lax.cond(
            num_steps > 0,
            do_collect,
            lambda s: s,
            state
        )
    

    def _one_train_step_inner(self, ts: TrainState, batch: BaseExperience) -> Tuple[TrainState, chex.Array]:
        """Inner training step logic without pmap decorator.

        Used inside fori_loop where pmap is applied at the outer level.
        Returns metrics as a stacked array for JAX compatibility.

        Args:
        - `ts`: TrainState
        - `batch`: minibatch of experiences

        Returns:
        - (TrainState, Array): updated TrainState and metrics array
        """
        # calculate loss, get gradients
        grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        (loss, (metrics_dict, updates)), grads = grad_fn(ts.params, ts, batch)

        # Compute gradient norm before averaging across devices
        grad_norm = jnp.sqrt(jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree.map(lambda x: jnp.sum(x ** 2), grads)
        ))

        # apply gradients
        grads = jax.lax.pmean(grads, axis_name='d')
        ts = ts.apply_gradients(grads=grads)

        # Compute parameter norm
        param_norm = jnp.sqrt(jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree.map(lambda x: jnp.sum(x ** 2), ts.params)
        ))

        # update batchnorm stats using lax.cond for JAX compatibility
        def update_batch_stats(t):
            return t.replace(batch_stats=jax.lax.pmean(updates.get('batch_stats', {}), axis_name='d'))

        def no_update(t):
            return t

        # Check if batch_stats exists - this is static so okay to use Python if
        if hasattr(ts, 'batch_stats') and 'batch_stats' in updates:
            ts = update_batch_stats(ts)

        # Stack metrics into array for JAX compatibility inside fori_loop
        # Order: loss, policy_loss, value_loss, policy_entropy, policy_accuracy, grad_norm, param_norm
        metrics_array = jnp.array([
            loss,
            metrics_dict.get('policy_loss', 0.0),
            metrics_dict.get('value_loss', 0.0),
            metrics_dict.get('policy_entropy', 0.0),
            metrics_dict.get('policy_accuracy', 0.0),
            grad_norm,
            param_norm,
        ])

        return ts, metrics_array

    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0,))
    def _train_steps_pmap(self,
        batches: BaseExperience,
        train_state: TrainState,
    ) -> Tuple[TrainState, chex.Array]:
        """JIT-compiled training steps using fori_loop.

        This is pmapped and uses fori_loop internally for full JIT compilation.
        Batches are pre-sampled outside pmap for correct buffer access.

        Args:
        - `batches`: pre-sampled batches, shape (num_steps, per_device_batch_size, ...)
        - `train_state`: current training state

        Returns:
        - (TrainState, Array): updated train state and metrics array (num_steps, num_metrics)
        """
        num_steps = jax.tree_util.tree_leaves(batches)[0].shape[0]
        num_metrics = 7  # loss, policy_loss, value_loss, policy_entropy, policy_accuracy, grad_norm, param_norm

        def loop_body(i, carry):
            ts, metrics_acc = carry

            # Get batch for this step
            batch = jax.tree.map(lambda x: x[i], batches)

            # Run training step
            ts, step_metrics = self._one_train_step_inner(ts, batch)

            # Store metrics for this step
            metrics_acc = metrics_acc.at[i].set(step_metrics)

            return ts, metrics_acc

        # Initialize metrics accumulator
        metrics_acc = jnp.zeros((num_steps, num_metrics))

        # Run the training loop
        train_state, all_metrics = jax.lax.fori_loop(
            0, num_steps, loop_body, (train_state, metrics_acc)
        )

        return train_state, all_metrics

    def train_steps(self,
        key: chex.PRNGKey,
        collection_state: CollectionState,
        train_state: TrainState,
        num_steps: int
    ) -> Tuple[CollectionState, TrainState, dict]:
        """Performs `num_steps` training steps using JIT-compiled fori_loop.
        Each step consists of sampling a minibatch from the replay buffer and updating the parameters.

        Args:
        - `key`: rng
        - `collection_state`: current collection state
        - `train_state`: current training state
        - `num_steps`: number of training steps to perform

        Returns:
        - (CollectionState, TrainState, dict):
            - updated collection state
            - updated training state
            - metrics
        """
        if num_steps == 0:
            return collection_state, train_state, {}

        # Get buffer state
        buffer_state = collection_state.buffer_state

        # Pre-sample all batches (sampling happens outside pmap for correct buffer access)
        keys = jax.random.split(key, num_steps)

        # Sample all batches at once using vmap
        all_batches = jax.vmap(
            lambda k: self.memory_buffer.sample(buffer_state, k, self.train_batch_size)
        )(keys)
        # all_batches shape: (num_steps, train_batch_size, ...)

        # Reshape batches for pmap: (num_steps, devices, per_device_batch, ...)
        all_batches = jax.tree.map(
            lambda x: x.reshape((num_steps, self.num_devices, -1) + x.shape[2:]),
            all_batches
        )
        # Transpose to (devices, num_steps, per_device_batch, ...)
        all_batches = jax.tree.map(
            lambda x: jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim))),
            all_batches
        )

        # Run JIT-compiled training loop
        train_state, all_metrics = self._train_steps_pmap(all_batches, train_state)

        # Aggregate metrics across steps (all_metrics shape: (devices, num_steps, num_metrics))
        # Take first device's metrics (they should be same after pmean)
        metrics_array = all_metrics[0]  # (num_steps, num_metrics)

        # Compute mean and std
        mean_metrics = metrics_array.mean(axis=0)
        std_metrics = metrics_array.std(axis=0)

        # Convert back to dict
        metric_names = ['loss', 'policy_loss', 'value_loss', 'policy_entropy', 'policy_accuracy', 'grad_norm', 'param_norm']
        metrics = {}
        for i, name in enumerate(metric_names):
            metrics[name] = mean_metrics[i]
            if name in ['loss', 'policy_loss', 'value_loss', 'grad_norm']:
                metrics[f'{name}_std'] = std_metrics[i]

        # return updated collection state, train state, and metrics
        return collection_state, train_state, metrics
    
    
    def log_metrics(self, metrics: dict, epoch: int, step: Optional[int] = None):
        """Logs metrics to console and wandb.

        Args:
        - `metrics`: dictionary of metrics
        - `epoch`: current epoch
        - `step`: current step
        """
        # Flatten metrics dict if it contains nested dictionaries
        flat_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat_metrics[f"{k}/{sub_k}"] = sub_v
            else:
                flat_metrics[k] = v

        # Extract key metrics for console display (only show important ones)
        console_keys = ['loss', 'policy_loss', 'value_loss', 'policy_accuracy', 'grad_norm']
        console_metrics = {}
        for k, v in flat_metrics.items():
            # Only show important scalar values in console output
            base_key = k.split('/')[-1] if '/' in k else k
            if base_key not in console_keys and not k.endswith('_avg_outcome'):
                continue
            if hasattr(v, 'item') and callable(getattr(v, 'item')):
                console_metrics[k] = f"{v.item():.4f}"
            elif isinstance(v, (int, float)):
                console_metrics[k] = f"{v:.4f}"

        # log to console
        print(f"Epoch {epoch}: {console_metrics}")

        # log to wandb
        if self.use_wandb:
            wandb_metrics = {}

            # Training metrics (loss, policy, value, etc.)
            training_keys = ['loss', 'policy_loss', 'value_loss', 'l2_reg', 'policy_entropy',
                           'policy_accuracy', 'policy_kl', 'value_accuracy', 'value_entropy',
                           'loss_std', 'policy_loss_std', 'value_loss_std',
                           'grad_norm', 'grad_norm_std', 'param_norm']
            for k in training_keys:
                if k in flat_metrics:
                    wandb_metrics[f'training/{k}'] = flat_metrics[k]

            # Copy metrics that already have proper namespacing
            for k, v in flat_metrics.items():
                if k.startswith(('buffer/', 'mcts/', 'game/', 'collect/', 'train/', 'eval/')):
                    wandb_metrics[k] = v
                # Handle tester metrics (e.g., pretrained_avg_outcome -> eval/pretrained_avg_outcome)
                elif k.endswith('_avg_outcome'):
                    wandb_metrics[f'eval/{k}'] = v
                elif k.endswith('_win_rate') or k.endswith('_loss_rate') or k.endswith('_draw_rate'):
                    wandb_metrics[f'eval/{k}'] = v
                # Handle perf/ namespace (legacy, keep for compatibility)
                elif k.startswith('perf/'):
                    wandb_metrics[k] = v

            wandb.log(wandb_metrics, step)


    def save_checkpoint(self, train_state: TrainState, epoch: int) -> None:
        """Saves an orbax checkpoint of the training state.
        
        Args:
        - `train_state`: current training state
        - `epoch`: current epoch
        """
        if ocp is None or orbax_utils is None or self.checkpoint_manager is None:  # pragma: no cover
            return
        # convert pmap-sharded train_state to a single-device one
        ckpt = jax.tree.map(lambda x: jax.device_get(x), train_state)
        # save checkpoint (async)
        self.checkpoint_manager.save(epoch, args=ocp.args.StandardSave(ckpt))


    def load_train_state_from_checkpoint(self, path_to_checkpoint: str, epoch: int) -> TrainState:
        """Loads a training state from a checkpoint.
        
        Args:
        - `path_to_checkpoint`: path to checkpoint
        - `epoch`: epoch to load
        
        Returns:
        - (TrainState): loaded training state
        """
        if ocp is None or orbax_utils is None or self.checkpoint_manager is None:  # pragma: no cover
            raise RuntimeError("Orbax is unavailable with this JAX; cannot load checkpoints.")
        # create dummy TrainState
        key = jax.random.PRNGKey(0)
        init_key, key = jax.random.split(key)
        init_keys = jnp.tile(init_key[None], (self.num_devices, 1))
        dummy_state = self.init_train_state(init_keys)
        # load checkpoint
        train_state = self.checkpoint_manager.restore(
                epoch, 
                items=dummy_state, 
                directory=path_to_checkpoint,
                # allowing for saved checkpoints on different number of jax devices (unsafe)
                restore_kwargs={
                    'strict': False,
                    }
                )
        return train_state
    

    def make_template_env_state(self) -> chex.ArrayTree:
        """Create a template environment state used for initializing data structures 
        that hold environment states to the correct shape.
        
        Returns:
        - (chex.ArrayTree): template environment state
        """
        env_state, _ = self.env_init_fn(jax.random.PRNGKey(0))
        return env_state
    
    
    def make_template_experience(self) -> BaseExperience:
        """Create a template experience used for initializing data structures
        that hold experiences to the correct shape.
        
        Returns:
        - (BaseExperience): template experience
        """
        env_state, metadata = self.env_init_fn(jax.random.PRNGKey(0))
        return BaseExperience(
            observation_nn=self.state_to_nn_input_fn(env_state),
            policy_mask=metadata.action_mask,
            policy_weights=jnp.zeros_like(metadata.action_mask, dtype=jnp.float32),
            reward=jnp.zeros_like(metadata.rewards),
            cur_player_id=metadata.cur_player_id
        )
    

    def init_collection_state(self, key: jax.random.PRNGKey, batch_size: int) -> CollectionState:
        """Initializes the collection state (see CollectionState).
        
        Args:
        - `key`: rng
        - `batch_size`: number of parallel environments

        Returns:
        - (CollectionState): initialized collection state
        """
        # make template experience
        template_experience = self.make_template_experience()
        # init buffer state
        buffer_state = self.memory_buffer.init(batch_size, template_experience)
        # init env state
        env_init_key, key = jax.random.split(key)
        env_keys = jax.random.split(env_init_key, batch_size)
        env_state, metadata = jax.vmap(self.env_init_fn)(env_keys)
        # init evaluator state
        eval_state = self.evaluator_train.init_batched(batch_size, template_embedding=self.template_env_state)
        # return collection state
        return CollectionState(
            eval_state=eval_state,
            env_state=env_state,
            buffer_state=buffer_state,
            metadata=metadata
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
            metrics["train/train_time_sec"] = train_duration
            metrics["train/train_steps_per_sec"] = self.train_steps_per_epoch * self.train_batch_size / jnp.maximum(train_duration, 1e-6)

           # Add performance metrics
            collection_steps = self.batch_size * (cur_epoch+1) * self.collection_steps_per_epoch
            metrics["collect/collect_time_sec"] = collect_duration
            metrics["collect/collect_steps_per_sec"] = self.batch_size * self.collection_steps_per_epoch / max(collect_duration, 1e-6)
            # we don't know how many games, 
            #metrics["perf/collect_game_steps_per_sec"] = (self.collection_steps_per_epoch * self.batch_size) / max(collect_duration, 1e-6)

            
           # Add replay buffer statistics
            buffer_state = collection_state.buffer_state
            populated = jnp.sum(buffer_state.populated)
            trainable_samples = jnp.sum(jnp.logical_and(buffer_state.populated, buffer_state.has_reward))
            total_capacity = buffer_state.populated.size
            metrics["buffer/populated"] = populated
            metrics["buffer/trainable_samples"] = trainable_samples
            metrics["buffer/fullness_pct"] = 100.0 * populated / total_capacity
            metrics["buffer/trainable_pct"] = 100.0 * trainable_samples / total_capacity

            # Add MCTS tree statistics (sample from first device, first batch element)
            if hasattr(self.evaluator_train, 'get_tree_stats'):
                # eval_state is sharded across devices, get first device's first element
                eval_state_sample = jax.tree.map(lambda x: x[0, 0], collection_state.eval_state)
                mcts_stats = self.evaluator_train.get_tree_stats(eval_state_sample)
                metrics.update(mcts_stats)

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
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.wait_until_finished()
                self.save_checkpoint(train_state, cur_epoch)
            # next epoch
            cur_epoch += 1
            
        # make sure last save task has finished
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.wait_until_finished() #
        # return state so that training can be continued!
        return TrainLoopOutput(
            collection_state=collection_state,
            train_state=train_state,
            test_states=tester_states,
            cur_epoch=cur_epoch
        )
