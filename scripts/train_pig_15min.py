"""Train a Pig agent for 15 minutes using UnifiedMCTS.

Validates against:
- Random player
- Perfect (optimal) player

Tracks first-player advantage in evaluations.

Usage:
    poetry run python scripts/train_pig_15min.py
"""

import os
import time
import pickle
import jax
import jax.numpy as jnp
import optax
import pgx
import flax.linen as nn

from core.evaluators.mcts import UnifiedMCTS, PUCTSelector
from core.evaluators.mcts.unified_mcts import linear_temp_schedule
from core.evaluators.nn_evaluator import RandomEvaluator
from core.evaluators.pig_optimal import PigPerfectPlayEvaluator
from core.memory.replay_memory import EpisodeReplayBuffer
from core.testing.two_player_baseline import TwoPlayerBaseline
from core.training.train import Trainer
from core.training.loss_fns import az_default_loss_fn
from core.types import StepMetadata


# ============================================================================
# Configuration
# ============================================================================
TRAINING_TIME_MINUTES = 30
SEED = 42

# Environment
ENV_NAME = "pig"

# Network
HIDDEN_SIZES = [64, 64]

# MCTS (training)
TRAIN_MCTS_ITERATIONS = 25
TRAIN_MCTS_MAX_NODES = 100
TRAIN_GUMBEL_K = 2  # Only 2 actions in Pig

# MCTS (evaluation) - smaller for faster testing
EVAL_MCTS_ITERATIONS = 20
EVAL_MCTS_MAX_NODES = 200

# Training
BATCH_SIZE = 32  # Number of parallel games for self-play
TRAIN_BATCH_SIZE = 64  # Batch size for gradient updates
TRAIN_STEPS_PER_EPOCH = 10
COLLECTION_STEPS_PER_EPOCH = 50
WARMUP_STEPS = 100
BUFFER_SIZE = 10000
MAX_EPISODE_STEPS = 200  # Pig games can be long

# Evaluation
EVAL_EPISODES = 128  # More games for stable metrics in stochastic game
EVAL_EVERY = 5  # Evaluate every N epochs

# Checkpointing
CHECKPOINT_DIR = "./checkpoints/pig"
WANDB_PROJECT = "turbozero-pig"

# ============================================================================
# Pig Step Functions for UnifiedMCTS
# ============================================================================

def make_pig_decision_step_fn(env):
    """Create decision step function for Pig.

    After a Roll action (0), the state becomes stochastic (waiting for die).
    Hold action (1) banks points and switches player.
    """
    def step_fn(state, action, key):
        new_state = env.step_deterministic(state, action)
        step_count = getattr(new_state, '_step_count', 0)

        # Roll action leads to stochastic state (unless already terminated)
        is_stochastic = jnp.logical_and(action == 0, ~new_state.terminated)

        metadata = StepMetadata(
            rewards=new_state.rewards,
            action_mask=new_state.legal_action_mask,
            terminated=new_state.terminated,
            cur_player_id=new_state.current_player,
            step=step_count,
            is_stochastic=jnp.asarray(is_stochastic, dtype=jnp.bool_),
        )
        return new_state, metadata
    return step_fn


def make_pig_stochastic_step_fn(env):
    """Create stochastic step function for Pig (die roll resolution)."""
    def step_fn(state, outcome, key):
        new_state = env.stochastic_step(state, outcome)
        step_count = getattr(new_state, '_step_count', 0)
        is_stoch = getattr(new_state, '_is_stochastic', False)

        metadata = StepMetadata(
            rewards=new_state.rewards,
            action_mask=new_state.legal_action_mask,
            terminated=new_state.terminated,
            cur_player_id=new_state.current_player,
            step=step_count,
            is_stochastic=jnp.asarray(is_stoch, dtype=jnp.bool_),
        )
        return new_state, metadata
    return step_fn


def make_pig_env_step_fn(env):
    """Create environment step function that resolves stochasticity automatically."""
    def step_fn(state, action, key):
        new_state = env.step(state, action, key)
        step_count = getattr(new_state, '_step_count', 0)
        is_stoch = getattr(new_state, '_is_stochastic', False)

        metadata = StepMetadata(
            rewards=new_state.rewards,
            action_mask=new_state.legal_action_mask,
            terminated=new_state.terminated,
            cur_player_id=new_state.current_player,
            step=step_count,
            is_stochastic=jnp.asarray(is_stoch, dtype=jnp.bool_),
        )
        return new_state, metadata
    return step_fn


def make_pig_env_init_fn(env):
    """Create environment initialization function."""
    def init_fn(key):
        state = env.init(key)
        metadata = StepMetadata(
            rewards=state.rewards,
            action_mask=state.legal_action_mask,
            terminated=state.terminated,
            cur_player_id=state.current_player,
            step=0,
            is_stochastic=jnp.array(False, dtype=jnp.bool_),
        )
        return state, metadata
    return init_fn


# ============================================================================
# Neural Network
# ============================================================================

class PigMLP(nn.Module):
    """Simple MLP for Pig with constant feature to avoid zero-input issues."""
    hidden_sizes: tuple
    num_actions: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Add constant feature to break symmetry for zero inputs
        batch_size = x.shape[0]
        ones = jnp.ones((batch_size, 1))
        x = jnp.concatenate([x, ones], axis=-1)

        # Hidden layers with GELU
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            x = nn.gelu(x)

        # Policy head
        policy_logits = nn.Dense(self.num_actions)(x)

        # Value head (scalar)
        value = nn.Dense(1)(x)
        value = jnp.tanh(value)  # Value in [-1, 1]
        value = jnp.squeeze(value, axis=-1)

        return policy_logits, value


def make_nn_eval_fn(network, state_to_nn_input_fn):
    """Create NN evaluation function for MCTS."""
    def eval_fn(state, params, key):
        obs = state_to_nn_input_fn(state)
        # Add batch dimension
        obs_batch = jnp.expand_dims(obs, 0)
        policy_logits, value = network.apply(params, obs_batch, train=False)
        return policy_logits[0], value[0]
    return eval_fn


def state_to_nn_input(state):
    """Convert Pig state to NN input."""
    return state.observation


# ============================================================================
# Main Training
# ============================================================================

def main():
    print("=" * 60)
    print("Pig Training with UnifiedMCTS")
    print("=" * 60)

    start_time = time.time()
    max_training_time = TRAINING_TIME_MINUTES * 60

    # Create environment
    print(f"\nCreating {ENV_NAME} environment...")
    env = pgx.make(ENV_NAME)
    print(f"  num_actions: {env.num_actions}")
    print(f"  stochastic_action_probs: {env.stochastic_action_probs}")

    # Create network
    print("\nCreating neural network...")
    network = PigMLP(
        hidden_sizes=tuple(HIDDEN_SIZES),
        num_actions=env.num_actions,
    )

    # Initialize network params
    key = jax.random.PRNGKey(SEED)
    key, init_key = jax.random.split(key)
    dummy_state = env.init(init_key)
    dummy_obs = jnp.expand_dims(dummy_state.observation, 0)
    params = network.init(init_key, dummy_obs)
    print(f"  params shape: {jax.tree.map(lambda x: x.shape, params)}")

    # Create evaluation function for MCTS
    nn_eval_fn = make_nn_eval_fn(network, state_to_nn_input)

    # Create step functions
    decision_step_fn = make_pig_decision_step_fn(env)
    stochastic_step_fn = make_pig_stochastic_step_fn(env)
    env_step_fn = make_pig_env_step_fn(env)
    env_init_fn = make_pig_env_init_fn(env)

    # Create MCTS evaluator for training
    print("\nCreating MCTS evaluators...")
    mcts_train = UnifiedMCTS(
        eval_fn=nn_eval_fn,
        action_selector=PUCTSelector(),
        policy_size=env.num_actions,
        max_nodes=TRAIN_MCTS_MAX_NODES,
        num_iterations=TRAIN_MCTS_ITERATIONS,
        gumbel_k=TRAIN_GUMBEL_K,
        decision_step_fn=decision_step_fn,
        stochastic_step_fn=stochastic_step_fn,
        stochastic_action_probs=jnp.array(env.stochastic_action_probs),
        temperature=1.0,  # Will be set by trainer via temp_func
    )

    # Create MCTS evaluator for testing (fewer iterations for speed)
    mcts_test = UnifiedMCTS(
        eval_fn=nn_eval_fn,
        action_selector=PUCTSelector(),
        policy_size=env.num_actions,
        max_nodes=EVAL_MCTS_MAX_NODES,
        num_iterations=EVAL_MCTS_ITERATIONS,
        gumbel_k=TRAIN_GUMBEL_K,
        decision_step_fn=decision_step_fn,
        stochastic_step_fn=stochastic_step_fn,
        stochastic_action_probs=jnp.array(env.stochastic_action_probs),
        temperature=0.0,  # Greedy for testing
    )

    # Create testers
    print("\nCreating testers...")

    # Random player baseline
    random_evaluator = RandomEvaluator(policy_size=env.num_actions)
    random_tester = TwoPlayerBaseline(
        num_episodes=EVAL_EPISODES,
        baseline_evaluator=random_evaluator,
        baseline_params=None,
        name="vs_random",
        epochs_per_test=EVAL_EVERY,
    )

    # Perfect (optimal) player baseline
    print("  Loading optimal Pig strategy...")
    perfect_evaluator = PigPerfectPlayEvaluator()
    perfect_tester = TwoPlayerBaseline(
        num_episodes=EVAL_EPISODES,
        baseline_evaluator=perfect_evaluator,
        baseline_params=None,
        name="vs_perfect",
        epochs_per_test=EVAL_EVERY,
    )

    # Create replay buffer
    print("\nCreating replay buffer...")
    replay_buffer = EpisodeReplayBuffer(
        capacity=BUFFER_SIZE,
        discount_factor=0.99,
        reward_scale=1.0,  # Pig rewards are already in [-1, 1]
    )

    # Create trainer
    print("\nCreating trainer...")
    trainer = Trainer(
        evaluator=mcts_train,
        evaluator_test=mcts_test,
        nn=network,
        loss_fn=az_default_loss_fn,
        optimizer=optax.adam(1e-3),
        memory_buffer=replay_buffer,
        batch_size=BATCH_SIZE,
        train_batch_size=TRAIN_BATCH_SIZE,
        train_steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
        collection_steps_per_epoch=COLLECTION_STEPS_PER_EPOCH,
        warmup_steps=WARMUP_STEPS,
        max_episode_steps=MAX_EPISODE_STEPS,
        env_step_fn=env_step_fn,
        env_init_fn=env_init_fn,
        state_to_nn_input_fn=state_to_nn_input,
        testers=[random_tester, perfect_tester],
        ckpt_dir=CHECKPOINT_DIR,
        wandb_project_name=WANDB_PROJECT,
    )

    # Set temperature schedule (linear decay from 1.0 to 0.1)
    def temp_schedule(step):
        # Rough estimate of steps over 15 minutes
        total_steps = max_training_time * 10  # Conservative estimate
        progress = min(step / total_steps, 1.0)
        return 1.0 - 0.9 * progress  # 1.0 -> 0.1

    trainer.set_temp_fn(temp_schedule)

    # Calculate approximate number of epochs
    # This is a rough estimate - we'll break out early based on time
    estimated_epochs_per_minute = 2  # Conservative
    num_epochs = int(TRAINING_TIME_MINUTES * estimated_epochs_per_minute * 2)
    print(f"\nStarting training (target: {TRAINING_TIME_MINUTES} min, ~{num_epochs} epochs)...")
    print("=" * 60)

    # Run training
    try:
        result = trainer.train_loop(
            seed=SEED,
            num_epochs=num_epochs,
            eval_every=EVAL_EVERY,
        )

        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        elapsed = time.time() - start_time
        print(f"Ran for {elapsed/60:.1f} minutes")

    # Save final model
    print("\nSaving model...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model_path = os.path.join(CHECKPOINT_DIR, "pig_model_final.pkl")

    # Extract params from trainer result
    if 'result' in dir() and result is not None:
        final_params = jax.tree.map(lambda x: x[0], result.train_state.params)
        with open(model_path, 'wb') as f:
            pickle.dump({'params': final_params}, f)
        print(f"Model saved to: {model_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
