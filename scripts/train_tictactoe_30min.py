"""Train a TicTacToe agent for 30 minutes using UnifiedMCTS.

TicTacToe is a deterministic, solved game - optimal play always results in a draw.
This script validates the training pipeline by checking:
- NN should learn to never lose against random player
- NN should learn to draw against optimal play (or near-optimal MCTS)

Usage:
    poetry run python scripts/train_tictactoe_30min.py
"""

import os
import time
import pickle
import jax
import jax.numpy as jnp
import optax
import pgx

from core.evaluators.mcts import UnifiedMCTS, PUCTSelector
from core.evaluators.nn_evaluator import RandomEvaluator
from core.memory.replay_memory import EpisodeReplayBuffer
from core.networks.azresnet import AZResnet, AZResnetConfig
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
ENV_NAME = "tic_tac_toe"

# Network - medium size for balance of quality and speed
NUM_BLOCKS = 4
NUM_CHANNELS = 64

# MCTS (training) - moderate iterations for quality/speed balance
TRAIN_MCTS_ITERATIONS = 100
TRAIN_MCTS_MAX_NODES = 200
TRAIN_GUMBEL_K = 9  # 9 possible actions in TicTacToe

# MCTS (evaluation) - same as training for consistency
EVAL_MCTS_ITERATIONS = 100
EVAL_MCTS_MAX_NODES = 200

# Training
BATCH_SIZE = 128       # More parallel games for faster data collection
TRAIN_BATCH_SIZE = 256  # Batch size for gradient updates
TRAIN_STEPS_PER_EPOCH = 16
COLLECTION_STEPS_PER_EPOCH = 32
WARMUP_STEPS = 64
BUFFER_SIZE = 30000    # Good buffer size
MAX_EPISODE_STEPS = 10  # TicTacToe max 9 moves

# Evaluation
EVAL_EPISODES = 256    # More evaluation games for reliable metrics
EVAL_EVERY = 10

# Checkpointing
CHECKPOINT_DIR = "./checkpoints/tictactoe"
WANDB_PROJECT = "turbozero-tictactoe"


# ============================================================================
# TicTacToe Step Functions for UnifiedMCTS
# ============================================================================

def make_tictactoe_decision_step_fn(env):
    """Create decision step function for TicTacToe (deterministic game)."""
    def step_fn(state, action, key):
        action = jnp.asarray(action, dtype=jnp.int32)
        new_state = env.step(state, action)
        step_count = new_state._step_count

        metadata = StepMetadata(
            rewards=new_state.rewards,
            action_mask=new_state.legal_action_mask,
            terminated=new_state.terminated,
            cur_player_id=new_state.current_player,
            step=step_count,
            is_stochastic=jnp.array(False, dtype=jnp.bool_),  # Deterministic game
        )
        return new_state, metadata
    return step_fn


def make_tictactoe_env_init_fn(env):
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

def make_nn_eval_fn(network, state_to_nn_input_fn):
    """Create NN evaluation function for MCTS."""
    def eval_fn(state, params, key):
        obs = state_to_nn_input_fn(state)
        obs_batch = jnp.expand_dims(obs, 0)
        policy_logits, value = network.apply(params, obs_batch, train=False)
        return policy_logits[0], value[0]
    return eval_fn


def state_to_nn_input(state):
    """Convert TicTacToe state to NN input."""
    return state.observation


# ============================================================================
# Main Training
# ============================================================================

def main():
    print("=" * 60)
    print("TicTacToe Training with UnifiedMCTS")
    print("=" * 60)

    start_time = time.time()
    max_training_time = TRAINING_TIME_MINUTES * 60

    # Create environment
    print(f"\nCreating {ENV_NAME} environment...")
    env = pgx.make(ENV_NAME)
    print(f"  num_actions: {env.num_actions}")
    print(f"  observation shape: {env.init(jax.random.PRNGKey(0)).observation.shape}")

    # Create network - AZResnet for board games
    print("\nCreating neural network...")
    network = AZResnet(AZResnetConfig(
        policy_head_out_size=env.num_actions,
        num_blocks=NUM_BLOCKS,
        num_channels=NUM_CHANNELS,
        value_head_type="default",  # [-1, 1] for win/loss
    ))

    # Initialize network params
    key = jax.random.PRNGKey(SEED)
    key, init_key = jax.random.split(key)
    dummy_state = env.init(init_key)
    dummy_obs = jnp.expand_dims(dummy_state.observation, 0)
    params = network.init(init_key, dummy_obs, train=False)
    print(f"  params count: {sum(x.size for x in jax.tree.leaves(params))}")

    # Create evaluation function for MCTS
    nn_eval_fn = make_nn_eval_fn(network, state_to_nn_input)

    # Create step functions (TicTacToe is deterministic - no stochastic step)
    decision_step_fn = make_tictactoe_decision_step_fn(env)
    env_step_fn = decision_step_fn  # Same for deterministic games
    env_init_fn = make_tictactoe_env_init_fn(env)

    # Calculate approximate number of epochs (needed for LR schedule)
    estimated_epochs_per_minute = 4
    num_epochs = int(TRAINING_TIME_MINUTES * estimated_epochs_per_minute)

    # Create MCTS evaluator for training
    # TicTacToe is deterministic - no stochastic probs needed
    # UnifiedMCTS handles JAX tracing internally by providing dummy arrays
    print("\nCreating MCTS evaluators...")

    # Gumbel policy improvement hyperparameters for low-simulation regime
    # Paper uses c_visit=50, c_scale=1 for Go/Chess with thousands of sims.
    # With 100 iterations and 9 actions, each action gets ~11 visits.
    # σ(Q) = c_scale * Q / (c_visit + max_N)
    #
    # For TicTacToe with max_visits ≈ 47:
    #   c_visit=50, c_scale=1: σ(Q) ≈ 0.01 * Q (too weak)
    #   c_visit=5, c_scale=5:  σ(Q) ≈ 0.10 * Q (moderate, stable)
    #   c_visit=1, c_scale=20: σ(Q) ≈ 0.42 * Q (too strong, noisy targets)
    #
    # Moderate scaling balances Q-value signal with stability.
    C_VISIT = 5.0   # Moderate offset for stability
    C_SCALE = 5.0   # Moderate scaling for Q-value signal

    mcts_train = UnifiedMCTS(
        eval_fn=nn_eval_fn,
        action_selector=PUCTSelector(),
        policy_size=env.num_actions,
        max_nodes=TRAIN_MCTS_MAX_NODES,
        num_iterations=TRAIN_MCTS_ITERATIONS,
        gumbel_k=min(TRAIN_GUMBEL_K, env.num_actions),
        decision_step_fn=decision_step_fn,
        # No stochastic_step_fn or stochastic_action_probs for deterministic games
        temperature=1.0,
        c_visit=C_VISIT,
        c_scale=C_SCALE,
    )

    # Create MCTS evaluator for testing
    mcts_test = UnifiedMCTS(
        eval_fn=nn_eval_fn,
        action_selector=PUCTSelector(),
        policy_size=env.num_actions,
        max_nodes=EVAL_MCTS_MAX_NODES,
        num_iterations=EVAL_MCTS_ITERATIONS,
        gumbel_k=min(TRAIN_GUMBEL_K, env.num_actions),
        decision_step_fn=decision_step_fn,
        temperature=0.0,  # Greedy for testing
        c_visit=C_VISIT,
        c_scale=C_SCALE,
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

    # Create replay buffer
    print("\nCreating replay buffer...")
    replay_buffer = EpisodeReplayBuffer(
        capacity=BUFFER_SIZE,
        discount_factor=1.0,  # No discounting for short games
        reward_scale=1.0,     # Rewards are already in [-1, 1]
    )

    # Create trainer
    print("\nCreating trainer...")

    # Learning rate schedule: warmup + cosine decay
    lr_warmup_steps = 500
    total_train_steps = num_epochs * TRAIN_STEPS_PER_EPOCH
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=1e-3,
        warmup_steps=lr_warmup_steps,
        decay_steps=total_train_steps,
        end_value=1e-5,
    )

    # Optimizer: Adam with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=lr_schedule),
    )

    trainer = Trainer(
        evaluator=mcts_train,
        evaluator_test=mcts_test,
        nn=network,
        loss_fn=az_default_loss_fn,
        optimizer=optimizer,
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
        testers=[random_tester],
        ckpt_dir=CHECKPOINT_DIR,
        wandb_project_name=WANDB_PROJECT,
    )

    # Temperature schedule (linear decay from 1.0 to 0.1)
    def temp_schedule(step):
        total_steps = max_training_time * 10
        progress = min(step / total_steps, 1.0)
        return 1.0 - 0.9 * progress

    trainer.set_temp_fn(temp_schedule)

    print(f"\nStarting training (target: {TRAINING_TIME_MINUTES} min, ~{num_epochs} epochs)...")
    print("Expected: NN should learn to never lose against random (win rate ~0.95+)")
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
        result = None

    # Save final model
    print("\nSaving model...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model_path = os.path.join(CHECKPOINT_DIR, "tictactoe_model_final.pkl")

    if result is not None:
        final_params = jax.tree.map(lambda x: x[0], result.train_state.params)
        with open(model_path, 'wb') as f:
            pickle.dump({'params': final_params}, f)
        print(f"Model saved to: {model_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
