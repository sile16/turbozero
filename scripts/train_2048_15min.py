#!/usr/bin/env python3
"""
2048 Training: 15-minute training with UnifiedMCTS
==================================================

Trains 2048 using UnifiedMCTS (Gumbel AlphaZero with stochastic support).
Tracks average score per epoch.

Usage:
    poetry run python scripts/train_2048_15min.py
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import optax
import pgx

from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.unified_mcts import UnifiedMCTS
from core.memory.replay_memory import EpisodeReplayBuffer
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer
from core.types import StepMetadata
from core.testing.single_player_tester import SinglePlayerTester


def create_step_fn(env):
    """Create step function for regular MCTS."""
    def step_fn(state, action, key):
        action = jnp.asarray(action, dtype=jnp.int32)
        new_state = env.step(state, action, key)
        return new_state, StepMetadata(
            rewards=new_state.rewards,
            action_mask=new_state.legal_action_mask,
            terminated=new_state.terminated,
            cur_player_id=new_state.current_player,
            step=new_state._step_count,
            is_stochastic=new_state._is_stochastic
        )
    return step_fn


def create_decision_step_fn(env):
    """Create step function for decision nodes (StochasticMCTS)."""
    def step_fn(state, action, key=None):
        action = jnp.asarray(action, dtype=jnp.int32)
        new_state = env.step_deterministic(state, action)
        return new_state, StepMetadata(
            rewards=new_state.rewards,
            action_mask=new_state.legal_action_mask,
            terminated=new_state.terminated,
            cur_player_id=new_state.current_player,
            step=new_state._step_count,
            is_stochastic=new_state._is_stochastic
        )
    return step_fn


def create_stochastic_step_fn(env):
    """Create step function for chance nodes (StochasticMCTS)."""
    def step_fn(state, outcome, key=None):
        outcome = jnp.asarray(outcome, dtype=jnp.int32)
        new_state = env.step_stochastic(state, outcome)
        return new_state, StepMetadata(
            rewards=new_state.rewards,
            action_mask=new_state.legal_action_mask,
            terminated=new_state.terminated,
            cur_player_id=new_state.current_player,
            step=new_state._step_count,
            is_stochastic=new_state._is_stochastic
        )
    return step_fn


def create_stochastic_aware_step_fn(env):
    """Create unified step function for StochasticMCTS training."""
    def step_fn(state, action, key=None):
        is_stochastic = getattr(state, '_is_stochastic', jnp.array(False))
        action = jnp.asarray(action, dtype=jnp.int32)

        stochastic_state = env.step_stochastic(state, action)
        decision_state = env.step_deterministic(state, action)

        new_state = jax.tree_util.tree_map(
            lambda a, b: jnp.where(is_stochastic, a, b),
            stochastic_state,
            decision_state
        )

        return new_state, StepMetadata(
            rewards=new_state.rewards,
            action_mask=new_state.legal_action_mask,
            terminated=new_state.terminated,
            cur_player_id=new_state.current_player,
            step=new_state._step_count,
            is_stochastic=new_state._is_stochastic
        )
    return step_fn


def create_init_fn(env):
    """Create init function for 2048."""
    def init_fn(key):
        state = env.init(key)
        return state, StepMetadata(
            rewards=state.rewards,
            action_mask=state.legal_action_mask,
            terminated=state.terminated,
            cur_player_id=state.current_player,
            step=state._step_count,
            is_stochastic=getattr(state, '_is_stochastic', jnp.array(False))
        )
    return init_fn


def state_to_nn_input(state):
    return state.observation


def compute_stochastic_probs():
    """Compute static stochastic action probabilities for 2048.
    32 outcomes: position (0-15) × tile_value (2=90%, 4=10%)
    This is used as a template/default - actual probs are computed dynamically.
    """
    probs = jnp.zeros(32)
    for pos in range(16):
        probs = probs.at[pos * 2].set(0.9 / 16)      # tile=2 (90%)
        probs = probs.at[pos * 2 + 1].set(0.1 / 16)  # tile=4 (10%)
    return probs


def compute_dynamic_stochastic_probs(state):
    """Compute stochastic probabilities conditioned on empty cells.

    In 2048, new tiles can only spawn in EMPTY cells.
    Outcome indexing: outcome = pos * 2 + tile_idx
    - tile_idx=0: tile value 2 (90% probability)
    - tile_idx=1: tile value 4 (10% probability)

    This ensures probabilities are correctly weighted by which cells are empty.
    """
    # Get the board - shape (4, 4), values are tile exponents (0 = empty)
    # pgx 2048 uses observation of shape (4, 4, 31) one-hot encoded
    # The raw board values are in state._board or state.observation
    # For simplicity, we check if a cell is empty by looking at the first channel
    # Actually pgx stores board as (4, 4) with log2 values, 0 = empty

    # Access the internal board representation
    board = state._board  # Shape: (4, 4), values are log2(tile) or 0 for empty

    # Flatten board to (16,)
    flat_board = board.flatten()

    # Empty cells have value 0
    empty_mask = (flat_board == 0).astype(jnp.float32)
    num_empty = jnp.sum(empty_mask)

    # Avoid division by zero - if no empty cells, uniform distribution
    num_empty_safe = jnp.maximum(num_empty, 1.0)

    # Probability per empty cell
    prob_per_cell = 1.0 / num_empty_safe

    # Build probability array (32 outcomes)
    probs = jnp.zeros(32)

    # For each position, set probabilities based on whether cell is empty
    for pos in range(16):
        is_empty = empty_mask[pos]
        probs = probs.at[pos * 2].set(is_empty * prob_per_cell * 0.9)      # tile=2 (90%)
        probs = probs.at[pos * 2 + 1].set(is_empty * prob_per_cell * 0.1)  # tile=4 (10%)

    return probs


def run_training(config: dict, num_epochs: int, seed: int = 42):
    """Run training for a fixed number of epochs, tracking scores."""
    print(f"\n{'='*60}")
    print(f"Training 2048 with UnifiedMCTS for {num_epochs} epochs")
    print(f"{'='*60}")

    env = pgx.make('2048')
    init_fn = create_init_fn(env)
    step_fn = create_stochastic_aware_step_fn(env)

    # Neural network
    resnet = AZResnet(AZResnetConfig(
        policy_head_out_size=env.num_actions,
        num_blocks=config['num_blocks'],
        num_channels=config['num_channels'],
    ))

    # Create evaluator with UnifiedMCTS
    stochastic_probs = compute_stochastic_probs()
    decision_step_fn = create_decision_step_fn(env)
    stochastic_step_fn = create_stochastic_step_fn(env)

    az_evaluator = UnifiedMCTS(
        eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
        action_selector=PUCTSelector(),
        stochastic_action_probs=stochastic_probs,  # Template for shape
        stochastic_probs_fn=compute_dynamic_stochastic_probs,  # Dynamic probs based on empty cells
        policy_size=env.num_actions,
        num_iterations=config['mcts_iterations'],
        max_nodes=config['mcts_iterations'] + 100,
        decision_step_fn=decision_step_fn,
        stochastic_step_fn=stochastic_step_fn,
        gumbel_k=4,  # 2048 has only 4 actions
        temperature=1.0,
    )
    az_evaluator_test = UnifiedMCTS(
        eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
        action_selector=PUCTSelector(),
        stochastic_action_probs=stochastic_probs,  # Template for shape
        stochastic_probs_fn=compute_dynamic_stochastic_probs,  # Dynamic probs based on empty cells
        policy_size=env.num_actions,
        num_iterations=config['mcts_iterations_test'],
        max_nodes=config['mcts_iterations_test'] + 50,
        decision_step_fn=decision_step_fn,
        stochastic_step_fn=stochastic_step_fn,
        gumbel_k=4,
        temperature=0.0,  # Greedy for testing
    )

    # Replay buffer
    replay_memory = EpisodeReplayBuffer(capacity=config['buffer_capacity'])

    # Calculate train steps
    train_steps_per_epoch = max(1, int(
        config['batch_size'] * config['collection_steps'] / config['train_batch_size']
    ))

    # Evaluation tester
    eval_tester = SinglePlayerTester(
        num_episodes=16,
        epochs_per_test=5,
        name='eval'
    )

    # Trainer with wandb
    trainer = Trainer(
        batch_size=config['batch_size'],
        train_batch_size=config['train_batch_size'],
        warmup_steps=0,
        collection_steps_per_epoch=config['collection_steps'],
        train_steps_per_epoch=train_steps_per_epoch,
        nn=resnet,
        loss_fn=partial(az_default_loss_fn, l2_reg_lambda=config['l2_reg']),
        optimizer=optax.adam(config['lr']),
        evaluator=az_evaluator,
        memory_buffer=replay_memory,
        max_episode_steps=config['max_episode_steps'],
        env_step_fn=step_fn,
        env_init_fn=init_fn,
        state_to_nn_input_fn=state_to_nn_input,
        testers=[eval_tester],
        evaluator_test=az_evaluator_test,
        data_transform_fns=[],
        wandb_project_name='turbozero-2048-15min',
        ckpt_dir='/tmp/turbozero_2048_unified',
        extra_wandb_config={'evaluator_type': 'UnifiedMCTS'},
    )

    # Run training
    start_time = time.time()
    output = trainer.train_loop(seed=seed, num_epochs=num_epochs, eval_every=5)
    elapsed = time.time() - start_time

    print(f"\nUnifiedMCTS training complete: {num_epochs} epochs in {elapsed:.1f}s")
    print(f"Seconds per epoch: {elapsed/num_epochs:.2f}")

    return {
        'evaluator_type': 'UnifiedMCTS',
        'epochs': num_epochs,
        'time': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description='2048 Training with UnifiedMCTS')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Configuration optimized for 2048 training
    # CRITICAL: buffer_capacity must be > max_episode_steps to avoid overwriting
    # during long episodes. 2048 games often exceed 500 steps.
    config = {
        'batch_size': 64,
        'train_batch_size': 256,
        'collection_steps': 32,
        'mcts_iterations': 50,
        'mcts_iterations_test': 50,
        'num_blocks': 4,
        'num_channels': 128,
        'lr': 1e-3,
        'l2_reg': 0.0001,
        'buffer_capacity': 2000,  # CRITICAL: Must be > max_episode_steps
        'max_episode_steps': 1000,
    }

    print("="*60)
    print(f"2048 Training with UnifiedMCTS ({args.epochs} epochs)")
    print("="*60)
    print(f"JAX devices: {jax.devices()}")
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    result = run_training(config, args.epochs, seed=args.seed)

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{result['evaluator_type']}: {result['epochs']} epochs in {result['time']:.1f}s ({result['time']/result['epochs']:.2f}s/epoch)")

    print("\nCheck wandb for detailed scores: https://wandb.ai/")


if __name__ == '__main__':
    main()
