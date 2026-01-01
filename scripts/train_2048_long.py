#!/usr/bin/env python3
"""
2048 Long Training Run: Maximize Score over 2-4 Hours
======================================================

Uses UnifiedMCTS (Gumbel AlphaZero with stochastic support).
Tracks max score achieved and saves best model.

Usage:
    poetry run python scripts/train_2048_long.py --hours 2
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


def create_decision_step_fn(env):
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
    probs = jnp.zeros(32)
    for pos in range(16):
        probs = probs.at[pos * 2].set(0.9 / 16)
        probs = probs.at[pos * 2 + 1].set(0.1 / 16)
    return probs


def main():
    parser = argparse.ArgumentParser(description='2048 Long Training')
    parser.add_argument('--hours', type=float, default=2.0, help='Training duration in hours')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    target_seconds = args.hours * 3600

    print("=" * 70)
    print(f"2048 Long Training: StochasticMCTS for {args.hours} hours")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")
    print(f"Target duration: {target_seconds:.0f}s ({args.hours:.1f} hours)")

    # Optimized config for long training
    # CRITICAL: buffer_capacity must be > max_episode_steps to avoid overwriting
    config = {
        'batch_size': 64,
        'train_batch_size': 256,
        'collection_steps': 32,
        'mcts_iterations': 100,  # More iterations for better policy
        'mcts_iterations_test': 100,
        'num_blocks': 6,  # Deeper network
        'num_channels': 128,  # Wide network
        'lr': 5e-4,  # Slightly lower LR for stability
        'l2_reg': 0.0001,
        'buffer_capacity': 4000,  # CRITICAL: Must be > max_episode_steps
        'max_episode_steps': 2000,
    }

    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    env = pgx.make('2048')
    init_fn = create_init_fn(env)
    step_fn = create_stochastic_aware_step_fn(env)

    resnet = AZResnet(AZResnetConfig(
        policy_head_out_size=env.num_actions,
        num_blocks=config['num_blocks'],
        num_channels=config['num_channels'],
    ))

    stochastic_probs = compute_stochastic_probs()
    decision_step_fn = create_decision_step_fn(env)
    stochastic_step_fn = create_stochastic_step_fn(env)

    evaluator = UnifiedMCTS(
        eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
        action_selector=PUCTSelector(),
        stochastic_action_probs=stochastic_probs,
        policy_size=env.num_actions,
        num_iterations=config['mcts_iterations'],
        max_nodes=config['mcts_iterations'] + 100,
        decision_step_fn=decision_step_fn,
        stochastic_step_fn=stochastic_step_fn,
        gumbel_k=4,  # 2048 has only 4 actions
        temperature=1.0,
    )

    evaluator_test = UnifiedMCTS(
        eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
        action_selector=PUCTSelector(),
        stochastic_action_probs=stochastic_probs,
        policy_size=env.num_actions,
        num_iterations=config['mcts_iterations_test'],
        max_nodes=config['mcts_iterations_test'] + 100,
        decision_step_fn=decision_step_fn,
        stochastic_step_fn=stochastic_step_fn,
        gumbel_k=4,
        temperature=0.0,  # Greedy for testing
    )

    replay_memory = EpisodeReplayBuffer(capacity=config['buffer_capacity'])
    train_steps_per_epoch = max(1, int(
        config['batch_size'] * config['collection_steps'] / config['train_batch_size']
    ))

    # Evaluation tester - test more frequently to track max score
    eval_tester = SinglePlayerTester(
        num_episodes=32,
        epochs_per_test=10,
        name='eval'
    )

    # Estimate epochs based on ~11s/epoch from previous run
    estimated_epochs = int(target_seconds / 15)  # Conservative estimate
    print(f"\nEstimated epochs: ~{estimated_epochs}")

    trainer = Trainer(
        batch_size=config['batch_size'],
        train_batch_size=config['train_batch_size'],
        warmup_steps=0,
        collection_steps_per_epoch=config['collection_steps'],
        train_steps_per_epoch=train_steps_per_epoch,
        nn=resnet,
        loss_fn=partial(az_default_loss_fn, l2_reg_lambda=config['l2_reg']),
        optimizer=optax.adam(config['lr']),
        evaluator=evaluator,
        memory_buffer=replay_memory,
        max_episode_steps=config['max_episode_steps'],
        env_step_fn=step_fn,
        env_init_fn=init_fn,
        state_to_nn_input_fn=state_to_nn_input,
        testers=[eval_tester],
        evaluator_test=evaluator_test,
        data_transform_fns=[],
        wandb_project_name='turbozero-2048-long',
        ckpt_dir='/tmp/turbozero_2048_long',
        extra_wandb_config={
            'evaluator_type': 'UnifiedMCTS',
            'target_hours': args.hours,
        },
    )

    print(f"\nStarting training...")
    start_time = time.time()

    # Run training with time limit
    epoch = 0
    max_score_seen = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= target_seconds:
            print(f"\nTime limit reached: {elapsed/3600:.2f} hours")
            break

        remaining = target_seconds - elapsed
        remaining_epochs = max(1, int(remaining / 15))

        print(f"\n--- Epoch {epoch} (elapsed: {elapsed/60:.1f}min, remaining: {remaining/60:.1f}min) ---")

        # Run 10 epochs at a time
        batch_epochs = min(10, remaining_epochs)
        output = trainer.train_loop(seed=args.seed + epoch, num_epochs=batch_epochs, eval_every=10)

        epoch += batch_epochs

        # Check max score from wandb logs
        elapsed = time.time() - start_time
        hours = elapsed / 3600
        print(f"Completed epoch {epoch}, {hours:.2f} hours elapsed")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time:.0f}s)")
    print(f"Total epochs: {epoch}")
    print(f"Seconds per epoch: {total_time/max(1,epoch):.2f}")
    print(f"\nCheck wandb for detailed max scores: https://wandb.ai/")
    print(f"Model saved to: /tmp/turbozero_2048_long")


if __name__ == '__main__':
    main()
