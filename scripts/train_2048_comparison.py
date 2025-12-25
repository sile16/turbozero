#!/usr/bin/env python3
"""
2048 Training: MCTS vs StochasticMCTS Comparison
=================================================

Reproduces key findings from "Planning in Stochastic Environments with a Learned Model"
(Stochastic MuZero, ICLR 2022) comparing:
- Standard AlphaZero MCTS (treats game as deterministic)
- StochasticMCTS (properly handles chance nodes for tile spawning)

Paper's 2048 config:
- 100 MCTS simulations
- 10-block ResNet, 256 hidden
- 32 chance outcomes (matches 16 positions × 2 tile values)
- Discount: 0.999
- Training: 20M steps, batch 1024, lr 0.0003

Usage:
    poetry run python scripts/train_2048_comparison.py [--epochs 50] [--quick]
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
import wandb

from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.alphazero import AlphaZero
from core.memory.replay_memory import EpisodeReplayBuffer
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.training.loss_fns import az_default_loss_fn
from core.training.train import Trainer
from core.types import StepMetadata
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.testing.single_player_tester import SinglePlayerTester


def create_decision_step_fn(env):
    """Create step function for decision nodes (player actions).

    For 2048: player chooses slide direction (0-3: up, down, left, right)
    Uses pgx 3.1.0 step_deterministic API (no key needed).

    Note: Accepts optional key argument for trainer compatibility, but ignores it.
    """
    def step_fn(state, action, key=None):
        # Ensure action is a JAX array
        action = jnp.asarray(action, dtype=jnp.int32)
        new_state = env.step_deterministic(state, action)
        return new_state, StepMetadata(
            rewards=new_state.rewards,
            action_mask=new_state.legal_action_mask,  # Size 4
            terminated=new_state.terminated,
            cur_player_id=new_state.current_player,
            step=new_state._step_count,
            is_stochastic=new_state._is_stochastic
        )
    return step_fn


def create_stochastic_step_fn(env):
    """Create step function for chance nodes (environment randomness).

    For 2048: environment spawns a tile (32 outcomes = 16 positions × 2 values)
    Uses pgx 3.1.0 step_stochastic API (no key needed).

    Note: Accepts optional key argument for compatibility, but ignores it.
    """
    def step_fn(state, outcome, key=None):
        # Ensure outcome is a JAX array
        outcome = jnp.asarray(outcome, dtype=jnp.int32)
        new_state = env.step_stochastic(state, outcome)
        return new_state, StepMetadata(
            rewards=new_state.rewards,
            action_mask=new_state.legal_action_mask,  # Size 4 (for next decision)
            terminated=new_state.terminated,
            cur_player_id=new_state.current_player,
            step=new_state._step_count,
            is_stochastic=new_state._is_stochastic
        )
    return step_fn


def create_mcts_step_fn(env):
    """Create step function for regular MCTS (non-stochastic-aware).

    Uses pgx backwards-compatible step() which:
    1. Calls step_deterministic
    2. If resulting state is stochastic, automatically calls stochastic_random_step

    This treats stochastic transitions as random noise - MCTS doesn't model them.
    """
    def step_fn(state, action, key):
        action = jnp.asarray(action, dtype=jnp.int32)
        # Use the backwards-compatible step that handles stochastic nodes with random sampling
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


def create_stochastic_aware_step_fn(env):
    """Create step function for StochasticMCTS (stochastic-aware).

    StochasticMCTS explicitly handles chance nodes:
    - At decision nodes: returns action (0-3 for 2048)
    - At chance nodes: returns outcome (0-31 for 2048)

    This step function checks current state type and calls appropriate pgx function.
    Uses tree_map with jnp.where for vmap compatibility.
    """
    def step_fn(state, action, key=None):
        is_stochastic = getattr(state, '_is_stochastic', jnp.array(False))
        action = jnp.asarray(action, dtype=jnp.int32)

        # Compute both (for vmap compatibility) and select based on state type
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
    """Convert 2048 state to neural network input."""
    return state.observation


def compute_stochastic_probs():
    """Compute stochastic action probabilities for 2048.

    32 outcomes: position (0-15) × tile_value (2=90%, 4=10%)
    outcome = position * 2 + (0 for tile=2, 1 for tile=4)
    """
    probs = jnp.zeros(32)
    for pos in range(16):
        probs = probs.at[pos * 2].set(0.9 / 16)      # tile=2 (90%)
        probs = probs.at[pos * 2 + 1].set(0.1 / 16)  # tile=4 (10%)
    return probs


def run_2048_training(evaluator_type: str, config: dict, seed: int = 0, wandb_run=None):
    """Run 2048 training with specified evaluator type.

    Args:
        evaluator_type: 'MCTS' or 'StochasticMCTS'
        config: Training configuration dict
        seed: Random seed
        wandb_run: Optional existing wandb run to use
    """
    print(f"\n{'='*60}")
    print(f"Training 2048 with {evaluator_type}")
    print(f"{'='*60}")

    env = pgx.make('2048')

    # Create step functions
    decision_step_fn = create_decision_step_fn(env)
    stochastic_step_fn = create_stochastic_step_fn(env)
    mcts_step_fn = create_mcts_step_fn(env)  # For regular MCTS (random stochastic sampling)
    stochastic_aware_step_fn = create_stochastic_aware_step_fn(env)  # For StochasticMCTS
    init_fn = create_init_fn(env)

    # Neural network - policy size depends on evaluator type
    if evaluator_type == 'StochasticMCTS':
        # StochasticMCTS: neural network only outputs decision actions (4)
        # This is more efficient - no wasted capacity on stochastic outcomes
        policy_size = env.num_actions  # 4
    else:
        # Regular MCTS: needs to handle combined action space
        policy_size = env.num_actions  # 4

    resnet = AZResnet(AZResnetConfig(
        policy_head_out_size=policy_size,
        num_blocks=config['num_blocks'],
        num_channels=config['num_channels'],
    ))

    # Create evaluator
    if evaluator_type == 'MCTS':
        # Standard MCTS - treats everything as deterministic
        az_evaluator = AlphaZero(MCTS)(
            eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
            action_selector=PUCTSelector(),
            num_iterations=config['mcts_iterations'],
            max_nodes=config['mcts_iterations'] + 100,
            branching_factor=env.num_actions,
            temperature=1.0,
        )
        az_evaluator_test = AlphaZero(MCTS)(
            eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
            action_selector=PUCTSelector(),
            num_iterations=config['mcts_iterations_test'],
            max_nodes=config['mcts_iterations_test'] + 20,
            branching_factor=env.num_actions,
            temperature=0.0,
        )
        # Regular MCTS uses step() which handles stochastic nodes with random sampling
        step_fn = mcts_step_fn
    else:
        # StochasticMCTS - handles chance nodes properly with progressive expansion
        stochastic_probs = compute_stochastic_probs()

        az_evaluator = StochasticMCTS(
            eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
            action_selector=PUCTSelector(),
            stochastic_action_probs=stochastic_probs,
            policy_size=policy_size,  # 4 decision actions
            num_iterations=config['mcts_iterations'],
            max_nodes=config['mcts_iterations'] + 100,
            decision_step_fn=decision_step_fn,
            stochastic_step_fn=stochastic_step_fn,
            temperature=1.0,
            progressive_threshold=1.0,  # Expand child when parent_visits * prob > 1.0
        )
        az_evaluator_test = StochasticMCTS(
            eval_fn=make_nn_eval_fn(resnet, state_to_nn_input),
            action_selector=PUCTSelector(),
            stochastic_action_probs=stochastic_probs,
            policy_size=policy_size,
            num_iterations=config['mcts_iterations_test'],
            max_nodes=config['mcts_iterations_test'] + 20,
            decision_step_fn=decision_step_fn,
            stochastic_step_fn=stochastic_step_fn,
            temperature=0.0,
            progressive_threshold=1.0,
        )
        # StochasticMCTS uses explicit step functions based on state type:
        # - At decision nodes: action is 0-3, call step_deterministic
        # - At chance nodes: action is 0-31, call step_stochastic
        step_fn = stochastic_aware_step_fn

    # Replay buffer
    replay_memory = EpisodeReplayBuffer(capacity=config['buffer_capacity'])

    # Calculate train steps per epoch
    train_steps_per_epoch = max(1, int(
        config['batch_size'] * config['collection_steps'] / config['train_batch_size']
    ))

    # Create evaluation tester
    eval_tester = SinglePlayerTester(
        num_episodes=config.get('eval_episodes', 16),
        epochs_per_test=1,  # Evaluate every epoch during testing
        name='eval'
    )

    # Create trainer
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
        wandb_project_name=config.get('wandb_project', 'turbozero-2048'),
        ckpt_dir=f'/tmp/turbozero_2048_{evaluator_type.lower()}',
        wandb_run=wandb_run,
        extra_wandb_config={'evaluator_type': evaluator_type},
    )

    # Training
    start_time = time.time()

    output = trainer.train_loop(
        seed=seed,
        num_epochs=config['epochs'],
        eval_every=config['eval_every']
    )

    elapsed = time.time() - start_time
    print(f"\n{evaluator_type} training complete in {elapsed:.1f}s")

    return output, elapsed


def main():
    parser = argparse.ArgumentParser(description='2048: MCTS vs StochasticMCTS Comparison')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--quick', action='store_true', help='Quick test with smaller config')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mcts-only', action='store_true', help='Only run MCTS')
    parser.add_argument('--stochastic-only', action='store_true', help='Only run StochasticMCTS')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()

    # wandb project name
    wandb_project = '' if args.no_wandb else 'turbozero-2048'

    # Configuration
    if args.quick:
        config = {
            'epochs': 10,
            'batch_size': 64,
            'train_batch_size': 256,
            'collection_steps': 32,
            'mcts_iterations': 50,
            'mcts_iterations_test': 25,
            'num_blocks': 4,
            'num_channels': 64,
            'lr': 1e-3,
            'l2_reg': 0.0001,
            'buffer_capacity': 200,
            'max_episode_steps': 1000,
            'eval_every': 2,
            'eval_episodes': 8,
            'wandb_project': wandb_project,
        }
    else:
        config = {
            'epochs': args.epochs,
            'batch_size': 128,
            'train_batch_size': 512,
            'collection_steps': 64,
            'mcts_iterations': 100,
            'mcts_iterations_test': 100,  # Same as training per paper
            'num_blocks': 6,
            'num_channels': 128,
            'lr': 3e-4,
            'l2_reg': 0.0001,
            'buffer_capacity': 500,
            'max_episode_steps': 2000,
            'eval_every': 5,
            'eval_episodes': 16,
            'wandb_project': wandb_project,
        }

    print("="*60)
    print("2048: MCTS vs StochasticMCTS Comparison")
    print("Based on 'Planning in Stochastic Environments' (ICLR 2022)")
    print("="*60)
    print(f"JAX devices: {jax.devices()}")
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    results = {}

    # Run MCTS training
    if not args.stochastic_only:
        # Create a separate wandb run for MCTS
        mcts_run = None
        if wandb_project:
            mcts_run = wandb.init(
                project=wandb_project,
                name=f'mcts-2048-{args.seed}',
                config={**config, 'evaluator_type': 'MCTS'},
                reinit=True
            )
        mcts_output, mcts_time = run_2048_training('MCTS', config, seed=args.seed, wandb_run=mcts_run)
        results['MCTS'] = {'time': mcts_time}
        if mcts_run:
            mcts_run.finish()

    # Run StochasticMCTS training
    if not args.mcts_only:
        # Create a separate wandb run for StochasticMCTS
        stochastic_run = None
        if wandb_project:
            stochastic_run = wandb.init(
                project=wandb_project,
                name=f'stochastic-mcts-2048-{args.seed}',
                config={**config, 'evaluator_type': 'StochasticMCTS'},
                reinit=True
            )
        stochastic_output, stochastic_time = run_2048_training(
            'StochasticMCTS', config, seed=args.seed, wandb_run=stochastic_run
        )
        results['StochasticMCTS'] = {'time': stochastic_time}
        if stochastic_run:
            stochastic_run.finish()

    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  Training time: {data['time']:.1f}s")

    print("\n" + "="*60)
    print("Key design improvement in this version:")
    print("- StochasticMCTS uses separate step functions for decisions vs chance")
    print("- Neural network only outputs 4 policy logits (decision actions)")
    print("- No shape padding hacks needed")
    print("="*60)


if __name__ == '__main__':
    main()
