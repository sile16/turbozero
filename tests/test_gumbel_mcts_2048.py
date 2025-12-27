#!/usr/bin/env python3
"""Test Gumbel MCTS vs Standard MCTS on 2048.

Compares:
1. Standard StochasticMCTS (50 simulations)
2. GumbelStochasticMCTS (16 simulations, k=16)
3. Regular MCTS without chance nodes (baseline)

Goal: Verify Gumbel achieves similar or better performance with fewer simulations.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
from functools import partial

import jax
import jax.numpy as jnp
import optax
import pgx

from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.evaluators.mcts.gumbel_stochastic_mcts import GumbelStochasticMCTS
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


def run_comparison(num_epochs: int = 50):
    """Run comparison between Standard and Gumbel StochasticMCTS."""

    print("=" * 70)
    print("Gumbel MCTS vs Standard MCTS Comparison on 2048")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")

    # Common config
    config = {
        'batch_size': 32,
        'train_batch_size': 128,
        'num_blocks': 4,
        'num_channels': 64,
        'lr': 1e-3,
        'l2_reg': 0.0001,
        'buffer_capacity': 200,
        'max_episode_steps': 1000,
    }

    env = pgx.make('2048')
    init_fn = create_init_fn(env)
    step_fn = create_stochastic_aware_step_fn(env)
    decision_step_fn = create_decision_step_fn(env)
    stochastic_step_fn = create_stochastic_step_fn(env)

    stochastic_probs = compute_stochastic_probs()

    results = {}

    # Test configurations
    configs_to_test = [
        {
            'name': 'Standard StochasticMCTS (50 sims)',
            'evaluator_class': StochasticMCTS,
            'num_iterations': 50,
            'extra_kwargs': {}
        },
        {
            'name': 'Gumbel StochasticMCTS (16 sims, k=16)',
            'evaluator_class': GumbelStochasticMCTS,
            'num_iterations': 16,
            'extra_kwargs': {'gumbel_k': 16}
        },
        {
            'name': 'Gumbel StochasticMCTS (8 sims, k=8)',
            'evaluator_class': GumbelStochasticMCTS,
            'num_iterations': 8,
            'extra_kwargs': {'gumbel_k': 8}
        },
    ]

    for cfg in configs_to_test:
        print(f"\n{'='*70}")
        print(f"Testing: {cfg['name']}")
        print(f"{'='*70}")

        # Create network
        resnet = AZResnet(AZResnetConfig(
            policy_head_out_size=env.num_actions,
            num_blocks=config['num_blocks'],
            num_channels=config['num_channels'],
        ))

        # Create evaluator
        evaluator_kwargs = {
            'eval_fn': make_nn_eval_fn(resnet, state_to_nn_input),
            'action_selector': PUCTSelector(),
            'stochastic_action_probs': stochastic_probs,
            'policy_size': env.num_actions,
            'num_iterations': cfg['num_iterations'],
            'max_nodes': cfg['num_iterations'] + 50,
            'decision_step_fn': decision_step_fn,
            'stochastic_step_fn': stochastic_step_fn,
            'temperature': 1.0,
            'progressive_threshold': 1.0,
        }
        evaluator_kwargs.update(cfg['extra_kwargs'])

        evaluator = cfg['evaluator_class'](**evaluator_kwargs)

        # Test evaluator (same config but temp=0 for greedy)
        test_kwargs = evaluator_kwargs.copy()
        test_kwargs['temperature'] = 0.0
        evaluator_test = cfg['evaluator_class'](**test_kwargs)

        replay_memory = EpisodeReplayBuffer(capacity=config['buffer_capacity'])

        eval_tester = SinglePlayerTester(
            num_episodes=16,
            epochs_per_test=10,
            name='eval'
        )

        trainer = Trainer(
            batch_size=config['batch_size'],
            train_batch_size=config['train_batch_size'],
            warmup_steps=0,
            collection_steps_per_epoch=16,
            train_steps_per_epoch=4,
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
            wandb_project_name=None,  # Disable wandb for test
            ckpt_dir=None,
        )

        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()

        try:
            output = trainer.train_loop(seed=42, num_epochs=num_epochs, eval_every=10)
            elapsed = time.time() - start_time

            results[cfg['name']] = {
                'elapsed': elapsed,
                'epochs_per_sec': num_epochs / elapsed,
                'status': 'success'
            }

            print(f"Completed in {elapsed:.1f}s ({num_epochs/elapsed:.2f} epochs/sec)")

        except Exception as e:
            elapsed = time.time() - start_time
            results[cfg['name']] = {
                'elapsed': elapsed,
                'status': 'error',
                'error': str(e)
            }
            print(f"ERROR: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, result in results.items():
        if result['status'] == 'success':
            print(f"{name}:")
            print(f"  Time: {result['elapsed']:.1f}s")
            print(f"  Speed: {result['epochs_per_sec']:.2f} epochs/sec")
        else:
            print(f"{name}: FAILED - {result.get('error', 'unknown')}")

    # Calculate speedup
    if 'Standard StochasticMCTS (50 sims)' in results and results['Standard StochasticMCTS (50 sims)']['status'] == 'success':
        baseline = results['Standard StochasticMCTS (50 sims)']['elapsed']
        for name, result in results.items():
            if result['status'] == 'success' and name != 'Standard StochasticMCTS (50 sims)':
                speedup = baseline / result['elapsed']
                print(f"\n{name} speedup: {speedup:.2f}x faster than baseline")

    return results


if __name__ == '__main__':
    results = run_comparison(num_epochs=30)
