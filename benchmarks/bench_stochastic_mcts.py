#!/usr/bin/env python3
"""
Benchmark implementation for StochasticMCTS evaluator.

Note: Linter errors about step_fn and jitted_step not being callable can be safely ignored.
These errors occur because the linter cannot understand JAX's JIT compilation and function
transformations. The functions are properly defined and compiled at runtime through JAX's
transformation system.
"""

import os
import sys
import time
import json
import signal
import argparse
import platform
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Import JAX
import jax
import jax.numpy as jnp
import chex

# Import backgammon environment
import pgx.backgammon as bg

# Import tqdm for progress bars
from tqdm import tqdm

# Import types
from core.types import StepMetadata

# Import MCTS and StochasticMCTS Evaluators
from core.evaluators.mcts import MCTS, MCTSTree
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.evaluators.mcts.action_selection import PUCTSelector

# Import common benchmark functionality
from benchmarks.benchmark_common import (
    BatchBenchResult,
    BenchmarkProfile,
    DEFAULT_MEMORY_LIMIT_GB,
    DEFAULT_BENCHMARK_DURATION,
    PROFILE_DIR,
    GRAPHS_DIR,
    format_human_readable,
    get_system_info,
    print_system_info,
    get_memory_usage,
    save_profile,
    load_profile,
    generate_benchmark_plots,
    print_benchmark_summary,
    validate_against_profile,
    select_batch_sizes_for_profile,
    create_profile_filename,
    print_summary_table,
    get_cpu_gpu_usage,
    BaseBenchmark,
)

# Import base MCTS benchmark
from benchmarks.bench_mcts import MCTSBenchmarkBase

# Import CLI utilities
from benchmarks.benchmark_cli import parse_batch_sizes

# Stochastic MCTS Constants
DEFAULT_NUM_ITERATIONS = 200  # Default number of MCTS iterations/simulations
MAX_NODES = 10000  # Maximum number of nodes in the MCTS tree

def dummy_apply_fn(params: chex.ArrayTree, obs: chex.Array, key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
    """Dummy NN apply function returning uniform policy logits and zero value."""
    # Get the action space size from the environment
    env = bg.Backgammon(simple_doubles=True)
    num_actions = env.num_actions
    
    # Returns (policy_logits, value)
    # Logits are preferred over probabilities for numerical stability with softmax/sampling.
    # Zero logits correspond to a uniform policy after masking/softmax.
    policy_logits = jnp.zeros(num_actions)
    value = jnp.zeros(())
    return policy_logits, value

dummy_params = {} # No actual params needed

def dummy_state_to_nn_input_fn(state: chex.ArrayTree) -> chex.Array:
    """Dummy function to convert environment state to NN input."""
    # The actual content doesn't matter much since dummy_apply_fn ignores it,
    # but shape might matter for JIT compilation if sizes aren't fixed.
    # Return a fixed-size dummy observation.
    return jnp.zeros(100)

class StochasticMCTSBenchmark(MCTSBenchmarkBase):
    """Benchmark implementation for StochasticMCTS."""
    
    def __init__(self, num_simulations: int):
        super().__init__(
            name="StochasticMCTS",
            description=f"Stochastic MCTS benchmark with {num_simulations} simulations",
            num_simulations=num_simulations
        )
    
    def create_evaluator(self) -> StochasticMCTS:
        """Create the StochasticMCTS evaluator."""
        return StochasticMCTS(
            eval_fn=dummy_apply_fn,
            action_selector=self.action_selector,
            stochastic_action_probs=self.env.stochastic_action_probs,
            branching_factor=self.num_actions,
            max_nodes=MAX_NODES,
            num_iterations=self.num_simulations,
            discount=-1.0,
            temperature=1.0,
            persist_tree=True
        )
    
    def make_env_step_fn(self, step_key):
        """Create an environment step function with fixed key."""
        def wrapped_step_fn(env_state, action, key):
            # Check if this is a stochastic state
            is_stochastic = env_state.is_stochastic if hasattr(env_state, 'is_stochastic') else False
            
            # Use conditional for stochastic vs deterministic step
            new_state = jax.lax.cond(
                is_stochastic,
                lambda s, a, k: self.env.stochastic_step(s, a),
                lambda s, a, k: self.env.step(s, a, k),
                env_state, action, key
            )
            
            metadata = StepMetadata(
                action_mask=new_state.legal_action_mask,
                terminated=new_state.terminated,
                rewards=new_state.rewards,
                cur_player_id=new_state.current_player,
                step=new_state._step_count if hasattr(new_state, '_step_count') else 0
            )
            return new_state, metadata
        return wrapped_step_fn
    
    def step_single_state(self, key: chex.PRNGKey, env_state: chex.ArrayTree, eval_state: MCTSTree):
        """Step a single environment using the StochasticMCTS evaluator."""
        eval_key, action_key, step_key = jax.random.split(key, 3)
        
        # Get metadata
        metadata = StepMetadata(
            action_mask=env_state.legal_action_mask,
            terminated=env_state.terminated,
            rewards=env_state.rewards,
            cur_player_id=env_state.current_player,
            step=env_state._step_count if hasattr(env_state, '_step_count') else 0
        )
        
        # Create step function
        current_env_step_fn = self.make_env_step_fn(step_key)
        
        # Determine if this is a stochastic state
        is_stochastic = env_state.is_stochastic if hasattr(env_state, 'is_stochastic') else False
        
        # Use conditional to call appropriate evaluate method
        mcts_output = jax.lax.cond(
            is_stochastic,
            lambda: self.evaluator.stochastic_evaluate(
                key=eval_key,
                eval_state=eval_state,
                env_state=env_state,
                root_metadata=metadata,
                params=dummy_params,
                env_step_fn=current_env_step_fn
            ),
            lambda: self.evaluator.evaluate(
                key=eval_key,
                eval_state=eval_state,
                env_state=env_state,
                root_metadata=metadata,
                params=dummy_params,
                env_step_fn=current_env_step_fn
            )
        )
        
        # Take action
        action = mcts_output.action
        
        # Step environment based on state type
        next_env_state = jax.lax.cond(
            is_stochastic,
            lambda: self.env.stochastic_step(env_state, action),
            lambda: self.env.step(env_state, action, step_key)
        )
        
        next_eval_state = self.evaluator.step(mcts_output.eval_state, action)
        
        return next_env_state, next_eval_state

def main():
    """Main entry point for the StochasticMCTS benchmark script."""
    from benchmarks.benchmark_cli import create_benchmark_parser, parse_batch_sizes
    
    parser = create_benchmark_parser(
        description="Benchmark for StochasticMCTS evaluator",
        include_mcts_args=True,
        default_num_iterations=DEFAULT_NUM_ITERATIONS
    )
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = StochasticMCTSBenchmark(args.num_simulations)
    
    # Parse custom batch sizes if provided
    custom_batch_sizes = None
    if args.batch_sizes:
        try:
            custom_batch_sizes = parse_batch_sizes(args.batch_sizes)
            print(f"Using custom batch sizes: {custom_batch_sizes}", flush=True)
        except ValueError as e:
            print(str(e), flush=True)
            sys.exit(1)
    
    if args.single_batch:
        # Run single batch benchmark
        result, nodes = benchmark.benchmark_batch_size(args.single_batch, args.duration)
        print("\nSingle Batch Benchmark Results Summary:")
        print_summary_table([result], title=f"Batch Size {args.single_batch} Results")
        print(f"Maximum nodes: {nodes}")
    else:
        # Run discovery or validation
        profile = benchmark.load_profile()
        
        if args.validate and profile:
            # Validation mode
            print("\nValidating against existing profile...", flush=True)
            results, max_nodes = benchmark.discover_optimal_batch_sizes(
                args.memory_limit,
                args.duration,
                profile["batch_sizes"]
            )
            validate_against_profile(results, profile, benchmark.system_info)
            print(f"Maximum nodes observed: {max_nodes}")
        else:
            # Discovery mode
            results, max_nodes = benchmark.discover_optimal_batch_sizes(
                args.memory_limit,
                args.duration,
                custom_batch_sizes
            )
            if results:
                benchmark.save_profile(results, {"max_node_count": max_nodes})

if __name__ == "__main__":
    main() 