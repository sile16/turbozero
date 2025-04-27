#!/usr/bin/env python3
"""
Base MCTS benchmark implementation that can be extended for different MCTS variants.

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

# Import MCTS Evaluator - Updated to use the correct classes
from core.evaluators.mcts import MCTS, MCTSConfig, MCTSTree, MCTSActionSelector
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
    create_profile_filename,
    save_profile,
    load_profile,
    generate_benchmark_plots,
    print_benchmark_summary,
    validate_against_profile,
    select_batch_sizes_for_profile,
    print_summary_table,
    get_cpu_gpu_usage,
    BaseBenchmark,
)

# Import CLI utilities
from benchmarks.benchmark_cli import parse_batch_sizes

# MCTS Constants
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

class MCTSBenchmarkBase(BaseBenchmark):
    """Base class for MCTS benchmarks providing common functionality."""
    
    def __init__(self, name: str, description: str, num_simulations: int):
        super().__init__(name, description)
        self.num_simulations = num_simulations
        self.env = bg.Backgammon(simple_doubles=True)
        self.num_actions = self.env.num_actions
        self.action_selector = PUCTSelector(c=1.0)
        
    def create_evaluator(self) -> MCTS:
        """Create the MCTS evaluator - override in subclasses."""
        return MCTS(
            eval_fn=dummy_apply_fn,
            action_selector=self.action_selector,
            branching_factor=self.num_actions,
            max_nodes=MAX_NODES,
            num_iterations=self.num_simulations,
            discount=-1.0,
            temperature=1.0,
            persist_tree=True
        )
    
    def make_env_step_fn(self, step_key):
        """Create an environment step function with fixed key."""
        def wrapped_step_fn(env_state, action):
            new_state = self.env.step(env_state, action, step_key)
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
        """Step a single environment using the MCTS evaluator."""
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
        
        # Evaluate with MCTS
        mcts_output = self.evaluator.evaluate(
            key=eval_key,
            eval_state=eval_state,
            env_state=env_state,
            root_metadata=metadata,
            params=dummy_params,
            env_step_fn=current_env_step_fn
        )
        
        # Take action
        action = mcts_output.action
        next_env_state = self.env.step(env_state, action, step_key)
        next_eval_state = self.evaluator.step(mcts_output.eval_state, action)
        
        return next_env_state, next_eval_state
    
    def step_batch_with_reset(self, key: chex.PRNGKey, env_states: chex.ArrayTree, eval_states: chex.ArrayTree):
        """Step a batch of environments with reset handling."""
        step_keys, reset_keys = jax.random.split(key, 2)
        batch_step_keys = jax.random.split(step_keys, self.batch_size)
        batch_reset_keys = jax.random.split(reset_keys, self.batch_size)
        
        # Vmap the single state step
        vectorized_step = jax.vmap(
            lambda k, s1, s2: self.step_single_state(k, s1, s2),
            in_axes=(0, 0, 0)
        )
        next_env_states, next_eval_states = vectorized_step(batch_step_keys, env_states, eval_states)
        
        # Check termination
        terminated = next_env_states.terminated
        
        # Reset terminated states
        reset_env_states = jax.vmap(self.env.init)(batch_reset_keys)
        reset_eval_states = jax.vmap(self.evaluator.reset)(eval_states)
        
        # Select between next and reset states
        def where_terminated(next_s, reset_s):
            return jnp.where(
                terminated.reshape(-1, *([1]*(len(next_s.shape)-1))),
                reset_s,
                next_s
            )
        
        final_env_states = jax.tree_util.tree_map(where_terminated, next_env_states, reset_env_states)
        final_eval_states = jax.tree_util.tree_map(where_terminated, next_eval_states, reset_eval_states)
        
        return final_env_states, final_eval_states, terminated
    
    def benchmark_batch_size(self, batch_size: int, max_duration: int = DEFAULT_BENCHMARK_DURATION) -> Tuple[BatchBenchResult, int]:
        """Run benchmark for a specific batch size."""
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1.")
        
        self.batch_size = batch_size
        print(f"\nBenchmarking MCTS: Batch={batch_size}, Sims={self.num_simulations} for {max_duration}s", flush=True)
        
        # Initialize
        self.evaluator = self.create_evaluator()
        key = jax.random.PRNGKey(0)
        
        # Initialize states
        key, env_init_key = jax.random.split(key)
        env_init_keys = jax.random.split(env_init_key, batch_size)
        env_states = jax.vmap(self.env.init)(env_init_keys)
        
        # Initialize evaluator states
        template_state = self.env.init(jax.random.PRNGKey(0))
        eval_states = self.evaluator.init_batched(batch_size, template_state)
        
        # Compile step function
        step_fn = jax.jit(self.step_batch_with_reset)  # type: ignore[assignment]
        
        # Warmup
        print("Compiling and warming up...", flush=True)
        try:
            print("First compilation pass...", flush=True)
            key, subkey = jax.random.split(key)
            # pylint: disable=not-callable
            new_states = step_fn(subkey, env_states, eval_states)
            jax.block_until_ready(new_states)
            print("Initial compilation successful", flush=True)
            
            print("Running warm-up iterations...", flush=True)
            for _ in range(num_warmup):
                key, subkey = jax.random.split(key)
                new_states = step_fn(subkey, *new_states)  # pylint: disable=not-callable
            jax.block_until_ready(new_states)
            print("Warm-up complete", flush=True)
            # pylint: enable=not-callable
            return new_states
        except Exception as e:
            print(f"Error warming up: {e}", flush=True)
            return None
        
        # Benchmark loop
        start_time = time.time()
        total_moves = 0
        completed_games = 0
        current_game_moves = np.zeros(batch_size, dtype=np.int32)
        game_lengths = []
        max_node_count = 0
        
        # Track memory
        initial_memory_gb = get_memory_usage()
        peak_memory_gb = initial_memory_gb
        
        with tqdm(total=max_duration, desc=f"MCTS B={batch_size} S={self.num_simulations}", unit="s") as pbar:
            while (elapsed_time := time.time() - start_time) < max_duration:
                # Run iteration
                result = self.run_benchmark_iteration(step_fn, [env_states, eval_states], pbar, start_time, max_duration)
                if result is None:
                    break
                    
                env_states, eval_states, terminated = result
                
                # Update statistics
                active_mask = ~terminated
                moves_this_step = np.sum(active_mask)
                total_moves += moves_this_step
                current_game_moves[active_mask] += 1
                
                # Track node count
                try:
                    eval_state_sample = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], eval_states))
                    if hasattr(eval_state_sample, 'next_free_idx'):
                        max_node_count = max(max_node_count, int(eval_state_sample.next_free_idx))
                except Exception as e:
                    print(f"Error getting node count: {e}", flush=True)
                
                # Process completed games
                if np.any(terminated):
                    terminated_indices = np.where(terminated)[0]
                    for i in terminated_indices:
                        completed_games += 1
                        game_length = current_game_moves[i]
                        game_lengths.append(game_length)
                        current_game_moves[i] = 0
                
                # Update memory tracking
                current_memory = get_memory_usage()
                peak_memory_gb = max(peak_memory_gb, current_memory)
                
                # Update progress display
                if time.time() - pbar.last_print_t > 0.5:
                    moves_per_sec = total_moves / elapsed_time
                    games_per_sec = completed_games / elapsed_time
                    pbar.set_postfix({
                        "moves/s": f"{format_human_readable(moves_per_sec)}/s",
                        "games/s": f"{format_human_readable(games_per_sec)}/s",
                        "mem": f"{peak_memory_gb:.2f}GB"
                    })
        
        # Calculate final metrics
        final_elapsed_time = time.time() - start_time
        moves_per_second = total_moves / final_elapsed_time
        games_per_second = completed_games / final_elapsed_time
        avg_game_length = np.mean(game_lengths) if game_lengths else 0
        median_game_length = np.median(game_lengths) if game_lengths else 0
        min_game_length = np.min(game_lengths) if game_lengths else 0
        max_game_length = np.max(game_lengths) if game_lengths else 0
        efficiency = moves_per_second / peak_memory_gb
        
        result = BatchBenchResult(
            batch_size=batch_size,
            moves_per_second=moves_per_second,
            games_per_second=games_per_second,
            avg_game_length=avg_game_length,
            median_game_length=median_game_length,
            min_game_length=min_game_length,
            max_game_length=max_game_length,
            memory_usage_gb=peak_memory_gb,
            efficiency=efficiency
        )
        
        return result, max_node_count
    
    def discover_optimal_batch_sizes(self, memory_limit_gb: float = DEFAULT_MEMORY_LIMIT_GB,
                                   max_duration: int = DEFAULT_BENCHMARK_DURATION,
                                   custom_batch_sizes: Optional[List[int]] = None) -> Tuple[List[BatchBenchResult], int]:
        """Discover optimal batch sizes through benchmarking."""
        print(f"\nDiscovering optimal batch sizes for {self.name} (memory limit: {memory_limit_gb:.2f}GB)", flush=True)
        
        all_results = []
        valid_results = []
        max_node_count = 0
        
        # Use custom batch sizes or generate sequence
        if custom_batch_sizes:
            batch_sizes = custom_batch_sizes
            print(f"Using custom batch sizes: {batch_sizes}", flush=True)
        else:
            batch_size = 1
            batch_sizes = []
            while batch_size <= 2048:  # Cap at reasonable maximum
                batch_sizes.append(batch_size)
                batch_size *= 2
            print(f"Using default progressive batch sizes: {batch_sizes}", flush=True)
        
        # Run benchmarks
        for batch_size in batch_sizes:
            try:
                result, nodes = self.benchmark_batch_size(batch_size, max_duration)
                all_results.append(result)
                valid_results.append(result)
                max_node_count = max(max_node_count, nodes)
                
                # Check termination conditions
                if result.memory_usage_gb >= memory_limit_gb:
                    print(f"Memory limit reached: {result.memory_usage_gb:.2f}GB >= {memory_limit_gb:.2f}GB", flush=True)
                    break
                    
                if len(valid_results) > 1:
                    perf_improvement = (valid_results[-1].moves_per_second / 
                                      valid_results[-2].moves_per_second - 1.0)
                    if perf_improvement < 0.1:
                        print("Diminishing returns detected (less than 10% improvement)", flush=True)
                        break
                
            except Exception as e:
                print(f"Error benchmarking batch size {batch_size}: {e}", flush=True)
                break
        
        return valid_results, max_node_count

class MCTSBenchmark(MCTSBenchmarkBase):
    """Standard MCTS benchmark implementation."""
    
    def __init__(self, num_simulations: int):
        super().__init__(
            name="MCTS",
            description=f"Standard MCTS benchmark with {num_simulations} simulations",
            num_simulations=num_simulations
        )

def main():
    """Main entry point for the MCTS benchmark script."""
    from benchmarks.benchmark_cli import create_benchmark_parser, parse_batch_sizes
    
    parser = create_benchmark_parser(
        description="Benchmark for MCTS evaluator",
        include_mcts_args=True,
        default_num_iterations=DEFAULT_NUM_ITERATIONS
    )
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = MCTSBenchmark(args.num_simulations)
    
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