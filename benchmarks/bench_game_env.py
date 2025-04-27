#!/usr/bin/env python3
"""
Benchmark for the pgx backgammon game environment.

This script measures the performance of the backgammon environment in terms of moves per second
for different batch sizes. It includes a discovery mode to find optimal batch sizes and 
a validation mode to verify performance against previous runs.

Note: Linter errors about step_fn and jitted_step not being callable can be safely ignored.
These errors occur because the linter cannot understand JAX's JIT compilation and function
transformations. The functions are properly defined and compiled at runtime through JAX's
transformation system.
"""

from calendar import c
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
from core.types import StepMetadata
import datetime

# For plotting results
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Import JAX before setting any environment variables
import jax
import jax.numpy as jnp
import chex

# Import backgammon environment
import pgx.backgammon as bg

# Import tqdm for progress bars
from tqdm import tqdm

# Import common benchmark functionality
from benchmarks.benchmark_common import (
    DEFAULT_MEMORY_LIMIT_GB,
    DEFAULT_BENCHMARK_DURATION,
    PROFILE_DIR,
    GRAPHS_DIR,
    format_human_readable,
    get_system_info,
    get_memory_usage,
    create_profile_filename,
    save_profile,
    load_profile,
    generate_benchmark_plots,
    print_summary_table,
    get_cpu_gpu_usage,
    BenchmarkProfile,
    BaseBenchmark,
)

# Import CLI utilities
from benchmarks.benchmark_cli import parse_batch_sizes

# Constants
HUMAN_READABLE_UNITS = ["", "K", "M", "B", "T"]


@dataclass
class BatchBenchResult:
    """Holds the results from benchmarking a single batch size."""
    batch_size: int
    moves_per_second: float = 0.0
    games_per_second: float = 0.0
    moves_per_second_per_game: float = 0.0
    avg_game_length: float = 0.0
    median_game_length: float = 0.0
    min_game_length: float = float('inf')
    max_game_length: float = 0.0
    memory_usage_gb: float = 0.0
    efficiency: float = 0.0 # moves/s / GB
    valid: bool = False


def random_action_from_mask(key, mask):
    """
    Sample a random action index based on a legal action mask using categorical sampling.
    This approach is robust to JAX transformations like jit, vmap, cond.

    Args:
        key: JAX PRNG key.
        mask: A boolean JAX array where True indicates a legal action.
              Assumes at least one True element if called on a non-terminal state.

    Returns:
        An integer JAX array representing the index of a randomly chosen legal action.
    """
    # Ensure mask is float to perform division
    mask_float = mask.astype(jnp.float32)
    # Calculate probabilities - uniform distribution over legal actions
    # Add small epsilon to sum to prevent division by zero if mask is all False (shouldn't happen in valid states)
    probs = mask_float / jnp.maximum(mask_float.sum(axis=-1, keepdims=True), 1e-9)

    # Convert probabilities to logits, handling log(0)
    # Add epsilon to probs before log for numerical stability
    logits = jnp.log(probs + 1e-9)

    # Sample action using categorical distribution based on logits
    # axis=-1 assumes mask/probs/logits are 1D or sampling should happen along the last axis
    return jax.random.categorical(key, logits=logits, axis=-1)


def print_memory_usage(stage: str):
    """Print current memory usage for a given stage."""
    if jax.default_backend() == 'gpu':
        memory = jax.devices()[0].memory_stats()
        used_gb = memory['bytes_in_use'] / (1024**3)
        total_gb = memory['bytes_limit'] / (1024**3)
        print(f"Memory usage at {stage}: {used_gb:.2f}GB / {total_gb:.2f}GB", flush=True)
        
        # Get CUDA memory info
        try:
            import torch
            torch_mem = torch.cuda.memory_allocated() / (1024**3)
            torch_cached = torch.cuda.memory_reserved() / (1024**3)
            print(f"PyTorch memory: {torch_mem:.2f}GB allocated, {torch_cached:.2f}GB cached", flush=True)
        except ImportError:
            pass
            
        # Get NVIDIA-SMI info
        try:
            import subprocess
            nvidia_smi = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'])
            gpu_mem = [int(x) for x in nvidia_smi.decode('utf-8').strip().split(',')]
            print(f"NVIDIA-SMI memory: {gpu_mem[0]/1024:.2f}GB used / {gpu_mem[1]/1024:.2f}GB total", flush=True)
        except Exception:
            pass
    else:
        print(f"Memory profiling not available for {jax.default_backend()} backend", flush=True)


def benchmark_batch_size(batch_size: int, max_duration: int = 120) -> 'BatchBenchResult':
    """
    Benchmark with individual state reset on termination within the JIT step.

    Args:
        batch_size: Batch size to test (must be >= 1)
        max_duration: Duration in seconds to run the benchmark

    Returns:
        BatchBenchResult containing detailed performance metrics.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    print(f"\n{'-'*30}", flush=True)
    print(f"Benchmarking batch size: {batch_size} (Individual Reset) for {max_duration} seconds", flush=True)
    print(f"{'-'*30}", flush=True)

    # Initialize environment
    print("Initializing environment...", flush=True)
    env = bg.Backgammon(simple_doubles=True)
    num_stochastic_outcomes = len(env.stochastic_action_probs)
    print_memory_usage("after env init")

    # Check if the environment has a 'reset' method, otherwise alias 'init'
    if not hasattr(env, 'reset'):
        print("Warning: Environment does not have a 'reset' method. Using 'init' as 'reset'.", flush=True)
        env.reset = env.init

    # ---- Define step functions ----
    print("Defining step functions (with integrated reset)...", flush=True)
    
    # Define a single step function
    def step_single_state(key, state):
        """Step a single state (original logic)."""
        key1, key2 = jax.random.split(key, 2)
        is_stochastic = jnp.bool_(state.is_stochastic)
        legal_actions_mask = state.legal_action_mask

        action = random_action_from_mask(key2, legal_actions_mask)

        return jax.lax.cond(
            is_stochastic,
            lambda _: env.stochastic_step(state, jax.random.randint(key1, (), 0, num_stochastic_outcomes)),
            lambda act_key: env.step(state, act_key[0], act_key[1]),
            operand=(action, key2)
        )

    print_memory_usage("after step_single_state definition")

    # Define the batch step with reset function
    def step_batch_with_reset(key, states):
        """Take steps for a batch of parallel games and reset terminated ones."""
        step_keys, reset_keys = jax.random.split(key, 2)
        batch_step_keys = jax.random.split(step_keys, batch_size)
        batch_reset_keys = jax.random.split(reset_keys, batch_size)

        vectorized_step = jax.vmap(step_single_state, in_axes=(0, 0))
        next_states = vectorized_step(batch_step_keys, states)

        terminated = next_states.terminated

        vectorized_reset = jax.vmap(env.reset, in_axes=(0,))
        reset_states = vectorized_reset(batch_reset_keys)

        final_states = jax.tree_util.tree_map(
            lambda next_s, reset_s: jnp.where(terminated.reshape(-1, *([1]*(next_s.ndim-1))), reset_s, next_s),
            next_states,
            reset_states
        )
        return final_states, terminated

    print_memory_usage("after step_batch_with_reset definition")

    # Initialize batch of states using vmap
    print(f"Initializing {batch_size} states...", flush=True)
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    init_keys = jax.random.split(init_key, batch_size)
    state = jax.vmap(env.init)(init_keys)

    print_memory_usage("after state initialization")

    # Compile step function
    step_fn = jax.jit(step_batch_with_reset)
    
    # Warmup compilation
    print("Compiling and warming up...", flush=True)
    try:
        print("First compilation pass...", flush=True)
        key, subkey = jax.random.split(key)
        new_state, _ = step_fn(subkey, state)
        jax.block_until_ready(new_state)
        print_memory_usage("after first compilation")
        
        print("Running warm-up iterations...", flush=True)
        for i in range(4):
            key, subkey = jax.random.split(key)
            new_state, _ = step_fn(subkey, new_state)
            jax.block_until_ready(new_state)
            print_memory_usage(f"after warm-up iteration {i+1}")
        state = new_state
    except Exception as e:
        print(f"Error during compilation/warm-up: {e}", flush=True)
        raise

    # ---- Benchmark ----
    print(f"Running benchmark for {max_duration} seconds (Individual Reset)...", flush=True)
    start_time = time.time()

    # Track statistics using NumPy arrays (host-side)
    current_game_moves = np.zeros(batch_size, dtype=np.int32)
    total_moves = 0
    completed_games = 0
    total_moves_in_completed_games = 0
    max_moves_per_game = 0
    min_moves_per_game = float('inf')
    game_lengths = []

    # Track memory usage
    initial_memory_gb = get_memory_usage()
    peak_memory_gb = initial_memory_gb
    print(f"Initial memory usage: {initial_memory_gb:.2f}GB", flush=True)

    print("Starting benchmark loop:", flush=True)
    with tqdm(total=max_duration, desc=f"Batch {batch_size} (Indiv Reset)", unit="s", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
        last_update_time = time.time()
        iteration_count = 0

        try:
            while (elapsed_time := time.time() - start_time) < max_duration:
                iteration_count += 1
                key, subkey = jax.random.split(key)

                # Execute the JIT-compiled step and reset function
                # It returns the next state (with resets applied) and the termination mask
                # pylint: disable=not-callable
                state, terminated_mask_jax = step_fn(subkey, state)
                # pylint: enable=not-callable
                # Block until operations are complete on device before pulling data to host
                jax.block_until_ready(state)

                # --- Corrected Move and Game Counting (using host-side logic) ---
                # Convert JAX array mask to NumPy *after* blocking
                terminated_mask_np = np.array(terminated_mask_jax)

                # Count moves only for games that were *not* terminated *before* this step.
                # Since reset happens *after* the step based on the *new* terminated state,
                # all non-terminated states contributed one move.
                # We use the inverse of the *current* termination mask as 'active_before_reset'.
                active_mask = ~terminated_mask_np
                moves_this_step = np.sum(active_mask) # Count how many states were active and took a step
                total_moves += moves_this_step
                current_game_moves[active_mask] += 1 # Increment moves only for active states

                # Check which games *just* terminated and were reset
                if np.any(terminated_mask_np):
                    terminated_indices = np.where(terminated_mask_np)[0]
                    for i in terminated_indices:
                        # This game finished in the step we just took
                        completed_games += 1
                        # The game length includes the final step
                        game_length = current_game_moves[i] + 1 # Add 1 for the step that terminated it
                        total_moves_in_completed_games += game_length
                        max_moves_per_game = max(max_moves_per_game, game_length)
                        min_moves_per_game = min(min_moves_per_game, game_length)
                        game_lengths.append(game_length)

                        # Reset the move counter for the game that just finished and was reset
                        current_game_moves[i] = 0

                # Update max moves seen across all games (even ongoing)
                if np.any(active_mask): # If any games are still ongoing
                     max_moves_per_game = max(max_moves_per_game, np.max(current_game_moves[active_mask]))


                # --- NO LONGER NEEDED: Batch Reset Logic ---
                # The reset is now handled inside step_fn for individual states

                # Update progress bar and metrics periodically
                current_time = time.time()
                if current_time - last_update_time > 0.5:
                    elapsed = current_time - start_time
                    pbar.n = round(elapsed)
                    moves_per_sec = total_moves / elapsed if elapsed > 0 else 0
                    games_per_sec = completed_games / elapsed if elapsed > 0 else 0
                    avg_moves = total_moves_in_completed_games / completed_games if completed_games > 0 else 0
                    try:
                        current_memory = get_memory_usage()
                        peak_memory_gb = max(peak_memory_gb, current_memory)
                    except NameError:
                        peak_memory_gb = 0
                        current_memory = 0

                    try:
                        moves_s_str = f"{format_human_readable(moves_per_sec)}/s"
                        games_s_str = f"{format_human_readable(games_per_sec)}/s"
                    except NameError:
                         moves_s_str = f"{moves_per_sec:.2f}/s"
                         games_s_str = f"{games_per_sec:.2f}/s"

                    pbar.set_postfix(
                        moves_s=moves_s_str,
                        games_s=games_s_str,
                        avg_moves=f"{avg_moves:.1f}",
                        mem=f"{peak_memory_gb:.2f}GB"
                    )
                    pbar.refresh()
                    last_update_time = current_time

        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user!", flush=True)
            elapsed_time = time.time() - start_time # Update elapsed time

        pbar.n = round(elapsed_time) # Final update
        pbar.refresh()


    # ---- Final calculations and statistics ----
    final_elapsed_time = time.time() - start_time

    # Update max moves one last time for any games still active at the end
    active_at_end = current_game_moves > 0 # Any game with moves > 0 was active at some point
    if np.any(active_at_end):
        max_moves_per_game = max(max_moves_per_game, np.max(current_game_moves[active_at_end]))
    if completed_games == 0 and min_moves_per_game == float('inf'):
        min_moves_per_game = 0 # Set min to 0 if no games completed

    moves_per_second = total_moves / final_elapsed_time if final_elapsed_time > 0 else 0
    games_per_second = completed_games / final_elapsed_time if final_elapsed_time > 0 else 0
    moves_per_second_per_game = moves_per_second / games_per_second if games_per_second > 0 else 0
    efficiency = moves_per_second / peak_memory_gb if peak_memory_gb > 0 else 0
    avg_game_length = total_moves_in_completed_games / completed_games if completed_games > 0 else 0
    median_game_length = np.median(game_lengths) if game_lengths else 0
    min_game_length = min_moves_per_game if game_lengths else 0
    max_game_length = max_moves_per_game if game_lengths else 0

    # --- Print Summary ---
    print("\nBenchmark complete! (Individual Reset)", flush=True)
    print(f"Batch Size: {batch_size}", flush=True)
    print(f"Total iterations (steps): {iteration_count}", flush=True)
    print(f"Total moves executed (steps on non-terminated states): {total_moves}", flush=True)
    print(f"Total completed games (individual resets): {completed_games}", flush=True)
    print(f"Elapsed time: {final_elapsed_time:.2f}s (Target: {max_duration}s)", flush=True)

    print("--- Performance Metrics ---", flush=True)
    try:
        print(f"Moves per second: {format_human_readable(moves_per_second)}/s", flush=True)
        print(f"Games per second (completed): {format_human_readable(games_per_second)}/s", flush=True)
        print(f"Moves per second per game: {format_human_readable(moves_per_second_per_game)}/s/game", flush=True)
        print(f"Efficiency (Moves/s/GB): {format_human_readable(efficiency)}/GB", flush=True)
    except NameError:
        print(f"Moves per second: {moves_per_second:.2f}/s", flush=True)
        print(f"Games per second (completed): {games_per_second:.2f}/s", flush=True)
        print(f"Moves per second per game: {moves_per_second_per_game:.2f}/s/game", flush=True)
        print(f"Efficiency (Moves/s/GB): {efficiency:.2f}/GB", flush=True)
    print(f"Peak Memory Usage: {peak_memory_gb:.2f}GB", flush=True)

    print("--- Game Length Statistics (Completed Games) ---", flush=True)
    print(f"Average moves per game: {avg_game_length:.2f}", flush=True)
    print(f"Median moves per game: {median_game_length:.1f}", flush=True)
    print(f"Minimum moves per game: {min_game_length if game_lengths else 'N/A'}", flush=True)
    print(f"Maximum moves (any game): {max_game_length}", flush=True)

    try:
        result = BatchBenchResult(
            batch_size=batch_size,
            moves_per_second=moves_per_second,
            games_per_second=games_per_second,
            moves_per_second_per_game=moves_per_second_per_game,
            avg_game_length=avg_game_length,
            median_game_length=median_game_length,
            min_game_length=min_game_length,
            max_game_length=max_game_length,
            memory_usage_gb=peak_memory_gb,
            efficiency=efficiency,
            valid=True
        )
    except NameError:
        print("Warning: BatchBenchResult class not found. Returning results as a dictionary.", flush=True)
        result = { # Fallback to dictionary
             'batch_size': batch_size, 'moves_per_second': moves_per_second, 'memory_usage_gb': peak_memory_gb,
             'games_per_second': games_per_second, 'moves_per_second_per_game': moves_per_second_per_game,
             'avg_game_length': avg_game_length, 'median_game_length': median_game_length,
             'min_game_length': min_game_length if game_lengths else 0, 'max_game_length': max_game_length,
             'total_moves': total_moves, 'completed_games': completed_games, 'elapsed_time': final_elapsed_time,
             'efficiency': efficiency, 'valid': True
        }

    return result

def discover_optimal_batch_sizes(
    memory_limit_gb: float = DEFAULT_MEMORY_LIMIT_GB,
    max_duration: int = 120,
    batch_sizes: Optional[List[int]] = None
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """
    Discover optimal batch sizes by increasing batch size until memory limit or diminishing returns.
    
    Args:
        memory_limit_gb: Maximum memory to use in GB
        max_duration: Duration in seconds for each batch size test (default: 120s = 2 minutes)
        batch_sizes: Optional list of batch sizes to test. If None, will use exponential growth.
    
    Returns:
        Tuple containing lists needed for BenchmarkProfile compatibility.
    """
    print(f"\n=== Discovering optimal batch sizes (memory limit: {memory_limit_gb:.2f}GB) ===", flush=True)
    print(f"Duration per batch size: {max_duration}s", flush=True)
    
    all_results: List[BatchBenchResult] = []
    valid_results: List[BatchBenchResult] = []  # Track only valid results
    
    if batch_sizes is None:
        # Start with batch size 1 and keep doubling
        batch_size = 1
        batch_sizes = []
        while True:
            batch_sizes.append(batch_size)
            batch_size *= 2
            if batch_size > 32768:  # Reasonable upper limit
                break
    else:
        print(f"Using provided batch sizes: {batch_sizes}", flush=True)
    
    last_perf_improvement = float('inf')
    
    print("Starting discovery process - will test provided batch sizes", flush=True)
    
    # Create a temporary profile for plotting
    temp_profile = BenchmarkProfile(
        platform=get_system_info()["platform"],
        processor=get_system_info()["processor"],
        jaxlib_type=get_system_info()["jaxlib_type"],
        device_info=get_system_info()["device_info"],
        python_version=get_system_info()["python_version"],
        jax_version=get_system_info()["jax_version"],
        batch_sizes=batch_sizes,
        moves_per_second=[],
        memory_usage_gb=[],
        games_per_second=[],
        moves_per_second_per_game=[],
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    
    with tqdm(total=len(batch_sizes), desc="Discovering Batch Sizes", unit="batch") as pbar:
        for batch_size in batch_sizes:
            try:
                result = benchmark_batch_size(batch_size, max_duration)
                all_results.append(result)
                
                if result.valid:
                    valid_results.append(result)
                    temp_profile.moves_per_second.append(result.moves_per_second)
                    temp_profile.memory_usage_gb.append(result.memory_usage_gb)
                    temp_profile.games_per_second.append(result.games_per_second)
                    temp_profile.moves_per_second_per_game.append(result.moves_per_second / batch_size)
                    
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
                
            pbar.update(1)
    
    print(f"\nDiscovery complete: Tested {len(batch_sizes)} batch sizes, {len(valid_results)} valid results")
    
    # Extract lists for plotting
    plot_batch_sizes = [r.batch_size for r in valid_results]
    plot_results = valid_results
    
    # Generate plots using the timestamp from the profile
    plot_filename = generate_benchmark_plots(plot_batch_sizes, plot_results, temp_profile.timestamp)
    
    # Return results in format compatible with BenchmarkProfile
    return (
        [r.batch_size for r in valid_results],
        [r.moves_per_second for r in valid_results],
        [r.memory_usage_gb for r in valid_results],
        [r.games_per_second for r in valid_results],
        [r.moves_per_second / r.batch_size for r in valid_results]
    )


def validate_against_profile(profile: BenchmarkProfile, max_duration: int = 120, batch_sizes: Optional[List[int]] = None) -> None:
    """
    Validate current performance against a saved profile using time-based runs.
    
    Args:
        profile: Profile to validate against
        max_duration: Duration in seconds for each batch size test (default: 120s = 2 minutes)
        batch_sizes: Optional list of batch sizes to test. If None, will use profile's batch sizes.
    """
    print("\n=== Validating against saved profile ===", flush=True)
    
    # Use provided batch sizes or fall back to profile's batch sizes
    test_batch_sizes = batch_sizes if batch_sizes is not None else profile.batch_sizes
    print(f"Testing {len(test_batch_sizes)} batch sizes: {test_batch_sizes}", flush=True)
    print(f"Duration per batch size: {max_duration}s", flush=True)
    
    current_results: List[BatchBenchResult] = []
    
    # Use tqdm for the outer loop tracking batch sizes being validated
    with tqdm(total=len(test_batch_sizes), desc="Validating Batch Sizes", unit="batch") as outer_pbar:
        for i, batch_size in enumerate(test_batch_sizes):
            print(f"\n{'='*50}", flush=True)
            print(f"Validation run {i+1}/{len(test_batch_sizes)}: Testing batch size {batch_size}", flush=True)
            print(f"{'='*50}", flush=True)
            
            # Temporarily silence stdout for benchmark to avoid progress bar confusion
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                # Benchmark with current system
                result = benchmark_batch_size(
                    batch_size, 
                    max_duration=max_duration
                )
            finally:
                # Restore stdout regardless of success/failure
                sys.stdout.close()
                sys.stdout = original_stdout
            
            current_results.append(result)
            outer_pbar.update(1) # Increment the outer progress bar
            
            # Immediate feedback after each batch size validation
            print(f"COMPLETED validation for batch size {batch_size}: {format_human_readable(result.moves_per_second)} moves/s, "
                  f"{format_human_readable(result.moves_per_second / result.games_per_second)} moves/s/game, "
                  f"{format_human_readable(result.games_per_second)} games/s, "
                  f"{result.memory_usage_gb:.2f}GB", flush=True)
    
    # --- Print Comparison and Summary --- 
    print("\n=== Validation Comparison: Current vs Profile ===", flush=True)
    header = (
        f"{'Batch':>7} | {'Prev Moves/s':>14} | {'Curr Moves/s':>14} | {'Diff (%)':>10} | "
        f"{'Prev Games/s':>14} | {'Curr Games/s':>14} | {'Diff (%)':>10} | "
        f"{'Curr Mem GB':>11}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    
    moves_diffs = []
    games_diffs = []
    
    for i, batch_size in enumerate(test_batch_sizes):
        # Get data for comparison
        current_result = current_results[i]
        previous_moves = profile.moves_per_second[i] if i < len(profile.moves_per_second) else 0
        previous_games = profile.games_per_second[i] if hasattr(profile, 'games_per_second') and profile.games_per_second and i < len(profile.games_per_second) else 0
        
        # Calculate differences
        moves_diff = (current_result.moves_per_second / previous_moves - 1.0) * 100 if previous_moves > 0 else 0
        games_diff = (current_result.games_per_second / previous_games - 1.0) * 100 if previous_games > 0 else 0
        moves_diffs.append(moves_diff)
        games_diffs.append(games_diff)
        
        moves_diff_str = f"{moves_diff:>+9.2f}%"
        games_diff_str = f"{games_diff:>+9.2f}%" if previous_games is not None else "N/A"
        
        print(f"{batch_size:>7} | "
              f"{format_human_readable(previous_moves):>14} | "
              f"{format_human_readable(current_result.moves_per_second):>14} | "
              f"{moves_diff_str:>10} | "
              f"{format_human_readable(previous_games) if previous_games is not None else 'N/A':>14} | "
              f"{format_human_readable(current_result.games_per_second):>14} | "
              f"{games_diff_str:>10} | "
              f"{current_result.memory_usage_gb:>11.2f}", flush=True)

    # Print the full summary table for the current run
    print_summary_table(current_results, title="Current Run Summary (Validation)")

    # Create comparison plots (ensure data exists)
    # Create a temporary profile with current results for plotting
    temp_profile = BenchmarkProfile(
        platform=profile.platform,
        processor=profile.processor,
        jaxlib_type=profile.jaxlib_type,
        device_info=profile.device_info,
        python_version=profile.python_version,
        jax_version=profile.jax_version,
        batch_sizes=test_batch_sizes,
        moves_per_second=[r.moves_per_second for r in current_results],
        memory_usage_gb=[r.memory_usage_gb for r in current_results],
        games_per_second=[r.games_per_second for r in current_results],
        moves_per_second_per_game=[r.moves_per_second_per_game for r in current_results],
    )
    
    comparison_plt_filename = None
    diff_plt_filename = None

    if test_batch_sizes and profile.moves_per_second and [r.moves_per_second for r in current_results]:
        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Get platform and processor for filenames
        platform_name = profile.platform.lower()
        processor_name = profile.processor.lower().replace('-', '_')
        
        # Plot moves per second comparison
        plt.figure(figsize=(12, 8))
        plt.xscale('log', base=2)
        plt.yscale('log')
        
        plt.plot(profile.batch_sizes, profile.moves_per_second, 
                marker='o', linestyle='-', linewidth=2, 
                label='Previous - Moves/s', color='tab:blue')
        plt.plot([r.batch_size for r in current_results], [r.moves_per_second for r in current_results], 
                marker='s', linestyle='--', linewidth=2, 
                label='Current - Moves/s', color='tab:orange')
        
        # If games/s data is available in both
        if hasattr(profile, 'games_per_second') and profile.games_per_second and [r.games_per_second for r in current_results]:
            plt.plot(profile.batch_sizes, profile.games_per_second, 
                    marker='o', linestyle='-', linewidth=2, 
                    label='Previous - Games/s', color='tab:green')
            plt.plot([r.batch_size for r in current_results], [r.games_per_second for r in current_results], 
                    marker='s', linestyle='--', linewidth=2, 
                    label='Current - Games/s', color='tab:red')
        
        # Add system information to the title
        title = (
            f"Benchmark Comparison: Previous vs Current\n"
            f"Platform: {profile.platform} | Processor: {profile.processor}\n"
            f"Backend: {profile.jaxlib_type} | Device: {profile.device_info}"
        )
        plt.title(title)
        plt.xlabel('Batch Size')
        plt.ylabel('Performance Metrics (log scale)')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend(loc='best')
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.xticks([r.batch_size for r in current_results], [r.batch_size for r in current_results])
        comparison_plt_filename = GRAPHS_DIR / f"{platform_name}_{processor_name}_comparison.png"
        plt.tight_layout()
        plt.savefig(comparison_plt_filename, dpi=300)
        plt.close() # Close the figure

        # Create percentage difference plot
        plt.figure(figsize=(10, 6))
        metrics_to_compare = [(moves_diffs, 'Moves/s Diff', 'tab:blue')]
        if games_diffs:
             metrics_to_compare.append((games_diffs, 'Games/s Diff', 'tab:green'))
        
        bar_width = 0.35
        index = np.arange(len(test_batch_sizes))
        
        for i, (diffs, label, color) in enumerate(metrics_to_compare):
            plt.bar(index + i * bar_width, diffs, bar_width, label=label, color=color, alpha=0.7)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Performance Difference (%)')
        plt.title('Performance Difference: Current vs Previous Profile')
        plt.xticks(index + bar_width / len(metrics_to_compare) / 2, test_batch_sizes)
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.legend(loc='best')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        diff_plt_filename = GRAPHS_DIR / f"{platform_name}_{processor_name}_diff.png"
        plt.tight_layout()
        plt.savefig(diff_plt_filename, dpi=300)
        plt.close() # Close the figure

        print(f"\nComparison plot saved to: {comparison_plt_filename}", flush=True)
        print(f"Performance difference plot saved to: {diff_plt_filename}", flush=True)
    else:
         print("\nSkipping comparison plots due to missing data.", flush=True)
    
    # Calculate overall statistics based on moves/s differences
    if moves_diffs:
        avg_diff = np.mean(moves_diffs)
        min_diff = np.min(moves_diffs)
        max_diff = np.max(moves_diffs)
        best_idx = moves_diffs.index(max_diff)
        worst_idx = moves_diffs.index(min_diff)
        
        print("\n=== Performance Change Summary (Moves/s vs Profile) ===", flush=True)
        print(f"Average performance change: {avg_diff:+.2f}%", flush=True)
        print(f"Best case (batch size {test_batch_sizes[best_idx]}): {max_diff:+.2f}%", flush=True)
        print(f"Worst case (batch size {test_batch_sizes[worst_idx]}): {min_diff:+.2f}%", flush=True)
        
        if avg_diff < -5:
            print("\n⚠️  WARNING: Current performance is significantly worse than the saved profile! ⚠️", flush=True)
        elif avg_diff > 5:
            print("\n✅ NOTICE: Current performance is significantly better than the saved profile!", flush=True)
        else:
            print("\n✓ Performance is within expected range of the saved profile.", flush=True)
    else:
        print("\nCould not calculate performance change summary.", flush=True)
    
    # Find optimal configurations based *only* on the current run's results
    if current_results:
        efficiencies = [r.efficiency for r in current_results]
        games_per_sec = [r.games_per_second for r in current_results]
        moves_per_sec = [r.moves_per_second for r in current_results]

        best_efficiency_idx = np.argmax(efficiencies)
        best_games_idx = np.argmax(games_per_sec) if any(g > 0 for g in games_per_sec) else 0
        fastest_idx = np.argmax(moves_per_sec)
        
        print("\n=== Optimal Configurations (Current Run) ===", flush=True)
        print(f"Best for moves/s: Batch size {current_results[fastest_idx].batch_size} "
              f"({format_human_readable(current_results[fastest_idx].moves_per_second)} moves/s)", flush=True)
        if any(g > 0 for g in games_per_sec):
            print(f"Best for games/s: Batch size {current_results[best_games_idx].batch_size} "
                  f"({format_human_readable(current_results[best_games_idx].games_per_second)} games/s)", flush=True)
        print(f"Best for efficiency: Batch size {current_results[best_efficiency_idx].batch_size} "
              f"({format_human_readable(current_results[best_efficiency_idx].efficiency)}/GB)", flush=True)
    else:
        print("\nCould not determine optimal configurations for the current run.", flush=True)


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the benchmark based on command line arguments."""
    # Setup signal handler for graceful interruption
    def signal_handler(sig, frame):
        print("\nBenchmark interrupted! Exiting gracefully...", flush=True)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Print system information
    system_info = get_system_info()
    print("\n=== System Information ===", flush=True)
    for key, value in system_info.items():
        print(f"{key}: {value}", flush=True)
    
    # Parse custom batch sizes if provided
    custom_batch_sizes = None
    if args.batch_sizes:
        try:
            custom_batch_sizes = parse_batch_sizes(args.batch_sizes)
            print(f"Using custom batch sizes: {custom_batch_sizes}", flush=True)
        except ValueError as e:
            print(str(e), flush=True)
            sys.exit(1)
    
    # Check if profile exists
    profile = load_profile()
    
    # Run in discover mode if requested or no profile exists
    if args.discover or profile is None:
        if profile is None:
            print("\nNo existing profile found, running in discovery mode...", flush=True)
        else:
            print("\nDiscovery mode explicitly requested", flush=True)
        
        # Run discovery with the specified parameters
        batch_sizes, moves_per_second, memory_usage_gb, games_per_second, moves_per_second_per_game = discover_optimal_batch_sizes(
            memory_limit_gb=args.memory_limit,
            max_duration=args.duration,
            batch_sizes=custom_batch_sizes
        )
        
        # Create and save profile
        print("\nCreating new benchmark profile...", flush=True)
        profile = BenchmarkProfile(
            platform=system_info["platform"],
            processor=system_info["processor"],
            jaxlib_type=system_info["jaxlib_type"],
            device_info=system_info["device_info"],
            python_version=system_info["python_version"],
            jax_version=system_info["jax_version"],
            batch_sizes=batch_sizes,
            moves_per_second=moves_per_second,
            memory_usage_gb=memory_usage_gb,
            games_per_second=games_per_second,
            moves_per_second_per_game=moves_per_second_per_game,
            duration=args.duration,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        save_profile(profile)
    else:
        # Run validation against existing profile
        print("\nProfile found - running validation mode", flush=True)
        validate_against_profile(profile, max_duration=args.duration, batch_sizes=custom_batch_sizes)


def main():
    """Main entry point for the benchmark script."""
    from benchmarks.benchmark_cli import create_benchmark_parser, parse_batch_sizes
    
    parser = create_benchmark_parser(
        description="Benchmark for pgx backgammon game environment",
        include_mcts_args=False
    )
    
    # Add game environment specific arguments
    parser.add_argument("--discover", action="store_true",
                       help="Force discovery mode even if profile exists")
    parser.add_argument("--target-game-count", type=int, default=1000,
                       help="Target number of games to complete for consistent statistics (default: 1000)")
    
    args = parser.parse_args()
    
    # If single batch mode is requested, just benchmark that batch size
    if args.single_batch:
        batch_size = args.single_batch
        print(f"Running single batch benchmark with batch size {batch_size}")
        print(f"Max duration: {args.duration}s")
        
        # Temporarily silence stdout for benchmark to avoid progress bar confusion
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w') if not args.verbose else sys.stdout
        try:
            result = benchmark_batch_size(
                batch_size, 
                max_duration=args.duration
            )
        finally:
            # Restore stdout regardless of success/failure
            if not args.verbose:
                sys.stdout.close()
                sys.stdout = original_stdout
        
        print("\nSingle Batch Benchmark Results Summary:")
        # Use the print_summary_table helper for consistent formatting
        print_summary_table([result], title=f"Batch Size {batch_size} Results")
    else:
        # Run the regular benchmark process
        run_benchmark(args)


if __name__ == "__main__":
    main() 