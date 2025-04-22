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
    select_batch_sizes_for_profile
)

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

# --- MCTS Benchmark Function ---

def benchmark_batch_size_mcts(
    batch_size: int,
    num_simulations: int,
    max_duration: int = DEFAULT_BENCHMARK_DURATION
) -> Tuple[BatchBenchResult, int]:  # Return the max node count as well
    """
    Benchmark MCTS evaluator with the Backgammon environment for a specific batch size.

    Args:
        batch_size: Number of parallel environments/evaluators.
        num_simulations: Number of MCTS simulations per move.
        max_duration: Duration in seconds to run the benchmark.

    Returns:
        Tuple of (BatchBenchResult containing detailed performance metrics, max_node_count)
    """
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    print(f"\n{'-'*30}", flush=True)
    print(f"Benchmarking MCTS: Batch={batch_size}, Sims={num_simulations} for {max_duration}s", flush=True)
    print(f"{'-'*30}", flush=True)

    # --- Initialization ---
    print("Initializing environment and evaluator...", flush=True)
    env = bg.Backgammon(simple_doubles=True)
    # Get the actual action space size
    num_actions = env.num_actions
    print(f"Backgammon action space size: {num_actions}", flush=True)
    
    # Create PUCTSelector (PUCT is used in AlphaZero)
    action_selector = PUCTSelector(c=1.0)
    
    # Create MCTS evaluator
    evaluator = MCTS(
        eval_fn=dummy_apply_fn,  # Leaf evaluation function
        action_selector=action_selector,  # Action selection strategy
        branching_factor=num_actions,  # Use actual action space size
        max_nodes=MAX_NODES,  # Max nodes in tree
        num_iterations=num_simulations,  # Number of MCTS simulations
        discount=-1.0,  # Negative for 2-player games
        temperature=1.0,  # Temperature for action selection
        persist_tree=True  # Keep tree between steps
    )

    # Define a template env state for evaluator init
    try:
        # Get initial state from the environment
        sample_key = jax.random.PRNGKey(0)
        template_env_state = env.init(sample_key)
    except Exception as e:
        print(f"Error initializing environment: {e}")
        raise

    # --- Define Step Functions ---
    print("Defining step functions...", flush=True)

    # We need a wrapper function that will capture the key
    def make_env_step_fn(step_key):
        """Create an env step function with a fixed key for MCTS."""
        def wrapped_step_fn(env_state, action):
            """Environment step function compatible with MCTS."""
            new_state = env.step(env_state, action, step_key)
            # Create metadata for the MCTS
            metadata = StepMetadata(
                action_mask=new_state.legal_action_mask,
                terminated=new_state.terminated,
                rewards=new_state.rewards,
                cur_player_id=new_state.current_player,
                step=new_state._step_count if hasattr(new_state, '_step_count') else 0
            )
            return new_state, metadata
        return wrapped_step_fn

    def step_single_state_mcts(key: chex.PRNGKey, env_state: chex.ArrayTree, eval_state: MCTSTree):
        """Steps a single environment using the MCTS evaluator."""
        eval_key, action_key, step_key = jax.random.split(key, 3)

        # Get metadata for the current state
        metadata = StepMetadata(
            action_mask=env_state.legal_action_mask,
            terminated=env_state.terminated,
            rewards=env_state.rewards,
            cur_player_id=env_state.current_player,
            step=env_state._step_count if hasattr(env_state, '_step_count') else 0
        )

        # Create an env_step_fn with the current key
        current_env_step_fn = make_env_step_fn(step_key)

        # Evaluate state with MCTS
        mcts_output = evaluator.evaluate(
            key=eval_key,
            eval_state=eval_state,
            env_state=env_state,
            root_metadata=metadata,
            params=dummy_params,
            env_step_fn=current_env_step_fn
        )
        
        # Get selected action from MCTS
        action = mcts_output.action

        # Step the environment with the selected action
        next_env_state = env.step(env_state, action, step_key)

        # Update MCTS state after action
        next_eval_state = evaluator.step(mcts_output.eval_state, action)

        # Return next states
        return next_env_state, next_eval_state


    def step_batch_with_reset_mcts(key: chex.PRNGKey, env_states: chex.ArrayTree, eval_states: chex.ArrayTree):
        """Takes steps for a batch of parallel games, using MCTS, and resets terminated ones."""
        step_keys, reset_keys = jax.random.split(key, 2)
        batch_step_keys = jax.random.split(step_keys, batch_size)
        batch_reset_keys = jax.random.split(reset_keys, batch_size)

        # Vmap the single state step function across the batch
        vectorized_step = jax.vmap(step_single_state_mcts, in_axes=(0, 0, 0))
        next_env_states, next_eval_states = vectorized_step(batch_step_keys, env_states, eval_states)

        # Check termination status *after* the step
        terminated = next_env_states.terminated

        # Get newly initialized states for terminated environments
        vectorized_env_init = jax.vmap(env.init)
        reset_env_states = vectorized_env_init(batch_reset_keys)

        # Reset MCTS states for terminated environments
        vectorized_eval_reset = jax.vmap(evaluator.reset)
        reset_eval_states = vectorized_eval_reset(eval_states)

        # Conditionally select: if terminated, use reset_state, otherwise use next_state
        def where_terminated(next_s, reset_s):
            return jnp.where(
                terminated.reshape(-1, *([1]*(len(next_s.shape)-1))), 
                reset_s, 
                next_s
            )
            
        final_env_states = jax.tree_util.tree_map(
            where_terminated,
            next_env_states,
            reset_env_states
        )
        
        # Apply similar logic to eval states
        final_eval_states = jax.tree_util.tree_map(
            lambda next_s, reset_s: jnp.where(
                terminated.reshape(-1, *([1]*(len(next_s.shape)-1))), 
                reset_s, 
                next_s
            ),
            next_eval_states,
            reset_eval_states
        )

        return final_env_states, final_eval_states, terminated

    # --- Initialization and Compilation ---
    print(f"Initializing {batch_size} states...", flush=True)
    key = jax.random.PRNGKey(0)
    key, env_init_key = jax.random.split(key, 2)
    env_init_keys = jax.random.split(env_init_key, batch_size)

    # Initialize batch of env states
    env_states = jax.vmap(env.init)(env_init_keys)

    # Initialize batch of evaluator states
    eval_states = evaluator.init_batched(batch_size, template_env_state)

    # Compile the step function with integrated reset
    print("Compiling MCTS step function...", flush=True)
    step_fn = jax.jit(step_batch_with_reset_mcts)

    # --- Warm-up JIT compilation ---
    print("Compiling and warming up MCTS step...", flush=True)
    try:
        print("First compilation pass...", flush=True)
        key, subkey = jax.random.split(key)
        # Ensure step_fn is compiled before first use
        new_env_states, new_eval_states, _ = step_batch_with_reset_mcts(subkey, env_states, eval_states)
        # Block until ready
        jax.block_until_ready(new_env_states)
        jax.block_until_ready(new_eval_states)
        print("Initial compilation successful", flush=True)

        print("Running warm-up iterations...", flush=True)
        for _ in range(4):
            key, subkey = jax.random.split(key)
            new_env_states, new_eval_states, _ = step_batch_with_reset_mcts(subkey, new_env_states, new_eval_states)
        jax.block_until_ready(new_env_states)
        jax.block_until_ready(new_eval_states)
        print("Warm-up complete", flush=True)
        env_states = new_env_states
        eval_states = new_eval_states
        
        # Now that we've successfully compiled and tested the function, we can
        # safely use its jitted version for the actual benchmark
        step_fn = jax.jit(step_batch_with_reset_mcts)
    except Exception as e:
        print(f"Error during compilation/warm-up: {e}", flush=True)
        raise

    # --- Benchmark Loop ---
    print(f"Running MCTS benchmark for {max_duration} seconds...", flush=True)
    start_time = time.time()

    current_game_moves = np.zeros(batch_size, dtype=np.int32)
    total_moves = 0
    completed_games = 0
    game_lengths = []
    max_node_count = 0  # Track maximum number of nodes

    initial_memory_gb = get_memory_usage()
    peak_memory_gb = initial_memory_gb
    print(f"Initial memory usage: {initial_memory_gb:.2f}GB", flush=True)

    with tqdm(total=max_duration, desc=f"MCTS B={batch_size} S={num_simulations}", unit="s", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
        last_update_time = time.time()
        iteration_count = 0

        try:
            while (elapsed_time := time.time() - start_time) < max_duration:
                iteration_count += 1
                key, subkey = jax.random.split(key)

                # Execute the JIT-compiled step function
                env_states, eval_states, terminated_mask_jax = step_fn(subkey, env_states, eval_states)
                jax.block_until_ready(env_states) # Block until device operations complete

                # Convert JAX mask to NumPy for host-side logic
                terminated_mask_np = np.array(terminated_mask_jax)

                # Count moves for games active before this step
                active_mask = ~terminated_mask_np
                moves_this_step = np.sum(active_mask)
                total_moves += moves_this_step
                current_game_moves[active_mask] += 1

                # Try to get node count from eval_state if possible
                # For MCTS, we can extract the tree size (next_free_idx)
                try:
                    # Extract tree sizes across the batch
                    # This is a vectorized operation that gets the tree size for each item in the batch
                    for i in range(batch_size):
                        tree_size = eval_states.next_free_idx[i]
                        max_node_count = max(max_node_count, int(tree_size))
                except (AttributeError, ValueError) as e:
                    # If we can't access the field, just continue
                    pass

                # Process completed games
                if np.any(terminated_mask_np):
                    terminated_indices = np.where(terminated_mask_np)[0]
                    for i in terminated_indices:
                        completed_games += 1
                        game_length = current_game_moves[i] + 1 # Final step included
                        game_lengths.append(game_length)
                        current_game_moves[i] = 0 # Reset move counter for the new game

                # Update progress bar periodically
                current_time = time.time()
                if current_time - last_update_time > 0.5:
                    elapsed = current_time - start_time
                    pbar.n = round(elapsed)
                    moves_per_sec = total_moves / elapsed if elapsed > 0 else 0
                    games_per_sec = completed_games / elapsed if elapsed > 0 else 0
                    avg_moves = np.mean(game_lengths) if game_lengths else 0
                    current_memory = get_memory_usage()
                    peak_memory_gb = max(peak_memory_gb, current_memory)

                    pbar.set_postfix(
                        moves_s=f"{format_human_readable(moves_per_sec)}/s",
                        games_s=f"{format_human_readable(games_per_sec)}/s",
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

    # --- Final Calculations ---
    final_elapsed_time = time.time() - start_time
    moves_per_second = total_moves / final_elapsed_time if final_elapsed_time > 0 else 0
    games_per_second = completed_games / final_elapsed_time if final_elapsed_time > 0 else 0
    avg_game_length = np.mean(game_lengths) if game_lengths else 0
    median_game_length = np.median(game_lengths) if game_lengths else 0
    min_game_length = np.min(game_lengths) if game_lengths else 0
    max_game_length = np.max(game_lengths) if game_lengths else 0
    efficiency = moves_per_second / peak_memory_gb if peak_memory_gb > 0 else 0

    print("\nMCTS Benchmark complete!", flush=True)
    # --- Print Summary ---
    print(f"Batch Size: {batch_size}, Sims: {num_simulations}", flush=True)
    print(f"Total iterations (steps): {iteration_count}", flush=True)
    print(f"Total moves executed: {total_moves}", flush=True)
    print(f"Total completed games: {completed_games}", flush=True)
    print(f"Elapsed time: {final_elapsed_time:.2f}s (Target: {max_duration}s)", flush=True)

    print("--- Performance Metrics ---", flush=True)
    print(f"Moves per second: {format_human_readable(moves_per_second)}/s", flush=True)
    print(f"Games per second (completed): {format_human_readable(games_per_second)}/s", flush=True)
    print(f"Efficiency (Moves/s/GB): {format_human_readable(efficiency)}/GB", flush=True)
    print(f"Peak Memory Usage: {peak_memory_gb:.2f}GB", flush=True)
    print(f"Maximum MCTS nodes: {max_node_count}", flush=True)

    print("--- Game Length Statistics (Completed Games) ---", flush=True)
    print(f"Average moves per game: {avg_game_length:.2f}", flush=True)
    print(f"Median moves per game: {median_game_length:.1f}", flush=True)
    print(f"Minimum moves per game: {min_game_length if game_lengths else 'N/A'}", flush=True)
    print(f"Maximum moves per game: {max_game_length if game_lengths else 'N/A'}", flush=True)


    return (BatchBenchResult(
        batch_size=batch_size,
        moves_per_second=moves_per_second,
        games_per_second=games_per_second,
        avg_game_length=avg_game_length,
        median_game_length=median_game_length,
        min_game_length=min_game_length,
        max_game_length=max_game_length,
        memory_gb=peak_memory_gb,
        efficiency=efficiency,
        valid=True # Assuming valid unless OOM or error occurred
    ), max_node_count)


# --- Discovery and Validation Functions (Adapted for MCTS) ---

def discover_optimal_batch_sizes_mcts(
    num_simulations: int,
    memory_limit_gb: float = DEFAULT_MEMORY_LIMIT_GB,
    max_duration: int = DEFAULT_BENCHMARK_DURATION,
    custom_batch_sizes: Optional[List[int]] = None
) -> Tuple[List[BatchBenchResult], int]:  # Return max node count as well
    """Discover optimal batch sizes for MCTS by increasing batch size or using custom batch sizes.
    
    Args:
        num_simulations: Number of MCTS simulations per move.
        memory_limit_gb: Maximum memory limit in GB.
        max_duration: Duration in seconds for each batch size test.
        custom_batch_sizes: Optional list of specific batch sizes to test instead of auto-discovery.
    
    Returns:
        Tuple of (valid_results, max_node_count) with benchmark results and max node count.
    """
    print(f"\n=== Discovering optimal batch sizes for MCTS (Sims={num_simulations}, MemLimit={memory_limit_gb:.2f}GB) ===", flush=True)
    print(f"Duration per batch size: {max_duration}s", flush=True)

    all_results: List[BatchBenchResult] = []
    valid_results: List[BatchBenchResult] = []
    global_max_node_count = 0  # Track maximum node count across all benchmarks
    
    # Use custom batch sizes if provided, otherwise use the doubling strategy
    if custom_batch_sizes:
        batch_sizes_to_test = custom_batch_sizes
        print(f"Using custom batch sizes: {batch_sizes_to_test}", flush=True)
    else:
        # Start with batch size 1 and double
        batch_sizes_to_test = []
        batch_size = 1
        while True:
            batch_sizes_to_test.append(batch_size)
            # If we've reached an obviously excessive batch size, stop adding more
            if batch_size >= 2048:
                break
            batch_size *= 2
        print(f"Using default progressive batch sizes: {batch_sizes_to_test}", flush=True)
    
    last_perf_improvement = float('inf')

    with tqdm(desc="Discovering Batch Sizes (MCTS)", unit="batch") as outer_pbar:
        for batch_size in batch_sizes_to_test:
            try:
                print(f"\n{'#'*50}", flush=True)
                print(f"Discovery iteration {outer_pbar.n + 1}: Testing Batch={batch_size}, Sims={num_simulations}", flush=True)
                print(f"{'#'*50}", flush=True)

                # --- Run Benchmark ---
                original_stdout = sys.stdout # Silence inner tqdm
                sys.stdout = open(os.devnull, 'w')
                try:
                    result, max_node_count = benchmark_batch_size_mcts(
                        batch_size=batch_size,
                        num_simulations=num_simulations,
                        max_duration=max_duration
                    )
                    # Update the global max node count (we'll access this directly in a moment)
                    # Leave the stdout redirection intact until the except or finally block
                except Exception as e:
                    # Restore stdout to show error
                    sys.stdout.close()
                    sys.stdout = original_stdout
                    print(f"\nError during MCTS benchmark for batch size {batch_size}: {e}", flush=True)
                    # Check for OOM
                    if "RESOURCE_EXHAUSTED" in str(e) or "Out of memory" in str(e):
                        print(f"Out of memory error for batch size {batch_size}. Stopping discovery.", flush=True)
                        # Create an invalid result marker
                        result = BatchBenchResult(batch_size=batch_size, valid=False, moves_per_second=0, games_per_second=0, avg_game_length=0, median_game_length=0, min_game_length=0, max_game_length=0, memory_gb=memory_limit_gb*2, efficiency=0)
                        all_results.append(result) # Add invalid result marker
                        break # Stop discovery
                    else:
                        print("Non-OOM Error encountered. Stopping discovery.", flush=True)
                        raise # Re-raise other errors
                finally:
                    if not sys.stdout.closed: # Ensure stdout is restored
                        sys.stdout.close()
                        sys.stdout = original_stdout
                # --- End Benchmark Run ---

                # Update the global max node count based on this run's result
                global_max_node_count = max(global_max_node_count, max_node_count)

                # Print summary of the result
                print(f"Completed MCTS benchmark for Batch={batch_size}, Sims={num_simulations}: ", end="", flush=True)
                if result.valid:
                    print(f"{format_human_readable(result.moves_per_second)}/s, "
                          f"avg moves: {result.avg_game_length:.1f}, "
                          f"memory: {result.memory_gb:.2f}GB", flush=True)
                    valid_results.append(result)
                    all_results.append(result)
                else:
                    print("Result marked as invalid (likely OOM or error).", flush=True)
                    # If result is invalid (OOM), we already broke the loop


                outer_pbar.update(1)

                # Only check termination conditions when using auto-discovery (not custom batch sizes)
                if not custom_batch_sizes and result.valid:
                    if len(valid_results) > 1:
                        # Check memory limit
                        if result.memory_gb >= memory_limit_gb:
                            print(f"Memory limit reached: {result.memory_gb:.2f}GB >= {memory_limit_gb:.2f}GB", flush=True)
                            break

                        # Check diminishing returns
                        perf_improvement = valid_results[-1].moves_per_second / valid_results[-2].moves_per_second - 1.0 if valid_results[-2].moves_per_second > 0 else float('inf')
                        print(f"Performance improvement (moves/s): {perf_improvement:.2%}", flush=True)
                        if perf_improvement < 0.1:
                             if last_perf_improvement < 0.2: # Requires two consecutive small improvements
                                print("Diminishing returns detected. Stopping discovery.", flush=True)
                                break
                             last_perf_improvement = perf_improvement
                        else:
                            last_perf_improvement = float('inf') # Reset counter

            except Exception as e:
                # Catch errors from the outer loop/logic if any
                print(f"Unexpected error during discovery loop for batch size {batch_size}: {e}", flush=True)
                break # Stop discovery on unexpected errors

    print(f"\nMCTS Discovery complete: Tested {len(all_results)} batch sizes, {len(valid_results)} valid results.", flush=True)
    print(f"Maximum MCTS nodes observed across all benchmarks: {global_max_node_count}", flush=True)

    # Generate plots and summary using only valid results
    if valid_results:
        plot_batch_sizes = [r.batch_size for r in valid_results]
        
        # Include only test name (mcts) and simulation count in the plot filename, no timestamp
        plot_filename = f"mcts_sims{num_simulations}"
        
        perf_plot, mem_plot = generate_benchmark_plots(
            plot_batch_sizes, 
            valid_results, 
            timestamp=plot_filename,
            include_efficiency=False,  # Don't show efficiency in graphs
            include_cpu_usage=True     # Show CPU usage in memory graph
        )

        print_benchmark_summary(valid_results) # Use the common summary printer

        print(f"\nMCTS benchmark plots saved:")
        print(f"  Performance: {perf_plot}")
        print(f"  Memory: {mem_plot}")
    else:
        print("\nNo successful MCTS benchmark runs completed.", flush=True)

    return valid_results, global_max_node_count  # Return both results and max node count


def save_mcts_profile(
    results: List[BatchBenchResult], 
    num_simulations: int,
    system_info: Dict[str, str],
    max_duration: int,
    max_node_count: int = 0  # Add parameter to store max node count
) -> Path:
    """Save MCTS benchmark results to a profile file."""
    # Create a profile directory if it doesn't exist
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create a unique filename for MCTS profile
    platform_name = system_info["platform"].lower()
    processor_name = system_info["processor"].lower().replace('-', '_')
    jaxlib_type = system_info["jaxlib_type"]
    
    filename = f"{platform_name}_{processor_name}_{jaxlib_type}_mcts_sims{num_simulations}.json"
    filepath = PROFILE_DIR / filename
    
    # Select representative batch sizes from results
    # Using select_batch_sizes_for_profile from benchmark_common.py
    selected_batch_sizes = select_batch_sizes_for_profile(results, num_sizes=6)
    
    # Extract data for the selected batch sizes
    selected_results = [r for r in results if r.batch_size in selected_batch_sizes]
    selected_results.sort(key=lambda r: r.batch_size)
    
    # Create profile data
    profile_data = {
        # System info
        "platform": system_info["platform"],
        "processor": system_info["processor"],
        "jaxlib_type": system_info["jaxlib_type"],
        "device_info": system_info["device_info"],
        "python_version": system_info["python_version"],
        "jax_version": system_info["jax_version"],
        
        # MCTS specific info
        "num_simulations": num_simulations,
        "max_nodes": MAX_NODES,
        "max_node_count_observed": max_node_count,  # Add observed max node count
        
        # Benchmark results
        "batch_sizes": [r.batch_size for r in selected_results],
        "moves_per_second": [r.moves_per_second for r in selected_results],
        "games_per_second": [r.games_per_second for r in selected_results],
        "memory_usage_gb": [r.memory_gb for r in selected_results],
        "efficiency": [r.efficiency for r in selected_results],
        
        # Metadata
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration": max_duration
    }
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(profile_data, f, indent=2)
    
    print(f"MCTS profile saved to {filepath}")
    return filepath


def load_mcts_profile(num_simulations: int) -> Optional[Dict[str, Any]]:
    """Load MCTS profile matching the current system and simulation count."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get system info for matching
    system_info = get_system_info()
    platform_name = system_info["platform"].lower()
    processor_type = "arm" if "arm" in system_info["processor"].lower() else system_info["processor"].split()[0].lower().replace('-', '_')
    jaxlib_type = system_info["jaxlib_type"]
    
    # Try to find an exact match first
    filename = f"{platform_name}_{processor_type}_{jaxlib_type}_mcts_sims{num_simulations}.json"
    filepath = PROFILE_DIR / filename
    
    if filepath.exists():
        print(f"Found matching MCTS profile: {filepath}", flush=True)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    # If no exact match, try to find a profile with the same platform and backend but different simulations
    pattern = f"{platform_name}_{processor_type}_{jaxlib_type}_mcts_sims*.json"
    compatible_profiles = list(PROFILE_DIR.glob(pattern))
    
    if compatible_profiles:
        # Use the most recent profile
        compatible_profiles.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        filepath = compatible_profiles[0]
        print(f"Using compatible MCTS profile: {filepath}", flush=True)
        print(f"Note: This profile uses a different number of simulations than requested.", flush=True)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    print(f"No compatible MCTS profile found for {num_simulations} simulations.", flush=True)
    return None


def validate_mcts_against_profile(
    profile: Dict[str, Any],
    num_simulations: int,
    max_duration: int = DEFAULT_BENCHMARK_DURATION
) -> None:
    """Compare current MCTS performance against a previous profile."""
    print(f"\n=== Validating MCTS Performance (Sims={num_simulations}) Against Profile ===", flush=True)
    
    if 'batch_sizes' not in profile or 'moves_per_second' not in profile:
        print("Profile doesn't contain required data for comparison", flush=True)
        return
    
    if profile.get('num_simulations', 0) != num_simulations:
        print(f"Warning: Profile uses {profile.get('num_simulations', 'unknown')} simulations, but validating with {num_simulations}", flush=True)
    
    # Run benchmarks for each batch size in the profile
    current_results = []
    batch_sizes = profile['batch_sizes']
    
    for batch_size in batch_sizes:
        print(f"\nValidating batch size {batch_size} with {num_simulations} simulations...")
        try:
            result = benchmark_batch_size_mcts(
                batch_size=batch_size,
                num_simulations=num_simulations,
                max_duration=max_duration
            )
            current_results.append(result)
        except Exception as e:
            print(f"Error benchmarking batch size {batch_size}: {e}", flush=True)
            # Continue with other batch sizes
    
    if not current_results:
        print("No valid benchmark results collected for comparison.", flush=True)
        return
    
    # Compare results
    print("\n=== Comparison Results ===", flush=True)
    print(f"{'Batch Size':^10} | {'Profile Moves/s':^15} | {'Current Moves/s':^15} | {'Diff %':^10}", flush=True)
    print("-" * 60, flush=True)
    
    diffs = []
    
    for result in current_results:
        batch_size = result.batch_size
        if batch_size in batch_sizes:
            idx = batch_sizes.index(batch_size)
            profile_moves = profile['moves_per_second'][idx]
            current_moves = result.moves_per_second
            diff_pct = ((current_moves - profile_moves) / profile_moves) * 100
            diffs.append(diff_pct)
            
            print(f"{batch_size:^10} | {profile_moves:^15,.2f} | {current_moves:^15,.2f} | {diff_pct:^+10.2f}%", flush=True)
    
    # Overall assessment
    if diffs:
        avg_diff = sum(diffs) / len(diffs)
        print(f"\nAverage performance difference: {avg_diff:+.2f}%", flush=True)
        
        if avg_diff > 5:
            print("Performance has IMPROVED compared to profile", flush=True)
        elif avg_diff < -5:
            print("Performance has DEGRADED compared to profile", flush=True)
        else:
            print("Performance is similar to profile", flush=True)
    
    # Generate comparison plots
    if current_results:
        system_info = get_system_info()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        validate_against_profile(current_results, profile, system_info)


def run_mcts_benchmark(args: argparse.Namespace) -> None:
    """Run the MCTS benchmark based on command line arguments."""
    # Setup signal handler
    def signal_handler(sig, frame):
        print("\nMCTS Benchmark interrupted! Exiting gracefully...", flush=True)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # Print system information
    system_info = get_system_info()
    print_system_info(system_info)
    
    # Get num_simulations from args
    num_simulations = args.num_simulations
    
    # Initialize max_node_count variable
    max_node_count = 0
    
    # Parse custom batch sizes if provided
    custom_batch_sizes = None
    if args.batch_sizes:
        try:
            # Split by comma, strip whitespace, and convert to integers
            custom_batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(',')]
            print(f"Using custom batch sizes: {custom_batch_sizes}", flush=True)
        except ValueError as e:
            print(f"Error parsing batch sizes: {e}. Format should be comma-separated integers.", flush=True)
            print("Example: --batch-sizes 1,2,4,8,16", flush=True)
            sys.exit(1)

    # --- Mode Selection ---
    if args.validate:
        # Validation mode - compare against existing profile
        print(f"\nRunning MCTS Validation Mode (Sims={num_simulations})", flush=True)
        profile = load_mcts_profile(num_simulations)
        if profile:
            validate_mcts_against_profile(profile, num_simulations, args.duration)
        else:
            print("No profile found for validation. Running discovery mode instead.", flush=True)
            results, max_node_count = discover_optimal_batch_sizes_mcts(
                num_simulations=num_simulations,
                memory_limit_gb=args.memory_limit,
                max_duration=args.duration,
                custom_batch_sizes=custom_batch_sizes
            )
            if results:
                save_mcts_profile(results, num_simulations, system_info, args.duration, max_node_count=max_node_count)
    
    elif args.single_batch:
        # Run only a single batch size
        print(f"\nRunning single MCTS benchmark: Batch={args.single_batch}, Sims={num_simulations}, Duration={args.duration}s")
        try:
            result, batch_max_node_count = benchmark_batch_size_mcts(
                batch_size=args.single_batch,
                num_simulations=num_simulations,
                max_duration=args.duration
            )
            print_benchmark_summary([result])
            # Update the overall max node count
            max_node_count = max(max_node_count, batch_max_node_count)
        except Exception as e:
            print(f"\nError during single MCTS benchmark: {e}", flush=True)
    
    else:
        # Run discovery mode
        print(f"\nRunning MCTS Discovery Mode (Sims={num_simulations})", flush=True)
        results, max_node_count = discover_optimal_batch_sizes_mcts(
            num_simulations=num_simulations,
            memory_limit_gb=args.memory_limit,
            max_duration=args.duration,
            custom_batch_sizes=custom_batch_sizes
        )
        # Save profile with results if we have valid results
        if results:
            save_mcts_profile(results, num_simulations, system_info, args.duration, max_node_count=max_node_count)


def main():
    """Main entry point for the MCTS benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark for MCTS Evaluator with pgx Backgammon")
    parser.add_argument("--num-simulations", type=int, default=DEFAULT_NUM_ITERATIONS,
                        help=f"Number of MCTS simulations per move (default: {DEFAULT_NUM_ITERATIONS})")
    parser.add_argument("--memory-limit", type=float, default=DEFAULT_MEMORY_LIMIT_GB,
                        help=f"Memory limit in GB for discovery mode (default: {DEFAULT_MEMORY_LIMIT_GB})")
    parser.add_argument("--duration", type=int, default=DEFAULT_BENCHMARK_DURATION,
                        help=f"Maximum duration of each batch size test in seconds (default: {DEFAULT_BENCHMARK_DURATION})")
    parser.add_argument("--single-batch", type=int, help="Test only a specific batch size instead of discovery")
    parser.add_argument("--validate", action="store_true", help="Validate against existing profiles")
    parser.add_argument("--batch-sizes", type=str, help="Comma-separated list of batch sizes to test (e.g., '1,2,4,8,16')")

    args = parser.parse_args()
    run_mcts_benchmark(args)


if __name__ == "__main__":
    main() 