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
    select_batch_sizes_for_profile
)

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

# --- Stochastic MCTS Benchmark Function ---

def benchmark_batch_size_stochastic_mcts(
    batch_size: int,
    num_simulations: int,
    max_duration: int = DEFAULT_BENCHMARK_DURATION
) -> Tuple[BatchBenchResult, int]:  # Return the max node count as well
    """
    Benchmark StochasticMCTS evaluator with the Backgammon environment for a specific batch size.

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
    print(f"Benchmarking StochasticMCTS: Batch={batch_size}, Sims={num_simulations} for {max_duration}s", flush=True)
    print(f"{'-'*30}", flush=True)

    # --- Initialization ---
    print("Initializing environment and evaluator...", flush=True)
    env = bg.Backgammon(simple_doubles=True)
    # Get the actual action space size
    num_actions = env.num_actions
    print(f"Backgammon action space size: {num_actions}", flush=True)
    
    # Create PUCTSelector (PUCT is used in AlphaZero)
    action_selector = PUCTSelector(c=1.0)
    
    # Create StochasticMCTS evaluator
    evaluator = StochasticMCTS(
        eval_fn=dummy_apply_fn,  # Leaf evaluation function
        action_selector=action_selector,  # Action selection strategy
        stochastic_action_probs=env.stochastic_action_probs,  # Stochastic action probabilities
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
        """Create an env step function with a fixed key for StochasticMCTS."""
        def wrapped_step_fn(env_state, action, key):
            """Environment step function compatible with StochasticMCTS."""
            # Check if this is a stochastic state or deterministic state
            is_stochastic = env_state.is_stochastic if hasattr(env_state, 'is_stochastic') else False
            
            # Stochastic step (e.g., dice roll) vs deterministic step (e.g., move pieces)
            new_state = jax.lax.cond(
                is_stochastic,
                lambda s, a, k: env.stochastic_step(s, a),
                lambda s, a, k: env.step(s, a, k),
                env_state, action, key
            )
                
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

    def step_single_state_stochastic_mcts(key: chex.PRNGKey, env_state: chex.ArrayTree, eval_state: MCTSTree):
        """Steps a single environment using the StochasticMCTS evaluator."""
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

        # Determine if this is a stochastic state
        is_stochastic = env_state.is_stochastic if hasattr(env_state, 'is_stochastic') else False
        
        # Use conditional to call the appropriate evaluate method
        mcts_output = jax.lax.cond(
            is_stochastic,
            # For stochastic states, use stochastic_evaluate
            lambda: evaluator.stochastic_evaluate(
                key=eval_key,
                eval_state=eval_state,
                env_state=env_state,
                root_metadata=metadata,
                params=dummy_params,
                env_step_fn=current_env_step_fn
            ),
            # For deterministic states, use regular evaluate
            lambda: evaluator.evaluate(
                key=eval_key,
                eval_state=eval_state,
                env_state=env_state,
                root_metadata=metadata,
                params=dummy_params,
                env_step_fn=current_env_step_fn
            )
        )
        
        # Get selected action from MCTS
        action = mcts_output.action

        # Step the environment based on state type
        next_env_state = jax.lax.cond(
            is_stochastic,
            lambda: env.stochastic_step(env_state, action),
            lambda: env.step(env_state, action, step_key)
        )

        # Update MCTS state after action
        next_eval_state = evaluator.step(mcts_output.eval_state, action)

        # Return next states
        return next_env_state, next_eval_state


    def step_batch_with_reset_stochastic_mcts(key: chex.PRNGKey, env_states: chex.ArrayTree, eval_states: chex.ArrayTree):
        """Takes steps for a batch of parallel games, using StochasticMCTS, and resets terminated ones."""
        step_keys, reset_keys = jax.random.split(key, 2)
        batch_step_keys = jax.random.split(step_keys, batch_size)
        batch_reset_keys = jax.random.split(reset_keys, batch_size)

        # Vmap the single state step function across the batch
        vectorized_step = jax.vmap(step_single_state_stochastic_mcts, in_axes=(0, 0, 0))
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
    
    # Initialize each evaluator state separately and create a batch
    eval_states_list = []
    for i in range(batch_size):
        init_key, key = jax.random.split(key)
        eval_state = evaluator.init(template_embedding=template_env_state)
        eval_states_list.append(eval_state)
    
    # Convert list to a batched array tree - using pytree leaves directly
    if eval_states_list:
        # Use the pytree flatten/unflatten approach for more reliable batching
        from jax.tree_util import tree_flatten, tree_unflatten
        
        # Flatten all the trees
        flat_trees_with_defs = [tree_flatten(tree) for tree in eval_states_list]
        # Extract the leaves and tree definition (structure)
        flat_leaves = [leaves for leaves, _ in flat_trees_with_defs]
        # Use the first tree's definition since all trees should have the same structure
        tree_def = flat_trees_with_defs[0][1]
        
        # Stack corresponding leaves from each tree
        stacked_leaves = [jnp.stack(leaf_group) for leaf_group in zip(*flat_leaves)]
        
        # Recreate the tree structure with stacked leaves
        eval_states = tree_unflatten(tree_def, stacked_leaves)
    else:
        print("Warning: No evaluator states initialized!")
        raise ValueError("Failed to initialize evaluator states")

    # Trace and compile functions
    print("JIT-compiling functions...", flush=True)
    try:
        # Create a function to step the batch of environments
        def step_batch(k, states, eval_states):
            return step_batch_with_reset_stochastic_mcts(k, states, eval_states)
        
        # JIT the step function
        jitted_step = jax.jit(step_batch)
        
        # Warmup run to precompile
        print("Warmup compilation...", flush=True)
        key, warmup_key = jax.random.split(key, 2)
        env_states, eval_states, _ = jitted_step(warmup_key, env_states, eval_states)
        print("Compilation complete.", flush=True)
    except Exception as e:
        print(f"Error during compilation: {e}", flush=True)
        raise
    
    # --- Performance Benchmark ---
    print(f"Starting benchmark for batch size {batch_size}...", flush=True)
    
    # Create arrays to store metrics
    max_mem_usage = 0.0
    num_complete_steps = 0
    num_complete_games = 0
    step_times = []
    game_count = np.zeros(1, dtype=int)
    batch_nodes = []
    
    # Get initial memory usage
    start_mem = get_memory_usage()
    
    # Start timer
    start_time = time.time()
    benchmark_deadline = start_time + max_duration
    current_time = start_time
    
    # Main benchmark loop
    try:
        with tqdm(total=max_duration, desc=f"Batch {batch_size}", unit="s", disable=None) as pbar:
            while current_time < benchmark_deadline:
                # Track memory before step
                current_mem = get_memory_usage()
                max_mem_usage = max(max_mem_usage, current_mem)
                
                # Step start time
                step_start = time.time()
                
                # Step the batch forward
                key, step_key = jax.random.split(key, 2)
                env_states, eval_states, terminated = jitted_step(step_key, env_states, eval_states)
                
                # Ensure operations complete before measuring time
                jax.block_until_ready(env_states)
                jax.block_until_ready(eval_states)
                
                # Step end time
                step_end = time.time()
                step_time = step_end - step_start
                step_times.append(step_time)
                
                # Count terminated games (reset during this step)
                terminated_count = int(np.sum(jax.device_get(terminated)))
                num_complete_games += terminated_count
                game_count[0] += terminated_count
                
                # Count steps
                num_complete_steps += batch_size
                
                # Update progress bar
                current_time = time.time()
                elapsed = current_time - start_time
                pbar.update(min(elapsed, max_duration) - pbar.n)
                
                # Record highest observed node count in the tree
                # For simplicity, we check the first tree in the batch
                try:
                    eval_state_sample = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], eval_states))
                    if hasattr(eval_state_sample, 'next_free_idx'):
                        batch_nodes.append(int(eval_state_sample.next_free_idx))
                except Exception as e:
                    print(f"Error getting node count: {e}", flush=True)
    
    except KeyboardInterrupt:
        print("Benchmark interrupted!", flush=True)
    except Exception as e:
        print(f"Error during benchmark: {e}", flush=True)
        raise
    finally:
        # Calculate final metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate stats
        if step_times:
            mean_step_time = np.mean(step_times)
            p50_step_time = np.percentile(step_times, 50)
            p95_step_time = np.percentile(step_times, 95)
            p99_step_time = np.percentile(step_times, 99)
            steps_per_second = batch_size / mean_step_time
            # Only report non-zero games_per_second if games actually completed
            games_per_second = (num_complete_games / total_time) if total_time > 0 and num_complete_games > 0 else 0
        else:
            mean_step_time = p50_step_time = p95_step_time = p99_step_time = steps_per_second = games_per_second = 0
        
        max_node_count = max(batch_nodes) if batch_nodes else 0
        
        print(f"Batch {batch_size} benchmark complete:", flush=True)
        print(f"  Steps: {num_complete_steps}, Completed Games: {num_complete_games}", flush=True)
        print(f"  Avg Step Time: {mean_step_time:.6f}s, Steps/Sec: {steps_per_second:.2f}", flush=True)
        print(f"  Games/Sec: {games_per_second:.2f} (will be 0 until games complete)", flush=True)
        print(f"  Max Memory: {format_human_readable(max_mem_usage)}", flush=True)
        print(f"  Max Nodes: {max_node_count}", flush=True)
        
        # Compute additional stats needed for BatchBenchResult
        avg_game_length = 0 if num_complete_games == 0 else num_complete_steps / num_complete_games
        memory_gb = max_mem_usage / 1e9  # Convert bytes to GB
        efficiency = steps_per_second / batch_size  # Steps per second per parallel environment
        
        # Create result object using the correct fields
        result = BatchBenchResult(
            batch_size=batch_size,
            moves_per_second=steps_per_second,
            games_per_second=games_per_second,
            avg_game_length=avg_game_length,
            median_game_length=avg_game_length,  # We don't track individual game lengths
            min_game_length=0,  # We don't track individual game lengths
            max_game_length=0,  # We don't track individual game lengths
            memory_gb=memory_gb,
            efficiency=efficiency
        )
        
        return result, max_node_count

def discover_optimal_batch_sizes_stochastic_mcts(
    num_simulations: int,
    memory_limit_gb: float = DEFAULT_MEMORY_LIMIT_GB,
    max_duration: int = DEFAULT_BENCHMARK_DURATION,
    custom_batch_sizes: Optional[List[int]] = None
) -> Tuple[List[BatchBenchResult], int]:  # Return max node count as well
    """
    Discover optimal batch sizes for StochasticMCTS by running benchmarks at different batch sizes.

    Args:
        num_simulations: Number of MCTS simulations per move.
        memory_limit_gb: Memory limit in GB to stay under.
        max_duration: Maximum duration in seconds for each batch size benchmark.
        custom_batch_sizes: Optional custom list of batch sizes to benchmark.

    Returns:
        Tuple of (List of benchmark results for each batch size, max node count)
    """
    print(f"\n{'='*80}")
    print(f"DISCOVERING OPTIMAL BATCH SIZES FOR STOCHASTIC MCTS - {num_simulations} sims")
    print(f"{'='*80}")
    
    # Print system info
    system_info = get_system_info()
    print_system_info(system_info)
    
    # Get batch sizes to benchmark
    if custom_batch_sizes:
        batch_sizes = custom_batch_sizes
    else:
        # Use default batch sizes if no results available yet
        batch_sizes = [1, 2, 4, 8, 16]
    print(f"Testing batch sizes: {batch_sizes}")
    
    # Run benchmarks for each batch size
    all_results = []
    max_node_counts = {}  # Track max nodes for each batch size
    max_node_count = 0
    for batch_size in batch_sizes:
        # Check if we should skip due to memory concerns
        if all_results:
            mem_per_env = all_results[-1].memory_gb / all_results[-1].batch_size
            estimated_mem_gb = mem_per_env * batch_size
            if estimated_mem_gb > memory_limit_gb:
                print(f"Skipping batch size {batch_size} - estimated memory usage {estimated_mem_gb:.2f} GB exceeds limit")
                continue
        
        # Run benchmark
        try:
            result, nodes = benchmark_batch_size_stochastic_mcts(batch_size, num_simulations, max_duration)
            all_results.append(result)
            max_node_counts[batch_size] = nodes
            max_node_count = max(max_node_count, nodes)
        except Exception as e:
            print(f"Error benchmarking batch size {batch_size}: {e}")
            print("Stopping batch size discovery")
            break
            
    # Sort results by batch size
    all_results.sort(key=lambda r: r.batch_size)
    
    # Display custom summary with node counts
    print_stochastic_mcts_summary(all_results, max_node_counts)
    
    # Return results and max node count
    return all_results, max_node_count

def print_stochastic_mcts_summary(results: List[BatchBenchResult], max_node_counts: Dict[int, int]) -> None:
    """Print a summary of StochasticMCTS benchmark results including max node counts."""
    if not results:
        print("No valid benchmark results to display")
        return
    
    print("\n=== Discovery Summary (Valid Results) ===")
    header = "  Batch |      Moves/s |      Games/s |   Mem (GB) |  Max Nodes | Efficiency"
    print(header)
    print("-" * len(header))
    
    for result in results:
        batch_size = result.batch_size
        moves_per_second = result.moves_per_second
        games_per_second = result.games_per_second
        memory_gb = result.memory_gb
        efficiency = result.efficiency
        max_nodes = max_node_counts.get(batch_size, 0)
        
        print(f"{batch_size:^8} | {moves_per_second:^12.2f} | {games_per_second:^12.2f} | "
              f"{memory_gb:^10.2f} | {max_nodes:^11} | {efficiency:^10.2f}/GB")
    
    # Find optimal configurations
    best_moves_idx = max(range(len(results)), key=lambda i: results[i].moves_per_second)
    best_games_idx = max(range(len(results)), key=lambda i: results[i].games_per_second)
    best_efficiency_idx = max(range(len(results)), key=lambda i: results[i].efficiency)
    
    print("\n=== Optimal Configurations (Discovered) ===")
    print(f"Best for moves/s: Batch size {results[best_moves_idx].batch_size} with {results[best_moves_idx].moves_per_second:.2f} moves/s")
    print(f"Best for games/s: Batch size {results[best_games_idx].batch_size} with {results[best_games_idx].games_per_second:.2f} games/s")
    print(f"Best for efficiency: Batch size {results[best_efficiency_idx].batch_size} with {results[best_efficiency_idx].efficiency:.2f}/GB")
    print(f"Largest tree: Batch size {max(max_node_counts.items(), key=lambda x: x[1])[0]} with {max(max_node_counts.values())} nodes")

def save_stochastic_mcts_profile(
    results: List[BatchBenchResult], 
    num_simulations: int,
    system_info: Dict[str, str],
    max_duration: int,
    max_node_count: int = 0
) -> Path:
    """
    Save StochasticMCTS benchmark results to a profile file.

    Args:
        results: List of benchmark results for different batch sizes.
        num_simulations: Number of MCTS simulations used in benchmarks.
        system_info: System information dict.
        max_duration: Maximum duration used for each benchmark.
        max_node_count: Maximum number of nodes observed in any tree.

    Returns:
        Path to the saved profile file.
    """
    # Create a unique filename for StochasticMCTS profile
    platform_name = system_info["platform"].lower()
    processor_name = system_info["processor"].lower().replace('-', '_')
    jaxlib_type = system_info["jaxlib_type"]
    
    filename = f"{platform_name}_{processor_name}_{jaxlib_type}_stochastic_mcts_sims{num_simulations}.json"
    filepath = PROFILE_DIR / filename
    
    # Create profile data structure
    profile_data = {
        # System info
        "platform": system_info["platform"],
        "processor": system_info["processor"],
        "jaxlib_type": system_info["jaxlib_type"],
        "device_info": system_info["device_info"],
        "python_version": system_info["python_version"],
        "jax_version": system_info["jax_version"],
        
        # StochasticMCTS specific info
        "implementation": "StochasticMCTS",
        "num_simulations": num_simulations,
        "max_duration": max_duration,
        "max_nodes": MAX_NODES,
        "max_observed_nodes": max_node_count,
        
        # Benchmark results
        "batch_sizes": [r.batch_size for r in results],
        "moves_per_second": [r.moves_per_second for r in results],
        "games_per_second": [r.games_per_second for r in results],
        "memory_usage_gb": [r.memory_gb for r in results],
        
        # Timestamp
        "timestamp": datetime.now().isoformat()
    }
    
    # Save profile
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(profile_data, f, indent=2)
    
    print(f"Saved profile to {filepath}")
    
    # Extract data for plots
    batch_sizes = [r.batch_size for r in results]
    moves_per_second = [r.moves_per_second for r in results]
    games_per_second = [r.games_per_second for r in results]
    memory_usage_gb = [r.memory_gb for r in results]
    efficiency = [r.efficiency for r in results]
    
    # Generate plots using the extracted data
    try:
        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        plot_title = f"StochasticMCTS-{num_simulations}sims"
        plot_filename = f"{platform_name}_{processor_name}_{jaxlib_type}_stochastic_mcts_sims{num_simulations}"
        
        # Create basic plots
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, moves_per_second, 'o-', label="Moves/s")
        plt.title(f"{plot_title} - Performance")
        plt.xlabel("Batch Size")
        plt.ylabel("Moves per Second")
        plt.grid(True)
        plt.savefig(GRAPHS_DIR / f"{plot_filename}_performance.png")
        plt.close()
        
        # Create memory usage plot
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, memory_usage_gb, 'o-', label="Memory (GB)")
        plt.title(f"{plot_title} - Memory Usage")
        plt.xlabel("Batch Size")
        plt.ylabel("Memory (GB)")
        plt.grid(True)
        plt.savefig(GRAPHS_DIR / f"{plot_filename}_memory.png")
        plt.close()
        
        print(f"Plots saved to {GRAPHS_DIR}")
    except Exception as e:
        print(f"Error generating plots: {e}")
    
    return filepath

def load_stochastic_mcts_profile(num_simulations: int) -> Optional[Dict[str, Any]]:
    """
    Load a StochasticMCTS benchmark profile for the given number of simulations.

    Args:
        num_simulations: Number of MCTS simulations.

    Returns:
        Loaded profile data, or None if not found.
    """
    # Get system info for matching
    system_info = get_system_info()
    platform_name = system_info["platform"].lower()
    processor_type = "arm" if "arm" in system_info["processor"].lower() else system_info["processor"].split()[0].lower().replace('-', '_')
    jaxlib_type = system_info["jaxlib_type"]
    
    # Try to find an exact match first
    filename = f"{platform_name}_{processor_type}_{jaxlib_type}_stochastic_mcts_sims{num_simulations}.json"
    filepath = PROFILE_DIR / filename
    
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    
    if filepath.exists():
        print(f"Found matching StochasticMCTS profile: {filepath}", flush=True)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    print(f"No StochasticMCTS profile found for {num_simulations} simulations.", flush=True)
    return None

def validate_stochastic_mcts_against_profile(
    profile: Dict[str, Any],
    num_simulations: int,
    max_duration: int = DEFAULT_BENCHMARK_DURATION
) -> None:
    """
    Validate current system performance against a saved StochasticMCTS benchmark profile.

    Args:
        profile: Loaded profile data to validate against.
        num_simulations: Number of MCTS simulations.
        max_duration: Maximum duration for each benchmark.
    """
    print(f"\n{'='*80}")
    print(f"VALIDATING PERFORMANCE AGAINST STOCHASTIC MCTS PROFILE - {num_simulations} sims")
    print(f"{'='*80}")
    
    # Get system info for comparison
    system_info = get_system_info()
    print_system_info(system_info)
    
    # Print profile system info
    print("\nBenchmark Profile System:")
    print_system_info(profile["system_info"])
    
    # Load reference results
    reference_results = [
        BatchBenchResult(**result_dict) for result_dict in profile["results"]
    ]
    
    # Get batch sizes from reference
    batch_sizes = [result.batch_size for result in reference_results]
    
    # Run validation benchmarks
    validation_results = []
    max_node_count = 0
    for batch_size in batch_sizes:
        try:
            result, nodes = benchmark_batch_size_stochastic_mcts(
                batch_size, 
                num_simulations,
                max_duration // 2  # Use shorter duration for validation
            )
            validation_results.append(result)
            max_node_count = max(max_node_count, nodes)
        except Exception as e:
            print(f"Error validating batch size {batch_size}: {e}")
            break
    
    # Validate against reference results
    validate_against_profile(validation_results, reference_results, system_info)

def run_stochastic_mcts_benchmark(args: argparse.Namespace) -> None:
    """Run StochasticMCTS benchmark based on command-line arguments."""
    # Define SIGINT handler to exit gracefully
    def signal_handler(sig, frame):
        print("\nBenchmark interrupted. Exiting gracefully...")
        sys.exit(0)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check if profile exists
    profile = load_stochastic_mcts_profile(args.num_simulations)
    
    if args.validate and profile:
        # Validation mode - run against existing profile
        validate_stochastic_mcts_against_profile(
            profile,
            args.num_simulations,
            args.max_duration
        )
    else:
        # Benchmark mode - run new benchmarks
        if profile and not args.force:
            print(f"Profile for StochasticMCTS with {args.num_simulations} simulations already exists.")
            print("Use --force to overwrite, or --validate to validate against it.")
            return
        
        # Run benchmarks
        results, max_node_count = discover_optimal_batch_sizes_stochastic_mcts(
            args.num_simulations,
            args.memory_limit,
            args.max_duration,
            args.batch_sizes
        )
        
        # Save profile if benchmarks succeeded
        if results:
            save_stochastic_mcts_profile(
                results,
                args.num_simulations,
                get_system_info(),
                args.max_duration,
                max_node_count
            )

def main():
    """Command-line entry point for StochasticMCTS benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark StochasticMCTS performance")
    
    parser.add_argument("--num-simulations", type=int, default=DEFAULT_NUM_ITERATIONS,
                        help=f"Number of MCTS simulations per move (default: {DEFAULT_NUM_ITERATIONS})")
    parser.add_argument("--max-duration", type=int, default=DEFAULT_BENCHMARK_DURATION,
                        help=f"Maximum duration in seconds for each batch size (default: {DEFAULT_BENCHMARK_DURATION})")
    parser.add_argument("--memory-limit", type=float, default=DEFAULT_MEMORY_LIMIT_GB,
                        help=f"Memory limit in GB (default: {DEFAULT_MEMORY_LIMIT_GB})")
    parser.add_argument("--batch-sizes", type=int, nargs="*",
                        help="Custom batch sizes to benchmark (default: auto-selected)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate against existing benchmark profile")
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite of existing benchmark profile")
    
    args = parser.parse_args()
    
    run_stochastic_mcts_benchmark(args)

if __name__ == "__main__":
    main() 