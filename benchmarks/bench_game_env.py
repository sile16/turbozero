#!/usr/bin/env python3
"""
Benchmark for the pgx backgammon game environment.

This script measures the performance of the backgammon environment in terms of moves per second
for different batch sizes. It includes a discovery mode to find optimal batch sizes and 
a validation mode to verify performance against previous runs.
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
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

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


# Constants
DEFAULT_MEMORY_LIMIT_GB = 16  # Maximum memory to use (in GB)
DEFAULT_BENCHMARK_DURATION = 30  # Duration of each batch size test in seconds
PROFILE_DIR = Path("benchmarks/profiles")
GRAPHS_DIR = Path("benchmarks/graphs")
HUMAN_READABLE_UNITS = ["", "K", "M", "B", "T"]


@dataclass
class BenchmarkProfile:
    """Profile containing hardware information and benchmark results."""
    # Hardware info
    platform: str
    processor: str
    jaxlib_type: str  # cpu, cuda, metal
    device_info: str
    python_version: str
    jax_version: str
    
    # Benchmark results
    batch_sizes: List[int]
    moves_per_second: List[float]
    memory_usage_gb: List[float]
    timestamp: str = None  # Creation date
    duration: int = DEFAULT_BENCHMARK_DURATION  # Duration used for each benchmark in seconds
    
    # Optional metrics for backward compatibility
    games_per_second: List[float] = None  
    moves_per_second_per_game: List[float] = None
    
    def __post_init__(self):
        # Set timestamp if none provided
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


def format_human_readable(num: float) -> str:
    """Format a number in human-readable form with appropriate units."""
    idx = 0
    while abs(num) >= 1000 and idx < len(HUMAN_READABLE_UNITS) - 1:
        num /= 1000
        idx += 1
    return f"{num:.2f}{HUMAN_READABLE_UNITS[idx]}"


def get_system_info() -> Dict[str, str]:
    """Get system information for the profile."""
    system_info = {
        "platform": platform.system(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__
    }
    
    # Determine JAXlib type (CPU, CUDA, Metal)
    jaxlib_type = "cpu"
    if jax.default_backend() == "gpu":
        jaxlib_type = "cuda"
    elif jax.default_backend() == "metal":
        jaxlib_type = "metal"
    system_info["jaxlib_type"] = jaxlib_type
    
    # Get detailed device info
    devices = jax.devices()
    if devices:
        system_info["device_info"] = str(devices[0])
    else:
        system_info["device_info"] = "No devices found"
    
    return system_info


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    if platform.system() == "Darwin":
        # macOS - use ps
        cmd = "ps -o rss= -p " + str(os.getpid())
        output = subprocess.check_output(cmd.split()).decode().strip()
        mem_kb = float(output)
        return mem_kb / 1024 / 1024  # Convert KB to GB
    elif platform.system() == "Linux":
        # Linux - use /proc/self/status
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    mem_kb = float(line.split()[1])
                    return mem_kb / 1024 / 1024  # Convert KB to GB
    
    # Default fallback (less accurate)
    import psutil
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / 1024 / 1024 / 1024  # Convert bytes to GB


def create_profile_filename() -> str:
    """Create a unique filename for the profile based on system info."""
    sys_info = get_system_info()
    platform_name = sys_info["platform"].lower()
    
    # Simplify processor info - just use "arm" for Apple Silicon
    if "arm" in sys_info["processor"].lower():
        processor_id = "arm"
    else:
        # For other processors, simplify to avoid special characters
        processor_id = sys_info["processor"].split()[0].lower().replace("-", "_")
    
    jaxlib_type = sys_info["jaxlib_type"]
    
    filename = f"{platform_name}_{processor_id}_{jaxlib_type}.json"
    print(f"Profile filename: {filename}", flush=True)
    return filename


def save_profile(profile: BenchmarkProfile) -> Path:
    """Save benchmark profile to a JSON file."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    
    filename = create_profile_filename()
    filepath = PROFILE_DIR / filename
    
    with open(filepath, 'w') as f:
        json.dump(asdict(profile), f, indent=2)
    
    print(f"Profile saved to {filepath}")
    return filepath


def load_profile() -> Optional[BenchmarkProfile]:
    """Load benchmark profile from a JSON file if it exists."""
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    
    # List all profiles in the directory
    all_profiles = list(PROFILE_DIR.glob("*.json"))
    print(f"Found {len(all_profiles)} profile(s) in {PROFILE_DIR}:", flush=True)
    for profile_path in all_profiles:
        print(f"  - {profile_path.name}", flush=True)
    
    # Try to find a match with the generated filename
    filename = create_profile_filename()
    filepath = PROFILE_DIR / filename
    
    if not filepath.exists():
        print(f"Warning: Could not find profile at {filepath}", flush=True)
        print("Trying to find a compatible profile...", flush=True)
        
        # If exact match not found, try to find a profile for the same platform and backend
        sys_info = get_system_info()
        platform_name = sys_info["platform"].lower()
        jaxlib_type = sys_info["jaxlib_type"]
        
        pattern = f"{platform_name}_*_{jaxlib_type}.json"
        compatible_profiles = list(PROFILE_DIR.glob(pattern))
        
        if compatible_profiles:
            # Use the most recent profile
            compatible_profiles.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            filepath = compatible_profiles[0]
            print(f"Using compatible profile: {filepath.name}", flush=True)
        else:
            print(f"No compatible profiles found for pattern: {pattern}", flush=True)
            return None
    
    print(f"Loading profile from {filepath}", flush=True)
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle missing duration field for backward compatibility
        if 'duration' not in data:
            data['duration'] = DEFAULT_BENCHMARK_DURATION
            print(f"Note: Adding default duration ({DEFAULT_BENCHMARK_DURATION}s) to loaded profile", flush=True)
        
        return BenchmarkProfile(**data)
    except Exception as e:
        print(f"Error loading profile: {e}", flush=True)
        return None


def random_action_from_mask(key, mask):
    """Sample a random action based on a legal action mask"""
    # Create a probability distribution over legal actions
    action_probs = jnp.where(mask, 1.0, 0.0)
    # Normalize (add small epsilon to avoid division by zero)
    action_probs = action_probs / (jnp.sum(action_probs) + 1e-10)
    # Sample from the distribution
    return jax.random.categorical(key, jnp.log(action_probs + 1e-10))


def benchmark_batch_size(batch_size: int, duration: int = DEFAULT_BENCHMARK_DURATION) -> Tuple[float, float, float, float]:
    """
    Benchmark the backgammon environment with a specific batch size.
    
    Args:
        batch_size: Batch size to test
        duration: Duration of the test in seconds
        
    Returns:
        Tuple of (moves_per_second, memory_usage_gb, games_per_second, moves_per_second_per_game)
    """
    print(f"\n{'-'*30}", flush=True)
    print(f"Benchmarking batch size: {batch_size} for {duration}s", flush=True)
    print(f"{'-'*30}", flush=True)
    
    # Initialize environment
    print("Initializing environment...", flush=True)
    env = bg.Backgammon(simple_doubles=True)
    num_stochastic_outcomes = len(env.stochastic_action_probs)
    
    # ---- Define step functions (non-jitted) ----
    print("Defining step functions...", flush=True)
    def step_single_state(key, state):
        """Step a single state."""
        # Split key for potential use in both branches
        key1, key2 = jax.random.split(key)
        
        # Get the stochastic flag as a boolean scalar - fix indexing error
        is_stochastic = jnp.bool_(state.is_stochastic)
        # Get the legal action mask from the state directly
        legal_actions_mask = state.legal_action_mask
        
        # Use jax.lax.cond to handle the conditional logic in a JIT-compatible way
        return jax.lax.cond(
            is_stochastic,
            # True branch: stochastic step
            lambda _: env.stochastic_step(state, jax.random.randint(key1, (), 0, num_stochastic_outcomes)),
            # False branch: deterministic step
            lambda _: env.step(state, random_action_from_mask(key2, legal_actions_mask), key2),
            operand=None
        )
    
    # ---- Initialize states based on batch size ----
    print("Initializing states...", flush=True)
    key = jax.random.PRNGKey(0)
    
    if batch_size == 1:
        # Single state case
        print("Using single state mode", flush=True)
        key, init_key = jax.random.split(key)
        state = env.init(init_key)
        print("Single state initialized", flush=True)
        
        # Compile the step function for single state
        print("Compiling step function for single state...", flush=True)
        step_fn = jax.jit(step_single_state)
    else:
        # For batched states, create a vmap'd function
        print(f"Using batched mode with {batch_size} parallel states", flush=True)
        def step_batch_states(key, states):
            """Take steps for a batch of states"""
            batch_keys = jax.random.split(key, batch_size)
            return jax.vmap(step_single_state)(batch_keys, states)
        
        # Initialize batch of states
        print(f"Initializing {batch_size} states...", flush=True)
        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, batch_size)
        states = []
        for i in range(batch_size):
            states.append(env.init(init_keys[i]))
        
        print("Converting states to batched JAX arrays...", flush=True)
        # Convert to JAX arrays
        state = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
        
        # Compile the step function for batched states
        print("Compiling step function for batched states...", flush=True)
        step_fn = jax.jit(step_batch_states)
    
    # ---- Warm-up JIT compilation ----
    print("Compiling and warming up...", flush=True)
    try:
        # First pass to trigger compilation
        print("First compilation pass...", flush=True)
        key, subkey = jax.random.split(key)
        new_state = step_fn(subkey, state)
        jax.block_until_ready(new_state)  # Ensure compilation completes
        print("Initial compilation successful", flush=True)
        
        # Additional warm-up iterations
        print("Running warm-up iterations...", flush=True)
        for i in range(4):
            key, subkey = jax.random.split(key)
            new_state = step_fn(subkey, new_state)
        jax.block_until_ready(new_state)
        print("Warm-up complete", flush=True)
        state = new_state
    except Exception as e:
        print(f"Error during compilation: {e}", flush=True)
        raise
    
    # ---- Benchmark ----
    print(f"Running benchmark for ~{duration} seconds...", flush=True)
    start_time = time.time()
    
    # Track overall statistics
    total_moves = 0
    completed_games = 0
    
    # Better tracking for game length statistics
    total_moves_in_completed_games = 0  # Sum of moves in all completed games
    max_moves_per_game = 0
    min_moves_per_game = float('inf')  # Initialize to infinity to find minimum
    single_move_games = 0  # Count games that terminate after just 1 move
    game_lengths = []  # Store all game lengths for calculating median
    
    # Track per-instance state
    current_game_moves = [0] * batch_size
    
    # Track memory usage
    memory_usage_gb = get_memory_usage()
    print(f"Initial memory usage: {memory_usage_gb:.2f}GB", flush=True)
    
    # Create a tqdm progress bar that updates based on time
    # Using a total of 100 for percentage calculation
    print("Starting benchmark loop:", flush=True)
    
    # Use a total of 100 for simple percentage display
    with tqdm(total=100, desc="Benchmarking", unit="%", bar_format='{l_bar}{bar}| {n:3.0f}%') as pbar:
        last_update_time = time.time()
        iteration_count = 0
        
        try:
            while (time.time() - start_time) < duration:
                iteration_count += 1
                key, subkey = jax.random.split(key)
                
                # Execute the step function
                state = step_fn(subkey, state)
                # Force completion of the operation
                jax.block_until_ready(state)
                
                # Count one move per state in the batch
                total_moves += batch_size
                
                # Update moves per game counters
                for i in range(batch_size):
                    current_game_moves[i] += 1
                
                # Check for completed games (assuming terminated flag indicates completed game)
                if batch_size == 1:
                    if state.terminated:
                        completed_games += 1
                        # Record the game length
                        game_length = current_game_moves[0]
                        total_moves_in_completed_games += game_length
                        max_moves_per_game = max(max_moves_per_game, game_length)
                        min_moves_per_game = min(min_moves_per_game, game_length)
                        game_lengths.append(game_length)
                        # Reset counter for the next game
                        current_game_moves[0] = 0
                        # Reinitialize for a new game
                        key, init_key = jax.random.split(key)
                        state = env.init(init_key)
                else:
                    # Process terminations in batched mode
                    terminated_states = state.terminated
                    # Convert to numpy for iteration
                    terminated_np = np.array(terminated_states)
                    
                    # Check if any games have terminated
                    term_count = np.sum(terminated_np)
                    if term_count > 0:
                        # Process each terminated game for statistics
                        for i in range(batch_size):
                            if terminated_np[i]:
                                completed_games += 1
                                # Record the game length before resetting the counter
                                game_length = current_game_moves[i]
                                
                                # Only count games with actual moves
                                if game_length > 0:
                                    total_moves_in_completed_games += game_length
                                    max_moves_per_game = max(max_moves_per_game, game_length)
                                    min_moves_per_game = min(min_moves_per_game, game_length)
                                    game_lengths.append(game_length)
                                    
                                    # Track single-move games
                                    if game_length == 1:
                                        single_move_games += 1
                                
                                # Reset counter for this game instance
                                current_game_moves[i] = 0
                        
                        # We need to reinitialize terminated games
                        # This is a critical improvement over the original code
                        
                        # Create replacement states for terminated games
                        key, init_key = jax.random.split(key)
                        
                        # The correct approach would be to use jax.lax.cond or similar to 
                        # conditionally replace states, but for benchmark purposes, we'll
                        # simply reinitialize ALL states once any game terminates
                        # This provides more accurate statistics by avoiding counting
                        # terminated states multiple times
                        
                        # Reset all states in the batch - this is a simple but effective solution
                        init_keys = jax.random.split(init_key, batch_size)
                        new_states = []
                        for i in range(batch_size):
                            new_states.append(env.init(init_keys[i]))
                        
                        # Convert back to batched JAX arrays
                        state = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *new_states)
                
                # Update progress bar and metrics (every 0.1 seconds to avoid overhead)
                current_time = time.time()
                if current_time - last_update_time > 0.1:
                    elapsed = current_time - start_time
                    progress_pct = min(100, (elapsed / duration) * 100)
                    
                    # Calculate performance metrics
                    moves_per_sec = total_moves / elapsed
                    games_per_sec = completed_games / elapsed if completed_games > 0 else 0
                    
                    # Calculate average moves per completed game
                    avg_moves = total_moves_in_completed_games / completed_games if completed_games > 0 else 0
                    
                    # Update peak memory usage
                    current_memory = get_memory_usage()
                    memory_usage_gb = max(memory_usage_gb, current_memory)
                    
                    # Update progress bar position directly based on percentage
                    pbar.n = int(progress_pct)
                    
                    # Update progress bar with metrics in the description
                    pbar.set_description(
                        f"Batch {batch_size}: {format_human_readable(moves_per_sec)}/s, "
                        f"Games: {format_human_readable(games_per_sec)}/s, "
                        f"Avg: {avg_moves:.1f}, Min: {min_moves_per_game if min_moves_per_game != float('inf') else 0}, Max: {max_moves_per_game}"
                    )
                    pbar.refresh()  # Force refresh
                    
                    last_update_time = current_time
                
        except KeyboardInterrupt:
            print("\nBenchmark interrupted!", flush=True)
    
    # Calculate final results
    elapsed_time = time.time() - start_time
    moves_per_second = total_moves / elapsed_time
    games_per_second = completed_games / elapsed_time if completed_games > 0 else 0
    moves_per_sec_per_game = moves_per_second / batch_size
    
    # Calculate average moves per game, avoiding division by zero
    if completed_games > 0 and total_moves_in_completed_games > 0:
        avg_moves_per_game = total_moves_in_completed_games / completed_games
    else:
        avg_moves_per_game = 0
        
    # Handle the case where no valid games were completed
    if min_moves_per_game == float('inf'):
        min_moves_per_game = 0
    
    # When doing multiple small batches, we might get low average moves if many games terminate immediately
    # Only report an average if it seems reasonable
    if avg_moves_per_game < 1 and batch_size > 1 and max_moves_per_game > 10:
        print("\nWarning: Very low average moves per game detected. Some games may be terminating prematurely.", flush=True)
    
    # Calculate median if we have game lengths
    median_moves = 0
    if game_lengths:
        game_lengths.sort()
        if len(game_lengths) % 2 == 0:
            median_moves = (game_lengths[len(game_lengths)//2 - 1] + game_lengths[len(game_lengths)//2]) / 2
        else:
            median_moves = game_lengths[len(game_lengths)//2]
    
    # Print batch mode explanation if needed
    if batch_size > 1 and completed_games > 0:
        single_move_percentage = (single_move_games / completed_games) * 100
        if single_move_percentage > 50:
            print("\nNote: In batch mode, many games (%.1f%%) terminated after just 1 move." % single_move_percentage)
            print("This is expected behavior in the pgx backgammon environment when using batched evaluation.")
    
    print("\nBenchmark complete!", flush=True)
    print(f"Total iterations: {iteration_count}", flush=True)
    print(f"Total moves: {total_moves}", flush=True)
    print(f"Total completed games: {completed_games}", flush=True)
    print(f"Total moves in completed games: {total_moves_in_completed_games}", flush=True)
    print(f"Average moves per game: {avg_moves_per_game:.2f}", flush=True)
    print(f"Median moves per game: {median_moves:.1f}", flush=True)
    print(f"Minimum moves per game: {min_moves_per_game}", flush=True)
    print(f"Maximum moves per game: {max_moves_per_game}", flush=True)
    print(f"Single-move games: {single_move_games} ({single_move_percentage:.1f}% of total)" if batch_size > 1 else "")
    print(f"Elapsed time: {elapsed_time:.2f}s", flush=True)
    print(f"Batch size {batch_size}: {format_human_readable(moves_per_second)} moves/s, "
          f"{format_human_readable(moves_per_sec_per_game)} moves/s/game, "
          f"{format_human_readable(games_per_second)} games/s, "
          f"Memory: {memory_usage_gb:.2f}GB", flush=True)
    
    # Print a note if a significant percentage of games terminated after just 1 move
    if completed_games > 0 and single_move_games > 0:
        pct_single_move = (single_move_games / completed_games) * 100
        if pct_single_move > 50:
            print(f"Single-move games: {single_move_games:,} ({pct_single_move:.1f}% of total)")
            if batch_size > 1:
                print("\nNOTE: The high percentage of single-move games in batch mode is expected behavior.")
                print("In the PGX backgammon environment, when using batched evaluation:")
                print("1. All games in a batch start simultaneously")
                print("2. When any game in the batch terminates, it's counted as completed")
                print("3. The terminated game is reset, but stats are already recorded")
                print("4. This leads to many quick terminations being recorded as complete games")
                print("5. For accurate game length statistics, use batch_size=1\n")
    
    return moves_per_second, memory_usage_gb, games_per_second, moves_per_sec_per_game


def generate_benchmark_plots(batch_sizes, metrics_data, timestamp=None):
    """
    Generate and save plots of benchmark results.
    
    Args:
        batch_sizes: List of batch sizes tested
        metrics_data: Dictionary of metric names to lists of values
        timestamp: Optional timestamp to include in filenames
    """
    # Create graphs directory if it doesn't exist
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set the timestamp for filenames
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create figure with log-log scale for better visualization
    plt.figure(figsize=(12, 8))
    
    # Plot batch size vs moves per second
    metrics_to_plot = [
        ('moves_per_second', 'Moves per Second', 'tab:blue'),
        ('moves_per_second_per_game', 'Moves per Second per Game', 'tab:green'),
        ('games_per_second', 'Games per Second', 'tab:red'),
    ]
    
    # Use log-log scale for better visualization of wide range of values
    plt.xscale('log', base=2)  # Log scale with base 2 for batch sizes
    plt.yscale('log')  # Log scale for metrics
    
    # Plot each metric
    for metric_name, label, color in metrics_to_plot:
        if metric_name in metrics_data and metrics_data[metric_name]:
            plt.plot(batch_sizes, metrics_data[metric_name], 
                    marker='o', linestyle='-', linewidth=2, 
                    label=label, color=color)
    
    # Add efficiency as a separate plot with twin axis
    if 'efficiency' in metrics_data and metrics_data['efficiency']:
        ax2 = plt.twinx()
        ax2.set_yscale('log')
        ax2.plot(batch_sizes, metrics_data['efficiency'], 
                marker='s', linestyle='--', linewidth=2,
                label='Efficiency (moves/s/GB)', color='tab:purple')
        ax2.set_ylabel('Efficiency (moves/s/GB)')
        ax2.grid(False)  # Disable grid for second axis to avoid clutter
        
        # Combine legends
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        plt.legend(loc='best')
    
    # Add labels and title
    plt.xlabel('Batch Size')
    plt.ylabel('Performance Metrics (log scale)')
    plt.title('Backgammon Environment Performance by Batch Size')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Format x-axis ticks to show actual batch sizes
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.xticks(batch_sizes, batch_sizes)
    
    # Annotate optimal points
    if 'moves_per_second' in metrics_data:
        best_moves_idx = metrics_data['moves_per_second'].index(max(metrics_data['moves_per_second']))
        plt.annotate(f"Best moves/s: {format_human_readable(metrics_data['moves_per_second'][best_moves_idx])}/s",
                    xy=(batch_sizes[best_moves_idx], metrics_data['moves_per_second'][best_moves_idx]),
                    xytext=(0, 30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    if 'games_per_second' in metrics_data and any(g > 0 for g in metrics_data['games_per_second']):
        best_games_idx = metrics_data['games_per_second'].index(max(metrics_data['games_per_second']))
        plt.annotate(f"Best games/s: {format_human_readable(metrics_data['games_per_second'][best_games_idx])}/s",
                    xy=(batch_sizes[best_games_idx], metrics_data['games_per_second'][best_games_idx]),
                    xytext=(0, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # Save the plot
    plt_filename = GRAPHS_DIR / f"benchmark_results_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plt_filename, dpi=300)
    print(f"\nBenchmark plot saved to: {plt_filename}", flush=True)
    
    # Generate additional plots for specific metrics
    
    # Memory usage plot
    if 'memory_usage_gb' in metrics_data:
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, metrics_data['memory_usage_gb'], 
                marker='o', linestyle='-', linewidth=2, color='tab:orange')
        plt.xscale('log', base=2)
        plt.xlabel('Batch Size')
        plt.ylabel('Memory Usage (GB)')
        plt.title('Memory Usage by Batch Size')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.xticks(batch_sizes, batch_sizes)
        plt_filename = GRAPHS_DIR / f"memory_usage_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(plt_filename, dpi=300)
        print(f"Memory usage plot saved to: {plt_filename}", flush=True)
    
    return plt_filename  # Return the main plot filename


def discover_optimal_batch_sizes(
    memory_limit_gb: float = DEFAULT_MEMORY_LIMIT_GB,
    duration: int = DEFAULT_BENCHMARK_DURATION
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """
    Discover optimal batch sizes by increasing batch size until memory limit or diminishing returns.
    
    Args:
        memory_limit_gb: Maximum memory to use in GB
        duration: Duration of each benchmark in seconds
    
    Returns:
        Tuple of (batch_sizes, moves_per_second, memory_usage_gb, games_per_second, moves_per_second_per_game)
    """
    print(f"\n=== Discovering optimal batch sizes (memory limit: {memory_limit_gb:.2f}GB) ===", flush=True)
    
    batch_sizes = []
    moves_per_second = []
    memory_usage_gb = []
    games_per_second = []
    moves_per_second_per_game = []
    
    # Start with batch size 1 and keep doubling
    batch_size = 1
    last_perf_improvement = float('inf')
    discovery_iteration = 1
    
    print("Starting discovery process - will test increasing batch sizes", flush=True)
    
    while True:
        # Benchmark current batch size
        try:
            print(f"\n{'#'*50}", flush=True)
            print(f"Discovery iteration {discovery_iteration}: Testing batch size {batch_size}", flush=True)
            print(f"{'#'*50}", flush=True)
            discovery_iteration += 1
            
            perf, memory, games_per_sec, moves_per_game = benchmark_batch_size(batch_size, duration)
            
            batch_sizes.append(batch_size)
            moves_per_second.append(perf)
            memory_usage_gb.append(memory)
            games_per_second.append(games_per_sec)
            moves_per_second_per_game.append(moves_per_game)
            
            # Check if we should continue
            if len(moves_per_second) > 1:
                perf_improvement = moves_per_second[-1] / moves_per_second[-2] - 1.0
                print(f"Performance improvement: {perf_improvement:.2%}", flush=True)
                
                # Stop if memory limit reached
                if memory >= memory_limit_gb:
                    print(f"Memory limit reached: {memory:.2f}GB >= {memory_limit_gb:.2f}GB", flush=True)
                    break
                
                # Stop if diminishing returns (less than 10% improvement)
                if perf_improvement < 0.1:
                    if last_perf_improvement < 0.2:  # Two consecutive small improvements
                        print("Diminishing returns detected (two consecutive small improvements)", flush=True)
                        break
                    last_perf_improvement = perf_improvement
                else:
                    last_perf_improvement = float('inf')  # Reset counter if we see a good improvement
        except Exception as e:
            print(f"Error benchmarking batch size {batch_size}: {e}", flush=True)
            print(f"Stopping at maximum batch size {batch_size // 2}", flush=True)
            break
        
        # Double batch size for next iteration
        batch_size *= 2
    
    print(f"\nDiscovery complete: Tested {len(batch_sizes)} batch sizes", flush=True)
    
    # Calculate efficiency for plotting
    efficiency = [perf / mem if mem > 0 else 0 for perf, mem in zip(moves_per_second, memory_usage_gb)]
    
    # Generate plots
    metrics_data = {
        'moves_per_second': moves_per_second,
        'games_per_second': games_per_second,
        'moves_per_second_per_game': moves_per_second_per_game,
        'memory_usage_gb': memory_usage_gb,
        'efficiency': efficiency
    }
    
    # Generate plot with all batch sizes
    plot_filename = generate_benchmark_plots(batch_sizes, metrics_data)
    
    # Select 4 interesting batch sizes for validation
    if len(batch_sizes) > 4:
        # Keep the smallest, largest, and two in the middle
        indices = [0, len(batch_sizes) // 3, 2 * len(batch_sizes) // 3, len(batch_sizes) - 1]
        batch_sizes = [batch_sizes[i] for i in indices]
        moves_per_second = [moves_per_second[i] for i in indices]
        memory_usage_gb = [memory_usage_gb[i] for i in indices]
        games_per_second = [games_per_second[i] for i in indices]
        moves_per_second_per_game = [moves_per_second_per_game[i] for i in indices]
    
    # Print summary table
    print("\n=== Discovery Summary ===", flush=True)
    print(f"{'Batch Size':>10} | {'Moves/Second':>15} | {'Moves/Game/s':>15} | {'Games/s':>12} | {'Memory (GB)':>12} | {'Efficiency':>15}", flush=True)
    print("-" * 95, flush=True)
    
    # Calculate efficiency as moves per second per memory GB
    for i, batch_size in enumerate(batch_sizes):
        perf = moves_per_second[i]
        memory = memory_usage_gb[i]
        moves_per_game = moves_per_second_per_game[i]
        games_per_sec = games_per_second[i]
        efficiency = perf / memory if memory > 0 else 0
        
        print(f"{batch_size:>10} | {format_human_readable(perf):>15} | {format_human_readable(moves_per_game):>15} | "
              f"{format_human_readable(games_per_sec):>12} | {memory:>12.2f} | {format_human_readable(efficiency):>15}/GB", flush=True)
    
    # Find optimal batch sizes for different metrics
    best_moves_idx = moves_per_second.index(max(moves_per_second))
    best_games_idx = games_per_second.index(max(games_per_second))
    best_efficiency_idx = [perf/mem if mem > 0 else 0 for perf, mem in zip(moves_per_second, memory_usage_gb)].index(
        max([perf/mem if mem > 0 else 0 for perf, mem in zip(moves_per_second, memory_usage_gb)]))
    
    print("\n=== Optimal Configurations ===", flush=True)
    print(f"Best for moves/s: Batch size {batch_sizes[best_moves_idx]} with {format_human_readable(moves_per_second[best_moves_idx])} moves/s", flush=True)
    print(f"Best for games/s: Batch size {batch_sizes[best_games_idx]} with {format_human_readable(games_per_second[best_games_idx])} games/s", flush=True)
    print(f"Best for efficiency: Batch size {batch_sizes[best_efficiency_idx]} with {format_human_readable(moves_per_second[best_efficiency_idx]/memory_usage_gb[best_efficiency_idx])}/GB", flush=True)
    print(f"\nCheck the benchmark plot at: {plot_filename}", flush=True)
    
    return batch_sizes, moves_per_second, memory_usage_gb, games_per_second, moves_per_second_per_game


def validate_against_profile(profile: BenchmarkProfile) -> None:
    """
    Validate current performance against a saved profile.
    
    Args:
        profile: Profile to validate against
    """
    print("\n=== Validating against saved profile ===", flush=True)
    print(f"Testing {len(profile.batch_sizes)} batch sizes from previous profile", flush=True)
    print("Batch sizes to test:", profile.batch_sizes, flush=True)
    
    current_results = []
    
    # Process each batch size with explicit progress output
    for i, batch_size in enumerate(profile.batch_sizes):
        print(f"\n{'='*50}", flush=True)
        print(f"Validation run {i+1}/{len(profile.batch_sizes)}: Testing batch size {batch_size}", flush=True)
        print(f"{'='*50}", flush=True)
        
        # Benchmark with current system
        current_perf, current_memory, current_games_per_sec, current_moves_per_game = benchmark_batch_size(batch_size, profile.duration if hasattr(profile, 'duration') else DEFAULT_BENCHMARK_DURATION)
        current_results.append((current_perf, current_memory, current_games_per_sec, current_moves_per_game))
        print(f"COMPLETED validation for batch size {batch_size}: {format_human_readable(current_perf)} moves/s, "
              f"{format_human_readable(current_moves_per_game)} moves/s/game, "
              f"{format_human_readable(current_games_per_sec)} games/s, "
              f"{current_memory:.2f}GB", flush=True)
    
    # Create a results table header
    print("\n=== Validation Results: Moves per Second ===", flush=True)
    print(f"{'Batch Size':>10} | {'Previous (moves/s)':>20} | {'Current (moves/s)':>20} | {'Difference':>10} | {'Memory (GB)':>12}", flush=True)
    print("-" * 80, flush=True)
    
    individual_diffs = []
    
    for i, batch_size in enumerate(profile.batch_sizes):
        # Get data for comparison
        previous_perf = profile.moves_per_second[i]
        current_perf, current_memory, current_games_per_sec, current_moves_per_game = current_results[i]
        
        # Calculate difference percentage
        diff_percentage = (current_perf / previous_perf - 1.0) * 100
        individual_diffs.append(diff_percentage)
        
        # Format display with color indicators (+ or - prefix)
        diff_str = f"{diff_percentage:>+8.2f}%"
        
        print(f"{batch_size:>10} | {format_human_readable(previous_perf):>20} | "
              f"{format_human_readable(current_perf):>20} | {diff_str:>10} | {current_memory:>12.2f}", flush=True)
    
    # If the profile has games per second data, show comparison
    games_diffs = []
    if hasattr(profile, 'games_per_second') and profile.games_per_second:
        print("\n=== Validation Results: Games per Second ===", flush=True)
        print(f"{'Batch Size':>10} | {'Previous (games/s)':>20} | {'Current (games/s)':>20} | {'Difference':>10}", flush=True)
        print("-" * 70, flush=True)
        
        for i, batch_size in enumerate(profile.batch_sizes):
            # Get data for comparison
            previous_games = profile.games_per_second[i]
            _, _, current_games, _ = current_results[i]
            
            # Calculate difference percentage - handle zero values
            if previous_games > 0 and current_games > 0:
                diff_percentage = (current_games / previous_games - 1.0) * 100
            else:
                diff_percentage = 0.0
            games_diffs.append(diff_percentage)
            
            # Format display with color indicators (+ or - prefix)
            diff_str = f"{diff_percentage:>+8.2f}%"
            
            print(f"{batch_size:>10} | {format_human_readable(previous_games):>20} | "
                  f"{format_human_readable(current_games):>20} | {diff_str:>10}", flush=True)
    
    # Create comparison plots
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Plot moves per second comparison
    plt.figure(figsize=(12, 8))
    plt.xscale('log', base=2)  # Log scale with base 2 for batch sizes
    plt.yscale('log')  # Log scale for metrics
    
    # Previous results
    plt.plot(profile.batch_sizes, profile.moves_per_second, 
            marker='o', linestyle='-', linewidth=2, 
            label='Previous - Moves/s', color='tab:blue')
    
    # Current results
    plt.plot(profile.batch_sizes, [r[0] for r in current_results], 
            marker='s', linestyle='--', linewidth=2, 
            label='Current - Moves/s', color='tab:orange')
    
    # If games/s data is available
    if hasattr(profile, 'games_per_second') and profile.games_per_second:
        plt.plot(profile.batch_sizes, profile.games_per_second, 
                marker='o', linestyle='-', linewidth=2, 
                label='Previous - Games/s', color='tab:green')
        plt.plot(profile.batch_sizes, [r[2] for r in current_results], 
                marker='s', linestyle='--', linewidth=2, 
                label='Current - Games/s', color='tab:red')
    
    # Add labels and title
    plt.xlabel('Batch Size')
    plt.ylabel('Performance Metrics (log scale)')
    plt.title('Benchmark Comparison: Previous vs Current')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(loc='best')
    
    # Format x-axis ticks to show actual batch sizes
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.xticks(profile.batch_sizes, profile.batch_sizes)
    
    # Save the comparison plot
    comparison_plt_filename = GRAPHS_DIR / f"benchmark_comparison_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(comparison_plt_filename, dpi=300)
    
    # Create a percentage difference plot
    plt.figure(figsize=(10, 6))
    metrics_to_compare = [
        (individual_diffs, 'Moves per Second', 'tab:blue'),
    ]
    
    if games_diffs:
        metrics_to_compare.append((games_diffs, 'Games per Second', 'tab:green'))
    
    # Bar chart of percentage differences
    bar_width = 0.35
    index = np.arange(len(profile.batch_sizes))
    
    for i, (diffs, label, color) in enumerate(metrics_to_compare):
        plt.bar(index + i * bar_width, diffs, bar_width, label=label, color=color, alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Batch Size')
    plt.ylabel('Performance Difference (%)')
    plt.title('Performance Difference: Current vs Previous')
    plt.xticks(index + bar_width / 2, profile.batch_sizes)
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.legend(loc='best')
    
    # Add a horizontal line at 0%
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Save the difference plot
    diff_plt_filename = GRAPHS_DIR / f"performance_difference_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(diff_plt_filename, dpi=300)
    
    print(f"\nComparison plot saved to: {comparison_plt_filename}", flush=True)
    print(f"Performance difference plot saved to: {diff_plt_filename}", flush=True)
    
    # Calculate overall statistics
    avg_diff = np.mean(individual_diffs)
    min_diff = np.min(individual_diffs)
    max_diff = np.max(individual_diffs)
    
    # Find the best and worst performing batch sizes
    best_idx = individual_diffs.index(max_diff)
    worst_idx = individual_diffs.index(min_diff)
    
    print("\n=== Performance Summary ===", flush=True)
    print(f"Average performance change: {avg_diff:+.2f}%", flush=True)
    print(f"Best case (batch size {profile.batch_sizes[best_idx]}): {max_diff:+.2f}%", flush=True)
    print(f"Worst case (batch size {profile.batch_sizes[worst_idx]}): {min_diff:+.2f}%", flush=True)
    
    # Determine if there's an overall performance regression
    if avg_diff < -5:
        print("\n⚠️  WARNING: Current performance is significantly worse than the saved profile! ⚠️", flush=True)
    elif avg_diff > 5:
        print("\n✅ NOTICE: Current performance is significantly better than the saved profile!", flush=True)
    else:
        print("\n✓ Performance is within expected range of the saved profile.", flush=True)
    
    # Find the most efficient batch size in current run
    efficiencies = [perf/mem if mem > 0 else 0 for perf, mem in [(r[0], r[1]) for r in current_results]]
    best_efficiency_idx = efficiencies.index(max(efficiencies))
    
    # Find the best games/s configuration
    games_per_sec = [r[2] for r in current_results]
    best_games_idx = games_per_sec.index(max(games_per_sec)) if any(g > 0 for g in games_per_sec) else 0
    
    # Find the fastest batch size in current run for moves/s
    fastest_idx = [r[0] for r in current_results].index(max([r[0] for r in current_results]))
    
    print("\n=== Optimal Configurations ===", flush=True)
    print(f"Best for moves/s: Batch size {profile.batch_sizes[fastest_idx]} "
          f"({format_human_readable(current_results[fastest_idx][0])} moves/s)", flush=True)
    
    if any(g > 0 for g in games_per_sec):
        print(f"Best for games/s: Batch size {profile.batch_sizes[best_games_idx]} "
              f"({format_human_readable(games_per_sec[best_games_idx])} games/s)", flush=True)
    
    print(f"Best for efficiency: Batch size {profile.batch_sizes[best_efficiency_idx]} "
          f"({format_human_readable(efficiencies[best_efficiency_idx])}/GB)", flush=True)


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
    
    # Check if profile exists
    profile = load_profile()
    
    # Run in discover mode if requested or no profile exists
    if args.discover or profile is None:
        if profile is None:
            print("\nNo existing profile found, running in discovery mode...", flush=True)
        else:
            print("\nDiscovery mode explicitly requested", flush=True)
        
        # Run discovery with the specified duration
        batch_sizes, moves_per_second, memory_usage_gb, games_per_second, moves_per_second_per_game = discover_optimal_batch_sizes(
            memory_limit_gb=args.memory_limit,
            duration=args.duration
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
        validate_against_profile(profile)


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark for pgx backgammon game environment")
    parser.add_argument("--discover", action="store_true", help="Force discovery mode even if profile exists")
    parser.add_argument("--memory-limit", type=float, default=DEFAULT_MEMORY_LIMIT_GB,
                        help=f"Memory limit in GB (default: {DEFAULT_MEMORY_LIMIT_GB})")
    parser.add_argument("--duration", type=int, default=DEFAULT_BENCHMARK_DURATION,
                        help=f"Duration of each batch size test in seconds (default: {DEFAULT_BENCHMARK_DURATION})")
    parser.add_argument("--single-batch", type=int, help="Test only a specific batch size")
    
    args = parser.parse_args()
    
    # If single batch mode is requested, just benchmark that batch size
    if args.single_batch:
        batch_size = args.single_batch
        print(f"Running single batch benchmark with batch size {batch_size}")
        moves_per_second, memory_usage_gb, games_per_second, moves_per_second_per_game = benchmark_batch_size(batch_size, args.duration)
        print("\nSingle Batch Benchmark Results:")
        print(f"Moves per second: {format_human_readable(moves_per_second)}/s")
        print(f"Moves per second per game: {format_human_readable(moves_per_second_per_game)}/s")
        print(f"Games per second: {format_human_readable(games_per_second)}/s")
        print(f"Memory usage: {memory_usage_gb:.2f}GB")
        print(f"Efficiency: {format_human_readable(moves_per_second/memory_usage_gb)}/GB")
    else:
        # Run the regular benchmark process
        run_benchmark(args)


if __name__ == "__main__":
    main() 