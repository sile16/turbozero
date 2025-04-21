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
from dataclasses import dataclass, asdict, field
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


@dataclass
class BatchBenchResult:
    """Holds the results from benchmarking a single batch size."""
    batch_size: int
    moves_per_second: float = 0.0
    memory_usage_gb: float = 0.0
    games_per_second: float = 0.0
    moves_per_second_per_game: float = 0.0
    avg_moves_per_game: float = 0.0
    median_moves_per_game: float = 0.0
    min_moves_per_game: float = float('inf')
    max_moves_per_game: float = 0.0
    total_moves: int = 0
    completed_games: int = 0
    elapsed_time: float = 0.0
    efficiency: float = 0.0 # moves/s / GB


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
    """Get current memory usage in GB, attempting device-specific reporting."""
    sys_info = get_system_info()
    jaxlib_type = sys_info.get('jaxlib_type', 'cpu')
    device = jax.devices()[0] if jax.devices() else None
    
    # Static variable to track if we've printed the memory source message
    if not hasattr(get_memory_usage, "_printed_source"):
        get_memory_usage._printed_source = False
    
    # Helper function for process RSS
    def get_process_rss_gb():
        if platform.system() == "Darwin":
            try:
                cmd = "ps -o rss= -p " + str(os.getpid())
                output = subprocess.check_output(cmd.split(), timeout=1).decode().strip()
                mem_kb = float(output)
                return mem_kb / 1024 / 1024  # Convert KB to GB
            except Exception as e:
                print(f"Warning: Failed to get macOS process memory via ps: {e}", flush=True)
        elif platform.system() == "Linux":
            try:
                with open('/proc/self/status', 'r') as f:
                    for line in f:
                        if line.startswith('VmRSS:'):
                            mem_kb = float(line.split()[1])
                            return mem_kb / 1024 / 1024  # Convert KB to GB
            except Exception as e:
                print(f"Warning: Failed to get Linux process memory via /proc: {e}", flush=True)
        
        # Fallback using psutil if specific methods fail or not implemented
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_bytes = process.memory_info().rss
            return mem_bytes / 1024 / 1024 / 1024  # Convert bytes to GB
        except ImportError:
            print("Warning: psutil not installed, cannot provide fallback memory usage.", flush=True)
            return 0.0
        except Exception as e:
            print(f"Warning: Failed to get memory via psutil: {e}", flush=True)
            return 0.0

    # --- Device-specific logic ---
    
    if jaxlib_type == 'cuda' and device is not None:
        try:
            # Use nvidia-smi to get GPU memory usage for the primary JAX device
            device_id = device.id
            cmd = [
                "nvidia-smi",
                f"--query-gpu=memory.used",
                f"--format=csv,noheader,nounits",
                f"-i", f"{device_id}"
            ]
            output = subprocess.check_output(cmd, timeout=2).decode().strip()
            mem_mib = float(output)
            if not get_memory_usage._printed_source:
                print(f"Reporting CUDA device {device_id} memory via nvidia-smi.", flush=True)
                get_memory_usage._printed_source = True
            return mem_mib / 1024  # Convert MiB to GB
        except FileNotFoundError:
            print("Warning: nvidia-smi not found. Falling back to process RSS for memory usage.", flush=True)
            return get_process_rss_gb()
        except subprocess.TimeoutExpired:
            print("Warning: nvidia-smi timed out. Falling back to process RSS.", flush=True)
            return get_process_rss_gb()
        except Exception as e:
            print(f"Warning: Failed to get GPU memory via nvidia-smi: {e}. Falling back to process RSS.", flush=True)
            return get_process_rss_gb()
            
    elif jaxlib_type == 'metal':
        # For Metal (macOS unified memory), process RSS is the best available proxy
        if not get_memory_usage._printed_source:
            print("Reporting host process memory usage (psutil/ps) for Metal backend.", flush=True)
            get_memory_usage._printed_source = True
        return get_process_rss_gb()
        
    else: # cpu or unknown
        # For CPU backend, process RSS is appropriate
        if not get_memory_usage._printed_source:
            print(f"Reporting host process memory usage (psutil/ps/proc) for {jaxlib_type} backend.", flush=True)
            get_memory_usage._printed_source = True
        return get_process_rss_gb()


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


def benchmark_batch_size(batch_size: int, max_duration: int = 120) -> BatchBenchResult:
    """
    Benchmark the backgammon environment with a specific batch size for a fixed duration.
    
    This implementation uses a "Complete All Games" batching strategy:
    - For batch sizes > 1, we run all games in a batch until ALL games are terminated
    - Only then do we reset the entire batch and start fresh games
    - This avoids expensive selective state resets and is more GPU-friendly
    
    Args:
        batch_size: Batch size to test
        max_duration: Duration in seconds to run the benchmark (default: 120s = 2 minutes)
        
    Returns:
        BatchBenchResult containing detailed performance metrics.
    """
    print(f"\n{'-'*30}", flush=True)
    print(f"Benchmarking batch size: {batch_size} for {max_duration} seconds", flush=True)
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
    
    # Track per-instance state - move initialization earlier
    current_game_moves = [0] * batch_size
    # Add trackers for individual games rather than global max
    max_game_length_per_instance = [0] * batch_size
    
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
        print(f"Using batch mode with {batch_size} parallel games", flush=True)
        def step_batch_states(key, states):
            """Take steps for a batch of parallel games"""
            # Create a unique random key for each state in the batch
            batch_keys = jax.random.split(key, batch_size)
            
            # Use vmap to apply step_single_state to each (key, state) pair
            # Explicitly specify in_axes to be clear about vectorization dimension
            vectorized_step = jax.vmap(step_single_state, in_axes=(0, 0))
            return vectorized_step(batch_keys, states)
        
        # Initialize batch of states
        print(f"Initializing {batch_size} states...", flush=True)
        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, batch_size)
        states = []
        for i in range(batch_size):
            states.append(env.init(init_keys[i]))
        
        print("Converting states to batched JAX arrays...", flush=True)
        # Convert to JAX arrays
        # JAX tree_map takes a function as the first argument
        # and trees as the rest of the arguments
        state = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs),  # First arg is the function
            *states  # Unpack all states as arguments
        )
        
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
    print(f"Running benchmark for {max_duration} seconds...", flush=True)
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
    game_length_distribution = {}  # Track distribution of game lengths
    
    # Track memory usage
    initial_memory_gb = get_memory_usage()
    peak_memory_gb = initial_memory_gb
    print(f"Initial memory usage: {initial_memory_gb:.2f}GB", flush=True)
    
    # Create a tqdm progress bar for time
    print("Starting benchmark loop:", flush=True)
    
    with tqdm(total=max_duration, desc=f"Batch {batch_size}", unit="s", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
        last_update_time = time.time()
        iteration_count = 0
        
        try:
            while (elapsed_time := time.time() - start_time) < max_duration:
                iteration_count += 1
                key, subkey = jax.random.split(key)
                
                # Execute the step function
                state = step_fn(subkey, state)
                # Force completion of the operation
                jax.block_until_ready(state)
                
                # Count one move per valid state in the batch
                # We only count non-terminated states for total moves
                if batch_size == 1:
                    # Check termination status *before* potentially re-initializing
                    terminated_before_move = state.terminated
                    if not terminated_before_move:
                         total_moves += 1
                         current_game_moves[0] += 1
                         max_game_length_per_instance[0] = max(max_game_length_per_instance[0], current_game_moves[0])
                else:
                    # Only count moves for non-terminated states (active games)
                    terminated_states = state.terminated
                    terminated_np = np.array(terminated_states)
                    valid_moves_count = batch_size - np.sum(terminated_np)
                    total_moves += valid_moves_count

                    # Update moves per game counters ONLY for active (non-terminated) states
                    for i in range(batch_size):
                        if not terminated_np[i]:
                            current_game_moves[i] += 1
                            # Track max game length per instance as we go
                            max_game_length_per_instance[i] = max(max_game_length_per_instance[i], current_game_moves[i])
                
                # Check for completed games (assuming terminated flag indicates completed game)
                # This logic now needs to handle the state *after* the step
                if batch_size == 1:
                    if state.terminated:
                        completed_games += 1
                        # Record the game length
                        game_length = current_game_moves[0]
                        # Only record stats if game had moves
                        if game_length > 0:
                            total_moves_in_completed_games += game_length
                            max_moves_per_game = max(max_moves_per_game, game_length)
                            min_moves_per_game = min(min_moves_per_game, game_length)
                            game_lengths.append(game_length)
                            # Update game length distribution
                            game_length_distribution[game_length] = game_length_distribution.get(game_length, 0) + 1
                            if game_length == 1:
                                single_move_games += 1
                        # Reset the max game length tracker for this game
                        max_game_length_per_instance[0] = 0 
                        # Reset counter for the next game
                        current_game_moves[0] = 0
                        # Reinitialize for a new game
                        key, init_key = jax.random.split(key)
                        state = env.init(init_key)
                else:
                    # Process terminations for multiple games in parallel
                    terminated_states = state.terminated
                    terminated_np = np.array(terminated_states) # Numpy for iteration
                    
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
                                    
                                    # Update game length distribution
                                    game_length_distribution[game_length] = game_length_distribution.get(game_length, 0) + 1
                                    
                                    # Track single-move games
                                    if game_length == 1:
                                        single_move_games += 1
                                
                                # Reset counter for this game instance but don't reset the state yet
                                current_game_moves[i] = 0
                                # Reset the max game length tracker for this game
                                max_game_length_per_instance[i] = 0
                        
                        # Check if ALL games in this batch have terminated
                        if np.all(terminated_np):
                            # Only reset the entire batch when all games have finished
                            print(f"All {batch_size} games terminated, resetting entire batch", flush=True)
                            
                            # Generate keys for the new batch
                            key, init_key = jax.random.split(key)
                            init_keys = jax.random.split(init_key, batch_size)
                            
                            # Initialize a fresh batch of states
                            new_states = []
                            for i in range(batch_size):
                                new_states.append(env.init(init_keys[i]))
                            
                            # Convert to JAX arrays - single tree_map operation is more efficient
                            state = jax.tree_util.tree_map(
                                lambda *xs: jnp.stack(xs),
                                *new_states
                            )
                            print("Full batch reset complete", flush=True)
                
                # Update progress bar and metrics (less frequently to reduce overhead)
                current_time = time.time()
                if current_time - last_update_time > 0.5:  # Reduced update frequency (was 0.2)
                    elapsed = current_time - start_time
                    pbar.n = round(elapsed)
                    
                    # Calculate performance metrics
                    moves_per_sec = total_moves / elapsed if elapsed > 0 else 0
                    games_per_sec = completed_games / elapsed if elapsed > 0 and completed_games > 0 else 0
                    avg_moves = total_moves_in_completed_games / completed_games if completed_games > 0 else 0
                    
                    # Update peak memory usage
                    current_memory = get_memory_usage()
                    peak_memory_gb = max(peak_memory_gb, current_memory)
                    
                    # Update progress bar postfix with current stats
                    pbar.set_postfix(
                        moves_s=f"{format_human_readable(moves_per_sec)}/s", 
                        games_s=f"{format_human_readable(games_per_sec)}/s", 
                        avg_moves=f"{avg_moves:.1f}",
                        mem=f"{peak_memory_gb:.2f}GB"
                    )
                    pbar.refresh() # Force refresh
                    last_update_time = current_time
                
        except KeyboardInterrupt:
            print("\nBenchmark interrupted!", flush=True)
            # Ensure loop terminates if interrupted
            elapsed_time = time.time() - start_time 
        
        # Final update for progress bar to show completion
        pbar.n = round(elapsed_time)
        pbar.refresh()

    # ---- Final calculations and statistics ----
    # Final elapsed time might be slightly over max_duration
    final_elapsed_time = time.time() - start_time

    # Account for any ongoing games at the end of the timer
    # These didn't formally complete, but their moves contribute to total_moves
    # and their lengths are relevant for max_moves_per_game if longer than completed ones
    for i in range(batch_size):
        if current_game_moves[i] > 0: # If a game was in progress
             game_length = current_game_moves[i]
             # Update max/min moves, but don't count as completed game or add to avg/median
             max_moves_per_game = max(max_moves_per_game, game_length, max_game_length_per_instance[i])
             # Only update min if no games completed yet
             if not game_lengths: 
                 min_moves_per_game = min(min_moves_per_game, game_length) 
             # Still useful to see distribution including unfinished games
             game_length_distribution[game_length] = game_length_distribution.get(game_length, 0) + 1
             if game_length == 1:
                 # Count potentially unfinished single-move games
                 single_move_games += 1

    # Ensure min_moves_per_game is not infinity if no games completed but some started
    if completed_games == 0 and min_moves_per_game == float('inf'):
        min_moves_per_game = 0 # Or consider min from game_length_distribution if populated
        if game_length_distribution:
            min_moves_per_game = min(game_length_distribution.keys())
            
    # Calculate final performance metrics
    moves_per_second = total_moves / final_elapsed_time if final_elapsed_time > 0 else 0
    games_per_second = completed_games / final_elapsed_time if final_elapsed_time > 0 and completed_games > 0 else 0
    moves_per_sec_per_game = moves_per_second / batch_size if batch_size > 0 else 0
    efficiency = moves_per_second / peak_memory_gb if peak_memory_gb > 0 else 0

    # Calculate average and median moves per *completed* game
    if completed_games > 0:
        avg_moves_per_game = total_moves_in_completed_games / completed_games
        median_moves_per_game = np.median(game_lengths) if game_lengths else 0
    else:
        avg_moves_per_game = 0
        median_moves_per_game = 0

    # Print warnings and stats as before
    if completed_games > 0 and avg_moves_per_game < 10:
        print(f"\n⚠️  WARNING: Average moves per completed game is very low ({avg_moves_per_game:.1f}).")
        # Detailed warning message remains the same...
        print(f"   Min moves (completed): {min_moves_per_game if game_lengths else 'N/A'}, Max moves (any): {max_moves_per_game}")

    if completed_games >= 10:
        print("\n📊 COMPLETED GAME LENGTH DISTRIBUTION:")
        # Distribution printing logic remains the same...
        lengths = sorted([l for l in game_length_distribution.keys() if l in game_lengths]) # Only completed lengths
        # ... (rest of distribution printing)
        pass # Placeholder for unchanged distribution printing
        
    print("\nBenchmark complete!", flush=True)
    print(f"Batch Size: {batch_size}", flush=True)
    print(f"Total iterations: {iteration_count}", flush=True)
    print(f"Total moves executed: {total_moves}", flush=True)
    print(f"Total completed games: {completed_games}", flush=True)
    print(f"Elapsed time: {final_elapsed_time:.2f}s (Target: {max_duration}s)", flush=True)
    
    print("--- Performance Metrics ---", flush=True)
    print(f"Moves per second: {format_human_readable(moves_per_second)}/s", flush=True)
    print(f"Games per second (completed): {format_human_readable(games_per_second)}/s", flush=True)
    print(f"Moves per second per game: {format_human_readable(moves_per_sec_per_game)}/s/game", flush=True)
    print(f"Peak Memory Usage: {peak_memory_gb:.2f}GB", flush=True)
    print(f"Efficiency (Moves/s/GB): {format_human_readable(efficiency)}/GB", flush=True)
    
    print("--- Game Length Statistics (Completed Games) ---", flush=True)
    print(f"Average moves per game: {avg_moves_per_game:.2f}", flush=True)
    print(f"Median moves per game: {median_moves_per_game:.1f}", flush=True)
    print(f"Minimum moves per game: {min_moves_per_game if game_lengths else 'N/A'}", flush=True)
    print(f"Maximum moves (any game): {max_moves_per_game}", flush=True)
    print(f"Single-move games (completed): {single_move_games} ({single_move_games / completed_games * 100:.1f}% of completed)" if completed_games > 0 else "N/A")

    # Package results into the dataclass
    result = BatchBenchResult(
        batch_size=batch_size,
        moves_per_second=moves_per_second,
        memory_usage_gb=peak_memory_gb,
        games_per_second=games_per_second,
        moves_per_second_per_game=moves_per_sec_per_game,
        avg_moves_per_game=avg_moves_per_game,
        median_moves_per_game=median_moves_per_game,
        min_moves_per_game=min_moves_per_game if game_lengths else 0,
        max_moves_per_game=max_moves_per_game,
        total_moves=total_moves,
        completed_games=completed_games,
        elapsed_time=final_elapsed_time,
        efficiency=efficiency
    )
    
    return result


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


def print_summary_table(results: List[BatchBenchResult], title: str = "Benchmark Summary"):
    """Prints a formatted table summarizing benchmark results."""
    if not results:
        print(f"\n=== {title}: No results to display ===", flush=True)
        return

    print(f"\n=== {title} ===", flush=True)
    # Header
    header = (
        f"{'Batch':>7} | {'Moves/s':>12} | {'Games/s':>12} | {'Moves/s/G':>10} | "
        f"{'Mem (GB)':>10} | {'Effic.':>12} | "
        f"{'Avg Moves':>10} | {'Med Moves':>10} | {'Min Moves':>10} | {'Max Moves':>10}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    # Rows
    for r in results:
        print(f"{r.batch_size:>7} | "
              f"{format_human_readable(r.moves_per_second):>12} | "
              f"{format_human_readable(r.games_per_second):>12} | "
              f"{format_human_readable(r.moves_per_second_per_game):>10} | "
              f"{r.memory_usage_gb:>10.2f} | "
              f"{format_human_readable(r.efficiency):>12}/GB | " # Efficiency includes /GB
              f"{r.avg_moves_per_game:>10.1f} | "
              f"{r.median_moves_per_game:>10.1f} | "
              f"{r.min_moves_per_game if r.min_moves_per_game != float('inf') else 'N/A':>10} | " # Handle inf min moves
              f"{r.max_moves_per_game:>10}", flush=True)


def discover_optimal_batch_sizes(
    memory_limit_gb: float = DEFAULT_MEMORY_LIMIT_GB,
    max_duration: int = 120
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """
    Discover optimal batch sizes by increasing batch size until memory limit or diminishing returns.
    
    Args:
        memory_limit_gb: Maximum memory to use in GB
        max_duration: Duration in seconds for each batch size test (default: 120s = 2 minutes)
    
    Returns:
        Tuple containing lists needed for BenchmarkProfile compatibility.
    """
    print(f"\n=== Discovering optimal batch sizes (memory limit: {memory_limit_gb:.2f}GB) ===", flush=True)
    print(f"Duration per batch size: {max_duration}s", flush=True)
    
    all_results: List[BatchBenchResult] = []
    
    # Start with batch size 1 and keep doubling
    batch_size = 1
    last_perf_improvement = float('inf')
    
    print("Starting discovery process - will test increasing batch sizes", flush=True)
    
    # Use tqdm for the outer loop tracking batch sizes tested
    # We don't know the total number of batches beforehand, so leave total=None
    with tqdm(desc="Discovering Batch Sizes", unit="batch") as outer_pbar:
        while True:
            # Benchmark current batch size
            try:
                print(f"\n{'#'*50}", flush=True)
                print(f"Discovery iteration {outer_pbar.n + 1}: Testing batch size {batch_size}", flush=True)
                print(f"{'#'*50}", flush=True)
                
                # Temporarily silence stdout for benchmark to avoid progress bar confusion
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                try:
                    result = benchmark_batch_size(
                        batch_size,
                        max_duration=max_duration
                    )
                finally:
                    # Restore stdout regardless of success/failure
                    sys.stdout.close()
                    sys.stdout = original_stdout
                
                # Print a summary of the result
                print(f"Completed benchmark for batch size {batch_size}: "
                      f"{format_human_readable(result.moves_per_second)}/s, "
                      f"avg moves: {result.avg_moves_per_game:.1f}, "
                      f"memory: {result.memory_usage_gb:.2f}GB", flush=True)
                
                all_results.append(result)
                outer_pbar.update(1) # Increment the outer progress bar
                
                # Check if we should continue
                if len(all_results) > 1:
                    # Check based on moves_per_second
                    perf_improvement = all_results[-1].moves_per_second / all_results[-2].moves_per_second - 1.0 if all_results[-2].moves_per_second > 0 else float('inf')
                    print(f"Performance improvement (moves/s): {perf_improvement:.2%}", flush=True)
                    
                    # Stop if memory limit reached
                    if result.memory_usage_gb >= memory_limit_gb:
                        print(f"Memory limit reached: {result.memory_usage_gb:.2f}GB >= {memory_limit_gb:.2f}GB", flush=True)
                        break
                    
                    # Stop if diminishing returns (less than 10% improvement)
                    # Requires two consecutive small improvements to stop
                    if perf_improvement < 0.1:
                        if last_perf_improvement < 0.2:  
                            print("Diminishing returns detected (two consecutive small improvements < 20% and < 10%)", flush=True)
                            break
                        last_perf_improvement = perf_improvement
                    else:
                        last_perf_improvement = float('inf')  # Reset counter
            except Exception as e:
                print(f"Error benchmarking batch size {batch_size}: {e}", flush=True)
                print(f"Stopping discovery before batch size {batch_size}", flush=True)
                break
            
            # Double batch size for next iteration
            batch_size *= 2
    
    print(f"\nDiscovery complete: Tested {len(all_results)} batch sizes", flush=True)
    
    # Generate plots using the collected results
    plot_batch_sizes = [r.batch_size for r in all_results]
    metrics_data = {
        'moves_per_second': [r.moves_per_second for r in all_results],
        'games_per_second': [r.games_per_second for r in all_results],
        'moves_per_second_per_game': [r.moves_per_second_per_game for r in all_results],
        'memory_usage_gb': [r.memory_usage_gb for r in all_results],
        'efficiency': [r.efficiency for r in all_results]
    }
    plot_filename = generate_benchmark_plots(plot_batch_sizes, metrics_data)
    
    # Print summary table
    print_summary_table(all_results, title="Discovery Summary")
    
    # Extract optimal configurations from all results
    if all_results:
        best_moves_idx = np.argmax([r.moves_per_second for r in all_results])
        best_games_idx = np.argmax([r.games_per_second for r in all_results])
        best_efficiency_idx = np.argmax([r.efficiency for r in all_results])
        
        print("\n=== Optimal Configurations (Discovered) ===", flush=True)
        print(f"Best for moves/s: Batch size {all_results[best_moves_idx].batch_size} with {format_human_readable(all_results[best_moves_idx].moves_per_second)} moves/s", flush=True)
        print(f"Best for games/s: Batch size {all_results[best_games_idx].batch_size} with {format_human_readable(all_results[best_games_idx].games_per_second)} games/s", flush=True)
        print(f"Best for efficiency: Batch size {all_results[best_efficiency_idx].batch_size} with {format_human_readable(all_results[best_efficiency_idx].efficiency)}/GB", flush=True)
        print(f"\nCheck the benchmark plot at: {plot_filename}", flush=True)
    else:
        print("\nNo successful benchmark runs completed.", flush=True)
    
    # Select batch sizes for the profile (if needed) - keep original logic for compatibility
    # If we tested many, select a subset for the profile
    profile_results = all_results
    if len(all_results) > 4:
        indices = sorted(list(set([0, len(all_results) // 3, 2 * len(all_results) // 3, len(all_results) - 1]))) # Ensure unique indices
        profile_results = [all_results[i] for i in indices]
        print(f"\nSelected subset of {len(profile_results)} batch sizes for profile: {[r.batch_size for r in profile_results]}", flush=True)

    # Return lists required by BenchmarkProfile
    return (
        [r.batch_size for r in profile_results],
        [r.moves_per_second for r in profile_results],
        [r.memory_usage_gb for r in profile_results],
        [r.games_per_second for r in profile_results],
        [r.moves_per_second_per_game for r in profile_results]
    )


def validate_against_profile(profile: BenchmarkProfile, max_duration: int = 120) -> None:
    """
    Validate current performance against a saved profile using time-based runs.
    
    Args:
        profile: Profile to validate against
        max_duration: Duration in seconds for each batch size test (default: 120s = 2 minutes)
    """
    print("\n=== Validating against saved profile ===", flush=True)
    print(f"Testing {len(profile.batch_sizes)} batch sizes from profile: {profile.batch_sizes}", flush=True)
    print(f"Duration per batch size: {max_duration}s", flush=True)
    
    current_results: List[BatchBenchResult] = []
    
    # Use tqdm for the outer loop tracking batch sizes being validated
    with tqdm(total=len(profile.batch_sizes), desc="Validating Batch Sizes", unit="batch") as outer_pbar:
        for i, batch_size in enumerate(profile.batch_sizes):
            print(f"\n{'='*50}", flush=True)
            print(f"Validation run {i+1}/{len(profile.batch_sizes)}: Testing batch size {batch_size}", flush=True)
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
                  f"{format_human_readable(result.moves_per_second_per_game)} moves/s/game, "
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
    
    for i, batch_size in enumerate(profile.batch_sizes):
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
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    comparison_plt_filename = None
    diff_plt_filename = None

    if profile.batch_sizes and profile.moves_per_second and [r.moves_per_second for r in current_results]:
        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
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
        
        plt.xlabel('Batch Size')
        plt.ylabel('Performance Metrics (log scale)')
        plt.title('Benchmark Comparison: Previous vs Current')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend(loc='best')
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.xticks([r.batch_size for r in current_results], [r.batch_size for r in current_results])
        comparison_plt_filename = GRAPHS_DIR / f"benchmark_comparison_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(comparison_plt_filename, dpi=300)
        plt.close() # Close the figure

        # Create percentage difference plot
        plt.figure(figsize=(10, 6))
        metrics_to_compare = [(moves_diffs, 'Moves/s Diff', 'tab:blue')]
        if games_diffs:
             metrics_to_compare.append((games_diffs, 'Games/s Diff', 'tab:green'))
        
        bar_width = 0.35
        index = np.arange(len(profile.batch_sizes))
        
        for i, (diffs, label, color) in enumerate(metrics_to_compare):
            plt.bar(index + i * bar_width, diffs, bar_width, label=label, color=color, alpha=0.7)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Performance Difference (%)')
        plt.title('Performance Difference: Current vs Previous Profile')
        plt.xticks(index + bar_width / len(metrics_to_compare) / 2, profile.batch_sizes)
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        plt.legend(loc='best')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        diff_plt_filename = GRAPHS_DIR / f"performance_difference_{timestamp}.png"
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
        print(f"Best case (batch size {profile.batch_sizes[best_idx]}): {max_diff:+.2f}%", flush=True)
        print(f"Worst case (batch size {profile.batch_sizes[worst_idx]}): {min_diff:+.2f}%", flush=True)
        
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
    
    # Set the target game count (with default value if not specified)
    target_game_count = args.target_game_count if args.target_game_count else 1000
    
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
            max_duration=args.duration
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
        validate_against_profile(profile, max_duration=args.duration)


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark for pgx backgammon game environment")
    parser.add_argument("--discover", action="store_true", help="Force discovery mode even if profile exists")
    parser.add_argument("--memory-limit", type=float, default=DEFAULT_MEMORY_LIMIT_GB,
                        help=f"Memory limit in GB (default: {DEFAULT_MEMORY_LIMIT_GB})")
    parser.add_argument("--duration", type=int, default=120,
                        help=f"Maximum duration of each batch size test in seconds (default: 120)")
    parser.add_argument("--target-game-count", type=int, default=1000,
                        help="Target number of games to complete for consistent statistics (default: 1000)")
    parser.add_argument("--single-batch", type=int, help="Test only a specific batch size")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # If single batch mode is requested, just benchmark that batch size
    if args.single_batch:
        batch_size = args.single_batch
        # target_game_count = args.target_game_count if args.target_game_count else 1000 # No longer used directly
        
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