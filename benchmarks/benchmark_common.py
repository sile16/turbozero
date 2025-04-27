#!/usr/bin/env python3
"""
Common functionality for benchmark scripts.
"""

import os
import sys
import time
import json
import platform
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any, Tuple, NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
import chex
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from datetime import datetime

# Constants
DEFAULT_MEMORY_LIMIT_GB = 24  # Maximum memory to use (in GB)
DEFAULT_BENCHMARK_DURATION = 30  # Duration of each batch size test in seconds

# Get the directory where the benchmark script is located
BENCHMARK_DIR = Path(__file__).parent.absolute()
PROFILE_DIR = BENCHMARK_DIR / "profiles"
GRAPHS_DIR = BENCHMARK_DIR / "graphs"
HUMAN_READABLE_UNITS = ["", "K", "M", "B", "T"]

# Define common data structures
class BatchBenchResult(NamedTuple):
    batch_size: int
    moves_per_second: float
    games_per_second: float
    avg_game_length: float
    median_game_length: float
    min_game_length: float
    max_game_length: float
    memory_usage_gb: float
    efficiency: float
    valid: bool = True


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
    """Get system information for benchmarking context."""
    return {
        "platform": platform.system(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_type": "cpu" if jax.devices()[0].platform == "cpu" else "gpu",
        "device_info": str(jax.devices()[0]),
    }


def print_system_info(system_info: Dict[str, str]) -> None:
    """Print system information to the console."""
    print("\n=== System Information ===")
    for key, value in system_info.items():
        print(f"{key}: {value}")


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


def get_cpu_gpu_usage() -> Tuple[float, float]:
    """Get CPU and GPU usage percentages."""
    cpu_usage = 0.0
    gpu_usage = 0.0
    
    # Get CPU usage
    try:
        import psutil
        cpu_usage = psutil.cpu_percent(interval=0.1)  # Sample over 0.1s
    except ImportError:
        print("Warning: psutil not installed, cannot measure CPU usage", flush=True)
    
    # Get GPU usage if available
    sys_info = get_system_info()
    if sys_info["jaxlib_type"] == "cuda":
        try:
            device = jax.devices()[0]
            device_id = device.id
            cmd = [
                "nvidia-smi",
                f"--query-gpu=utilization.gpu",
                f"--format=csv,noheader,nounits",
                f"-i", f"{device_id}"
            ]
            output = subprocess.check_output(cmd, timeout=2).decode().strip()
            gpu_usage = float(output)
        except Exception as e:
            print(f"Warning: Failed to get GPU usage via nvidia-smi: {e}", flush=True)
    
    return cpu_usage, gpu_usage


def random_action_from_mask(key, mask):
    """Sample a random action index based on the legal action mask."""
    logits = jnp.where(mask, 0.0, -1e9)
    return jax.random.categorical(key, logits)


def create_results_dir(name: str = "graphs") -> Path:
    """Create a directory for benchmark results if it doesn't exist."""
    results_dir = Path("benchmarks") / name
    results_dir.mkdir(exist_ok=True, parents=True)
    return results_dir


def generate_benchmark_plots(
    batch_sizes: List[int], 
    metrics_data: List[BatchBenchResult], 
    timestamp: Optional[str] = None,
    include_efficiency: bool = True,
    include_cpu_usage: bool = False
) -> Tuple[str, str]:
    """Generate and save plots of benchmark results.
    
    Args:
        batch_sizes: List of batch sizes that were benchmarked
        metrics_data: List of benchmark results
        timestamp: Optional string to include in filenames (e.g., test name and simulation count)
        include_efficiency: Whether to include efficiency metrics in the plots
        include_cpu_usage: Whether to include CPU usage in the memory plot
    
    Returns:
        Tuple of (performance_plot_path, memory_plot_path)
    """
    results_dir = create_results_dir()
    
    system_info = get_system_info()
    platform_name = system_info["platform"].lower()
    processor_name = system_info["processor"].lower()
    
    # Extract metrics for plotting
    moves_per_second = [result.moves_per_second for result in metrics_data]
    games_per_second = [result.games_per_second for result in metrics_data]
    memory_usage = [result.memory_usage_gb for result in metrics_data]
    efficiency = [result.efficiency for result in metrics_data]
    
    # Create performance plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # First plot for raw performance
    ax1.plot(batch_sizes, moves_per_second, 'b-o', label='Moves/s')
    ax1.set_xscale('log', base=2)
    ax1.set_ylabel('Moves per Second', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xlabel('Batch Size')
    ax1.set_title('Performance Scaling with Batch Size')
    ax1.grid(True, alpha=0.3)
    
    # Add games/s as a secondary y-axis
    ax1_right = ax1.twinx()
    ax1_right.plot(batch_sizes, games_per_second, 'r-^', label='Games/s')
    ax1_right.set_ylabel('Games per Second', color='r')
    ax1_right.tick_params(axis='y', labelcolor='r')
    
    # Create legend for both lines
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_right.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Second plot for memory usage
    ax2.plot(batch_sizes, memory_usage, 'g-o', label='Memory (GB)')
    ax2.set_xscale('log', base=2)
    ax2.set_ylabel('Memory Usage (GB)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_xlabel('Batch Size')
    ax2.set_title('Memory Usage vs Batch Size')
    ax2.grid(True, alpha=0.3)
    
    # Add efficiency as a secondary y-axis if requested
    if include_efficiency:
        ax2_right = ax2.twinx()
        ax2_right.plot(batch_sizes, efficiency, 'm-^', label='Efficiency')
        ax2_right.set_ylabel('Efficiency (Moves/s/GB)', color='m')
        ax2_right.tick_params(axis='y', labelcolor='m')
        
        # Create legend for both lines
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_right.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax2.legend(loc='upper left')
    
    plt.tight_layout()
    
    # Define filenames
    timestamp_str = f"_{timestamp}" if timestamp else ""
    performance_plot_path = results_dir / f"{platform_name}_{processor_name}{timestamp_str}.png"
    
    plt.savefig(performance_plot_path, dpi=120)
    plt.close()
    
    # Create separate memory plot with memory as percentage and CPU usage if requested
    plt.figure(figsize=(10, 6))
    
    # Estimate system RAM and convert memory to percentage
    try:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        memory_percent = [mem / total_memory_gb * 100 for mem in memory_usage]
        plt.plot(batch_sizes, memory_percent, 'g-o', label='Memory Usage (%)')
    except ImportError:
        # Fall back to absolute values if psutil not available
        memory_percent = memory_usage
        plt.plot(batch_sizes, memory_usage, 'g-o', label='Memory Usage (GB)')
    
    # Add CPU usage if requested
    if include_cpu_usage:
        # We don't have historical CPU usage per batch size, so we'll run a sample now
        current_cpu_usage, _ = get_cpu_gpu_usage()
        # Just use the current CPU usage as a horizontal line for reference
        plt.axhline(y=current_cpu_usage, color='orange', linestyle='-', label=f'Current CPU Usage ({current_cpu_usage:.1f}%)')
    
    # Add moves per second (scaled down) for reference
    plt.plot(batch_sizes, [m/max(moves_per_second)*50 for m in moves_per_second], 'b-^', label='Moves/s (scaled)')
    
    plt.xscale('log', base=2)
    plt.xlabel('Batch Size')
    plt.ylabel('Value')
    plt.title('Memory Usage and CPU Utilization')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    memory_plot_path = results_dir / f"{platform_name}_{processor_name}_memory{timestamp_str}.png"
    plt.savefig(memory_plot_path, dpi=120)
    plt.close()
    
    print(f"Benchmark plot saved to: {performance_plot_path}")
    print(f"Memory usage plot saved to: {memory_plot_path}")
    
    return str(performance_plot_path), str(memory_plot_path)


def validate_against_profile(
    batch_results: List[BatchBenchResult],
    profile_path: Path,
    system_info: Dict[str, str]
) -> None:
    """Compare current benchmark results against a previous profile."""
    if not profile_path.exists():
        print(f"No profile found at {profile_path}")
        return
    
    with open(profile_path, 'r') as f:
        profile = json.load(f)
    
    if 'batch_sizes' not in profile or 'moves_per_second' not in profile:
        print("Profile doesn't contain required data for comparison")
        return
    
    # Extract current results
    current_batch_sizes = [result.batch_size for result in batch_results]
    current_moves_per_second = [result.moves_per_second for result in batch_results]
    
    # Create comparison directory if it doesn't exist
    results_dir = create_results_dir("graphs/comparisons")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract platform info for filenames
    platform_name = system_info["platform"].lower()
    processor_name = system_info["processor"].lower()
    
    # Find common batch sizes for comparison
    common_batch_sizes = []
    profile_moves = []
    current_moves = []
    
    for i, b in enumerate(current_batch_sizes):
        if b in profile['batch_sizes']:
            idx = profile['batch_sizes'].index(b)
            common_batch_sizes.append(b)
            profile_moves.append(profile['moves_per_second'][idx])
            current_moves.append(current_moves_per_second[i])
    
    if not common_batch_sizes:
        print("No common batch sizes found for comparison")
        return
    
    # Create a comparison plot
    plt.figure(figsize=(12, 6))
    width = 0.35
    x = np.arange(len(common_batch_sizes))
    
    plt.bar(x - width/2, profile_moves, width, label='Previous')
    plt.bar(x + width/2, current_moves, width, label='Current')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Moves per Second')
    plt.title('Performance Comparison: Previous vs Current')
    plt.xticks(x, common_batch_sizes)
    plt.legend()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = results_dir / f"{platform_name}_{processor_name}_comparison_{timestamp}.png"
    plt.savefig(comparison_path, dpi=120)
    plt.close()
    
    # Create a difference plot
    percentage_diff = [(c - p) / p * 100 for p, c in zip(profile_moves, current_moves)]
    
    plt.figure(figsize=(12, 6))
    plt.bar(common_batch_sizes, percentage_diff)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.xscale('log', base=2)
    plt.xlabel('Batch Size')
    plt.ylabel('Performance Difference (%)')
    plt.title('Performance Difference: (Current - Previous) / Previous * 100%')
    
    diff_path = results_dir / f"{platform_name}_{processor_name}_diff_{timestamp}.png"
    plt.savefig(diff_path, dpi=120)
    plt.close()
    
    print(f"Comparison plot saved to: {comparison_path}")
    print(f"Difference plot saved to: {diff_path}")
    
    # Print summary
    avg_diff = sum(percentage_diff) / len(percentage_diff)
    print(f"\n=== Performance Comparison Summary ===")
    print(f"Average performance difference: {avg_diff:.2f}%")
    
    if avg_diff > 5:
        print("Performance IMPROVED compared to previous profile")
    elif avg_diff < -5:
        print("Performance DEGRADED compared to previous profile")
    else:
        print("Performance is similar to previous profile")


def print_benchmark_summary(results: List[BatchBenchResult]) -> None:
    """Print a formatted summary of benchmark results."""
    valid_results = [r for r in results if r.valid]
    
    if not valid_results:
        print("No valid benchmark results to display")
        return
    
    print("\n=== Discovery Summary (Valid Results) ===")
    header = f"{'Batch':>7} | {'Moves/s':>12} | {'Games/s':>12} | {'Moves/s/G':>10} | {'Mem (GB)':>10} | " \
             f"{'Effic.':>12} | {'Avg Moves':>10} | {'Med Moves':>10} | {'Min Moves':>10} | {'Max Moves':>10}"
    print(header)
    print("-" * 100)
    
    for result in valid_results:
        print(f"{result.batch_size:>7} | {result.moves_per_second:>12,.2f} | {result.games_per_second:>12,.2f} | "
              f"{result.moves_per_second / result.batch_size:>10,.2f} | {result.memory_usage_gb:>10,.2f} | "
              f"{result.efficiency:>12,.2f}/GB | {result.avg_game_length:>10,.1f} | {result.median_game_length:>10,.1f} | "
              f"{result.min_game_length:>10} | {result.max_game_length:>10}")
    
    # Print optimal configurations
    moves_optimal = max(valid_results, key=lambda x: x.moves_per_second)
    games_optimal = max(valid_results, key=lambda x: x.games_per_second)
    efficiency_optimal = max(valid_results, key=lambda x: x.efficiency)
    
    print("\n=== Optimal Configurations (Discovered) ===")
    print(f"Best for moves/s: Batch size {moves_optimal.batch_size} with {moves_optimal.moves_per_second:,.2f} moves/s")
    print(f"Best for games/s: Batch size {games_optimal.batch_size} with {games_optimal.games_per_second:,.2f} games/s")
    print(f"Best for efficiency: Batch size {efficiency_optimal.batch_size} with {efficiency_optimal.efficiency:,.2f}/GB")


def select_batch_sizes_for_profile(results: List[BatchBenchResult], num_sizes: int = 4) -> List[int]:
    """Select a meaningful subset of batch sizes for inclusion in the profile."""
    valid_results = [r for r in results if r.valid]
    
    if len(valid_results) <= num_sizes:
        return [r.batch_size for r in valid_results]
    
    # Always include the smallest and largest batch sizes
    smallest = min(valid_results, key=lambda x: x.batch_size).batch_size
    largest = max(valid_results, key=lambda x: x.batch_size).batch_size
    
    # Include the batch size with the best raw performance
    best_perf = max(valid_results, key=lambda x: x.moves_per_second).batch_size
    
    # Include the batch size with the best games/s if different
    best_games = max(valid_results, key=lambda x: x.games_per_second).batch_size
    
    # Start with these candidates
    selected = [smallest, largest, best_perf, best_games]
    selected = list(set(selected))  # Remove duplicates
    
    # If we need more, add intermediate sizes evenly distributed
    batch_sizes = sorted([r.batch_size for r in valid_results])
    
    if len(selected) < num_sizes:
        # How many more do we need?
        needed = num_sizes - len(selected)
        
        # Filter out already selected batch sizes
        remaining = [b for b in batch_sizes if b not in selected]
        
        if remaining:
            # Take evenly spaced items
            indices = np.linspace(0, len(remaining) - 1, needed).astype(int)
            additional = [remaining[i] for i in indices]
            selected.extend(additional)
    
    return sorted(selected[:num_sizes])  # Return sorted list, capped at num_sizes 


class BaseBenchmark:
    """Base class for all benchmarks providing common functionality."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.system_info = get_system_info()
    
    def warmup_compilation(self, step_fn, states, num_warmup=4):
        """Common warmup and compilation logic."""
        print("Compiling and warming up...", flush=True)
        key = jax.random.PRNGKey(0)
        
        try:
            print("First compilation pass...", flush=True)
            key, subkey = jax.random.split(key)
            # Pass states as a tuple
            new_states = step_fn(subkey, states[0], states[1])
            jax.block_until_ready(new_states)
            print("Initial compilation successful", flush=True)
            
            print("Running warm-up iterations...", flush=True)
            for _ in range(num_warmup):
                key, subkey = jax.random.split(key)
                # Pass states as a tuple
                new_states = step_fn(subkey, new_states[0], new_states[1])
            jax.block_until_ready(new_states)
            print("Warm-up complete", flush=True)
            return new_states
            
        except Exception as e:
            print(f"Error during compilation/warm-up: {e}", flush=True)
            raise
    
    def run_benchmark_iteration(self, step_fn, states, pbar, start_time, max_duration):
        """Run a single benchmark iteration with progress tracking."""
        current_time = time.time()
        if current_time - start_time >= max_duration:
            return None
            
        try:
            # Execute step
            key = jax.random.PRNGKey(0)
            key, step_key = jax.random.split(key)
            new_states = step_fn(step_key, *states)
            jax.block_until_ready(new_states)
            
            # Update progress
            elapsed = current_time - start_time
            pbar.n = round(elapsed)
            pbar.refresh()
            
            return new_states
            
        except Exception as e:
            print(f"Error during benchmark iteration: {e}", flush=True)
            return None
    
    def save_profile(self, results: List[BatchBenchResult], extra_info: Dict[str, Any] = None):
        """Save benchmark results to a profile."""
        profile_data = {
            # System info
            "platform": self.system_info["platform"],
            "processor": self.system_info["processor"],
            "jaxlib_type": self.system_info["jaxlib_type"],
            "device_info": self.system_info["device_info"],
            "python_version": self.system_info["python_version"],
            "jax_version": self.system_info["jax_version"],
            
            # Benchmark info
            "name": self.name,
            "description": self.description,
            "timestamp": datetime.now().isoformat(),
            
            # Results
            "batch_sizes": [r.batch_size for r in results],
            "moves_per_second": [r.moves_per_second for r in results],
            "games_per_second": [r.games_per_second for r in results],
            "memory_usage_gb": [r.memory_usage_gb for r in results],
        }
        
        # Add any extra info
        if extra_info:
            profile_data.update(extra_info)
        
        # Create filename
        filename = f"{self.system_info['platform'].lower()}_{self.name.lower()}.json"
        filepath = PROFILE_DIR / filename
        
        # Save
        PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        print(f"Profile saved to {filepath}")
        return filepath
    
    def load_profile(self) -> Optional[Dict[str, Any]]:
        """Load a matching profile for this benchmark."""
        filename = f"{self.system_info['platform'].lower()}_{self.name.lower()}.json"
        filepath = PROFILE_DIR / filename
        
        if filepath.exists():
            print(f"Found matching profile: {filepath}", flush=True)
            with open(filepath, 'r') as f:
                return json.load(f)
        
        print(f"No matching profile found for {self.name}", flush=True)
        return None 

def print_summary_table(results: List[BatchBenchResult], title: str = None) -> None:
    """Print a formatted summary table of benchmark results."""
    if not results:
        print("No results to display")
        return
        
    if title:
        print(f"\n=== {title} ===")
    
    header = (
        f"{'Batch':>7} | {'Moves/s':>12} | {'Games/s':>12} | {'Mem (GB)':>10} | "
        f"{'Efficiency':>12} | {'Avg Moves':>10}"
    )
    print(header)
    print("-" * len(header))
    
    for result in results:
        print(
            f"{result.batch_size:>7} | "
            f"{format_human_readable(result.moves_per_second):>12} | "
            f"{format_human_readable(result.games_per_second):>12} | "
            f"{result.memory_usage_gb:>10.2f} | "
            f"{result.efficiency:>12.2f} | "
            f"{result.avg_game_length:>10.1f}"
        ) 