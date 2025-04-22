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
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import chex
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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


def print_summary_table(results: List[Any], title: str = "Benchmark Summary"):
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