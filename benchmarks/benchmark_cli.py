"""Common command line argument parsing for benchmark scripts."""

import argparse
from typing import Optional
from benchmarks.benchmark_common import DEFAULT_MEMORY_LIMIT_GB, DEFAULT_BENCHMARK_DURATION

def parse_batch_sizes(batch_sizes_str: Optional[str]) -> Optional[list[int]]:
    """Parse comma-separated batch sizes string into a list of integers.
    
    Args:
        batch_sizes_str: Comma-separated string of integers, or None
        
    Returns:
        List of integers if string provided, None otherwise
        
    Raises:
        ValueError: If string cannot be parsed into valid integers
    """
    if not batch_sizes_str:
        return None
        
    try:
        return [int(b.strip()) for b in batch_sizes_str.split(',')]
    except ValueError as e:
        raise ValueError(f"Error parsing batch sizes: {e}. Format should be comma-separated integers (e.g., '1,2,4,8,16').")

def add_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add common benchmark arguments to an argument parser.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument("--memory-limit", type=float, default=DEFAULT_MEMORY_LIMIT_GB,
                       help=f"Memory limit in GB (default: {DEFAULT_MEMORY_LIMIT_GB})")
    parser.add_argument("--duration", type=int, default=DEFAULT_BENCHMARK_DURATION,
                       help=f"Maximum duration of each batch size test in seconds (default: {DEFAULT_BENCHMARK_DURATION})")
    parser.add_argument("--batch-sizes", type=str,
                       help="Comma-separated list of batch sizes to test (e.g., '1,2,4,8,16')")
    parser.add_argument("--single-batch", type=int, 
                       help="Test only a specific batch size instead of discovery")
    parser.add_argument("--validate", action="store_true",
                       help="Validate against existing benchmark profile")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")

def add_mcts_specific_args(parser: argparse.ArgumentParser, default_num_iterations: int) -> None:
    """Add MCTS-specific arguments to an argument parser.
    
    Args:
        parser: ArgumentParser to add arguments to
        default_num_iterations: Default number of MCTS iterations/simulations
    """
    parser.add_argument("--num-simulations", type=int, default=default_num_iterations,
                       help=f"Number of MCTS simulations per move (default: {default_num_iterations})")
    parser.add_argument("--force", action="store_true",
                       help="Force overwrite of existing benchmark profile")

def create_benchmark_parser(description: str, 
                          include_mcts_args: bool = False,
                          default_num_iterations: int = 200) -> argparse.ArgumentParser:
    """Create an argument parser with common benchmark arguments.
    
    Args:
        description: Description of the benchmark script
        include_mcts_args: Whether to include MCTS-specific arguments
        default_num_iterations: Default number of MCTS iterations (only used if include_mcts_args=True)
        
    Returns:
        ArgumentParser configured with common benchmark arguments
    """
    parser = argparse.ArgumentParser(description=description)
    add_common_benchmark_args(parser)
    
    if include_mcts_args:
        add_mcts_specific_args(parser, default_num_iterations)
        
    return parser 