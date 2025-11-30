# MCTS Benchmark Results

Benchmark comparing per-child discount (new) vs global discount (old) implementation.
Date: 2025-11-30

## Test Configuration

- **Environment**: PGX Backgammon with `simple_doubles=True`
- **Evaluator**: StochasticMCTS with pip count evaluation function
- **Iterations per run**: 100
- **Runs per configuration**: 5 (CPU) / 5 (CUDA)
- **Metric**: MCTS iterations per second

## CPU Results (JAX CPU backend)

| Max Nodes | Old (global discount) | New (per-child discount) | % Change |
|-----------|----------------------|--------------------------|----------|
| 100       | 84.58 iter/s         | 86.71 iter/s            | +2.5%    |
| 500       | 81.73 iter/s         | 84.35 iter/s            | +3.2%    |
| 1000      | 78.59 iter/s         | 83.07 iter/s            | +5.7%    |
| 2000      | 76.76 iter/s         | 77.93 iter/s            | +1.5%    |

**CPU Summary**: ~2-6% faster with per-child discount (likely due to eliminating `jnp.any()` reduction)

## CUDA Results (NVIDIA GPU)

| Max Nodes | Old (global discount) | New (per-child discount) | % Change |
|-----------|----------------------|--------------------------|----------|
| 100       | 1515.32 iter/s       | 1512.02 iter/s          | -0.2%    |
| 500       | 1536.03 iter/s       | 1587.87 iter/s          | +3.4%    |
| 1000      | 1554.39 iter/s       | 1498.87 iter/s          | -3.6%    |
| 2000      | 1486.30 iter/s       | 1510.33 iter/s          | +1.6%    |

**CUDA Summary**: Performance-neutral (variations within noise margin)

## CPU vs GPU Comparison

### Speedup Factor (GPU/CPU ratio)

| Max Nodes | CPU iter/s | GPU iter/s | GPU Speedup |
|-----------|------------|------------|-------------|
| 100       | 86.71      | 1512.02    | **17.4x**   |
| 500       | 84.35      | 1587.87    | **18.8x**   |
| 1000      | 83.07      | 1498.87    | **18.0x**   |
| 2000      | 77.93      | 1510.33    | **19.4x**   |

**Average GPU speedup: ~18.4x faster than CPU**

### Total Iterations Analysis

For a typical run of 100 iterations × 5 runs = 500 total iterations:

| Platform | Max Nodes=100 | Max Nodes=500 | Max Nodes=1000 | Max Nodes=2000 |
|----------|---------------|---------------|----------------|----------------|
| CPU      | ~5.8 sec      | ~5.9 sec      | ~6.0 sec       | ~6.4 sec       |
| GPU      | ~0.33 sec     | ~0.31 sec     | ~0.33 sec      | ~0.33 sec      |

### Scaling Behavior

**CPU**: Shows slight performance degradation as max_nodes increases (~10% slower from 100 to 2000 nodes)

**GPU**: Relatively flat performance across all max_nodes values, suggesting:
- Memory bandwidth not yet saturated
- Good parallelization of tree operations
- Could potentially handle larger batch sizes efficiently

## Batch Size Recommendations

Based on benchmarks:

1. **Single-game MCTS** (current benchmark): GPU provides ~18x speedup
2. **Training with batched self-play**: GPU advantage should increase further due to:
   - Multiple parallel games can share GPU compute
   - Neural network evaluation is highly parallelizable
   - Memory transfer overhead amortized across batch

**Suggested batch sizes**:
- CPU: 1-4 parallel games (memory constrained)
- GPU: 32-256 parallel games (compute optimized)

## Conclusions

1. **Correctness improvement**: Per-child discount fixes incorrect value propagation when children have different player transitions
2. **CPU performance**: Slight improvement (~3% average)
3. **GPU performance**: No regression (within variance)
4. **GPU advantage**: ~18x faster than CPU for single-game MCTS iterations
