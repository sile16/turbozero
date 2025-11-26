# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TurboZero is a vectorized implementation of AlphaZero written in JAX. It provides Monte Carlo Tree Search (MCTS) with subtree persistence, batched replay memory, and a complete training/evaluation loop. The codebase is designed to be JIT-compiled, parallelized across GPUs, and fully compatible with `jax.vmap`, `jax.pmap`, and `jax.jit`.

## Project Goals

This fork extends the original [lowrollr/turbozero](https://github.com/lowrollr/turbozero) to build a **backgammon AI** using:

- **PGX Backgammon** as the game environment (use latest from [sile16/pgx](https://github.com/sile16/pgx) main branch)
- **Stochastic MCTS** to handle dice rolls as chance nodes
- **Multi-move per turn** architecture where each pip move is treated as a separate node in the MCTS tree

### Key Design Decisions

1. **Per-pip moves as actions**: Instead of treating a full backgammon turn (potentially 2-4 pip moves) as a single action, we treat each individual pip move as a state change. This dramatically reduces the action space to a manageable size.

2. **Stochastic nodes**: Dice rolls are modeled as stochastic nodes in the MCTS tree, with probabilities from `env.stochastic_action_probs`.

3. **Player continuity**: A player may take multiple consecutive moves within a turn before the opponent plays.

## Current Status & Next Steps

1. ✅ Implemented `StochasticMCTS` extending base MCTS for stochastic games
2. ✅ **Review current codebase** - verified correctness of stochastic MCTS implementation
3. ✅ **Review tests** - tests properly validate the stochastic behavior
4. ✅ **Run tests** - all tests passing

## Recent Changes (2025-11-25)

### 1. Fixed `is_node_stochastic` to return JAX boolean (stochastic_mcts.py:531-553)

**Problem**: The static methods `is_node_stochastic()` and `is_node_idx_stochastic()` returned Python `False` when the `is_stochastic` attribute didn't exist. This could cause tracing issues in JIT-compiled code.

**Fix**: Now returns `jnp.array(False)` instead of Python `False`:
```python
is_stochastic = getattr(node.embedding, 'is_stochastic', None)
if is_stochastic is None:
    return jnp.array(False)
return is_stochastic
```

### 2. Implemented expectimax value calculation for stochastic nodes (stochastic_mcts.py:414-530)

**Problem**: The original backpropagation simply propagated the sampled value through stochastic nodes, which doesn't properly account for the expected value across all possible outcomes.

**Fix**: Added `compute_expectimax_value()` method and modified `backpropagate()` to use it:
- At stochastic nodes, the value propagated is now the weighted sum of child Q-values based on their stochastic action probabilities: `V(node) = Σ P(action) * V(child)`
- This is theoretically correct for expectimax in stochastic games
- The method normalizes probabilities for existing children only (handles partial exploration)

### Design Decisions Confirmed as Correct

1. **Backpropagation child parameter**: The `backpropagate(tree, parent, child, value)` signature is correct - the child parameter is needed to calculate discount factors based on player changes, which can happen at any node (not just every transition like in base MCTS).

2. **Zero policy weights at stochastic nodes**: `stochastic_evaluate()` returns `jnp.zeros(branching_factor)` for policy weights. This is acceptable because:
   - Consistent return shape is required
   - `StochasticTrainer.collect()` correctly skips stochastic states when adding to replay buffer

3. **visit_node logic**: In `iterate()`, the node is visited once via `new_node()`/`visit_node()`, then backprop updates ancestors only (starting from parent). This is correct.

## Commands

```bash
# Install dependencies (uses poetry, NOT pip)
poetry install

# Run tests with verbose output
poetry run pytest -sv

# Run a specific test file
poetry run pytest tests/test_stochastic_mcts1.py -sv

# Launch Jupyter kernel
poetry run python -m ipykernel install --user --name turbozero

# For GPU with CUDA 12, add JAX CUDA source then install:
poetry source add jax https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
poetry add jax[cuda12]
```

## Architecture

### Core Components

- **`core/evaluators/`**: Evaluation strategies for game states
  - `evaluator.py`: Base `Evaluator` class - evaluates environment states and returns actions with policy weights
  - `mcts/mcts.py`: Batched MCTS implementation operating on `MCTSTree` state objects
  - `mcts/stochastic_mcts.py`: Extension of MCTS supporting stochastic nodes (e.g., for backgammon dice rolls)
  - `mcts/state.py`: Data structures: `MCTSTree`, `MCTSNode`, `MCTSOutput`, `TraversalState`, `BackpropState`
  - `mcts/action_selection.py`: `MCTSActionSelector` for tree traversal action selection
  - `alphazero.py`: AlphaZero wrapper adding Dirichlet noise exploration

- **`core/training/`**: Training infrastructure
  - `train.py`: `Trainer` class implementing the AlphaZero training loop with self-play collection, replay buffer, and gradient updates
  - `stochastic_train.py`: Training utilities for stochastic environments
  - `loss_fns.py`: Loss functions for policy and value heads

- **`core/memory/`**: Experience replay
  - `replay_memory.py`: `EpisodeReplayBuffer` storing trajectories with `BaseExperience` and `ReplayBufferState`

- **`core/networks/`**: Neural network architectures
  - `azresnet.py`: AlphaZero ResNet with residual blocks, policy head, and value head
  - `mlp.py`: Simple MLP alternative

- **`core/trees/`**: Tree data structure utilities
  - `tree.py`: Tree initialization and manipulation

- **`core/types.py`**: Type definitions including `StepMetadata`, `EnvStepFn`, `EvalFn`, etc.

### Key Data Flow

1. **Self-play**: `Trainer` runs batched episodes using `Evaluator.evaluate()` to get actions
2. **MCTS search**: Each evaluation performs `num_iterations` tree traversals with select→expand→evaluate→backpropagate
3. **Experience storage**: Game trajectories stored in `EpisodeReplayBuffer`
4. **Training**: Sample batches from replay buffer, compute loss, update network

## JAX Coding Guidelines

When modifying JIT-compiled functions:

- **Avoid Python control flow**: Use `jax.lax.cond`, `jax.lax.switch`, or vectorized operations instead of `if-then-else`
- **Use JAX loops**: Prefer `jax.lax.fori_loop` over Python `for` loops
- **Manage random keys**: Properly split and propagate keys for reproducibility
- **Maintain functional purity**: No side effects in JIT-compiled code

## Stochastic MCTS

The codebase supports stochastic games (like backgammon) via `StochasticMCTS`:

- `env.stochastic_action_probs` provides probabilities for stochastic actions
- Stochastic node values are computed as weighted sums of leaf values
- The `step` function handles both player actions and stochastic actions

## Code Style

- Make minimal, targeted edits - avoid unnecessary diffs
- Preserve existing formatting, comments, and whitespace
- Add new functions at end of files when possible
- Do not refactor existing code unless necessary for correctness
