# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TurboZero is a vectorized implementation of AlphaZero written in JAX. It provides Monte Carlo Tree Search (MCTS) with subtree persistence, batched replay memory, and a complete training/evaluation loop. The codebase is designed to be JIT-compiled, parallelized across GPUs, and fully compatible with `jax.vmap`, `jax.pmap`, and `jax.jit`.

## Project Goals

This fork extends the original [lowrollr/turbozero](https://github.com/lowrollr/turbozero) to build a **backgammon AI** using:

- **PGX Backgammon** as the game environment (use latest from [sile16/pgx](https://github.com/sile16/pgx) main branch)
- **Unified MCTS** to handle stochastic nodes, multi-steps per player, i.e, Gumbel AlphaZero approach. 
- **Multi-move per turn** architecture where each pip move is treated as a separate node in the MCTS tree

### Key Design Decisions

1. **Per-pip moves as actions**: Instead of treating a full backgammon turn (potentially 1-4 pip moves) as a single action, we treat each individual pip move as a state change. This dramatically reduces the action space to a manageable size.

2. **Stochastic nodes**: Dice rolls are modeled as stochastic nodes in the MCTS tree, with probabilities from `env.stochastic_action_probs`.

3. **Player continuity**: A player may take multiple consecutive moves within a turn before the opponent plays.

## Current Status & Next Steps

1. ✅ Consolidated all MCTS variants into single `UnifiedMCTS` class
2. ✅ Implemented temperature annealing (temperature as function of epoch)
3. ✅ Always subtree persistence (removed `persist_tree` flag)
4. ✅ Always Gumbel-Top-k at decision roots
5. ✅ Handles both stochastic and deterministic games
6. ⏳ Remove old MCTS files after migration complete
7. ⏳ Update training code to use UnifiedMCTS


### 1. Implemented expectimax value calculation for stochastic nodes (stochastic_mcts.py:414-530)

**Problem**: The original backpropagation simply propagated the sampled value through stochastic nodes, which doesn't properly account for the expected value across all possible outcomes.

**Fix**: Added `compute_expectimax_value()` method and modified `backpropagate()` to use it:
- At stochastic nodes, the value propagated is now the weighted sum of child Q-values based on their stochastic action probabilities: `V(node) = Σ P(action) * V(child)`
- This is theoretically correct for expectimax in stochastic games
- The method normalizes probabilities for existing children only (handles partial exploration)

### Design Decisions Confirmed as Correct

1. **Backpropagation child parameter**: The `backpropagate(tree, parent, child, value)` signature is correct - the child parameter is needed to calculate discount factors based on player changes, which can happen at any node (not just every transition like in base MCTS).

2. **Zero policy weights at stochastic nodes**: `stochastic_evaluate()` returns `jnp.zeros(branching_factor)` for policy weights. This is acceptable because:
   - Consistent return shape is required
   - `StochasticTrainer.collect()` correctly skips stochastic states when adding to replay buffer , need to make sure trainer doesn't train policy on a stochastic node but can train on the value. 

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
  - `mcts/unified_mcts.py`: **Primary MCTS implementation** - unified class handling all game types with Gumbel-Top-k and temperature annealing
  - `mcts/mcts.py`: (Legacy) Base MCTS implementation - will be removed
  - `mcts/stochastic_mcts.py`: (Legacy) MCTS for stochastic games - will be removed
  - `mcts/state.py`: Data structures: `MCTSTree`, `MCTSNode`, `StochasticMCTSNode`, `MCTSOutput`, `TraversalState`
  - `mcts/action_selection.py`: `MCTSActionSelector` (PUCTSelector) for tree traversal action selection
  - `mcts/gumbel.py`: Gumbel-Top-k sampling utilities for efficient root exploration
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

### Per-Child Discount in deterministic_action_selector

**Location**: `stochastic_mcts.py:161-188`

The `deterministic_action_selector` computes per-child discount factors based on player transitions:

```python
discounts = 1.0 - 2.0 * jnp.abs(current_player - child_players)
```

This formula produces:
- `1.0` when players are the same (0 vs 0 or 1 vs 1) - keep value
- `-1.0` when players differ (0 vs 1 or 1 vs 0) - invert value

JAX broadcasting handles the vector of discounts correctly when multiplied with Q-values in the PUCT selector. This correctly handles games where different actions at the same node can result in different player transitions.

## Code Style

- Make minimal, targeted edits - avoid unnecessary diffs
- Preserve existing formatting, comments, and whitespace
- Add new functions at end of files when possible
- Do not refactor existing code unless necessary for correctness

## JAX GPU Performance (2025-12-16)

### Critical: JIT Compilation Required

**Without JIT**: Each `env.step()` call triggers recompilation - 100 steps takes ~42 seconds
**With JIT**: Same 100 steps takes ~0.04 seconds (1000x speedup)

```python
# WRONG - triggers recompilation each call
for i in range(100):
    state = env.step(state, action, key)

# CORRECT - JIT compile once, reuse
step_jit = jax.jit(env.step)
_ = step_jit(state, 0, key)  # Warmup
for i in range(100):
    state = step_jit(state, action, key)
```

### Do NOT Force CPU Mode

Remove any `os.environ['JAX_PLATFORMS'] = 'cpu'` from test scripts. The RTX 4090 provides massive speedups.

### Vectorized Game Evaluation Pattern

For batch evaluation, use `jax.vmap` + `jax.lax.scan`:

```python
v_init = jax.vmap(env.init)
v_step = jax.vmap(env.step)

@jax.jit
def play_batch(key):
    keys = jax.random.split(key, batch_size)
    states = v_init(keys)
    final_rewards = jnp.zeros((batch_size, 2))

    def game_step(carry, _):
        states, final_rewards, key = carry
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, batch_size)

        actions = jax.vmap(policy_fn)(states)
        new_states = v_step(states, actions, step_keys)

        # IMPORTANT: Preserve rewards when games terminate
        # (pgx resets rewards to 0 when stepping terminated states)
        newly_terminated = (~states.terminated) & new_states.terminated
        final_rewards = jnp.where(
            newly_terminated[:, None],
            new_states.rewards,
            final_rewards
        )
        return (new_states, final_rewards, key), None

    (_, final_rewards, _), _ = jax.lax.scan(
        game_step, (states, final_rewards, key), None, length=300
    )
    return final_rewards
```

### pgx Terminated State Behavior

pgx environments reset `rewards` to `[0, 0]` when you step a terminated state. The state stays terminated but rewards are lost. You should call `env.init()` to start a new game.

**Why this matters:** With `jax.lax.scan` running fixed-length loops on batched games, some games finish early but keep getting stepped.

**Solution 1: Track rewards at termination**
```python
newly_terminated = (~states.terminated) & new_states.terminated
final_rewards = jnp.where(newly_terminated[:, None], new_states.rewards, final_rewards)
```

**Solution 2: Auto-reset wrapper (preferred)**
```python
def auto_reset_step(env):
    @jax.jit
    def wrapped_step(state, action, key):
        key, reset_key = jax.random.split(key)
        next_state = env.step(state, action, key)
        # If was already terminated, reset instead
        next_state = jax.lax.cond(
            state.terminated,
            lambda: env.init(reset_key),
            lambda: next_state
        )
        return next_state
    return wrapped_step
```

The auto-reset wrapper allows continuous `jax.lax.scan` loops where games automatically restart when they end. See also `pgx/pgx/experimental/wrappers.py` for pgx's wrapper implementations.

## pgx Pig Game Notes

### Action Space

- `num_actions = 6` but only actions 0-1 are meaningful for decisions
- Action 0 = Roll
- Action 1 = Hold (only legal when `legal_action_mask[1] = True`)

### State Machine

1. **Start of turn**: `_turn_total=0`, only Roll is legal
2. **After rolling**: `_is_stochastic=True`, both Roll and Hold become legal
3. **Hold action**: Banks `_turn_total` to `_scores[current_player]`, switches player
4. **Roll a 1**: Loses all `_turn_total`, switches player (resolved via random key)

### Stochastic States

When `_is_stochastic=True`:
- The state is a "chance node" where die outcome is pending
- Both Roll (continue) and Hold (bank points) are legal
- The random `key` passed to `env.step()` determines the die outcome
- Action choice (0 or 1) determines whether to continue rolling or hold

### Optimal Strategy

Precomputed optimal strategy available at `/tmp/pig_optimal_strategy.npz`:
- `V[i,j,k]` = win probability with my_score=i, opp_score=j, turn_total=k
- `policy[i,j,k]` = 1 if should hold, 0 if should roll
- Optimal first-player win rate: 52.7%

### Hold-20 Baseline

Simple but effective strategy:
```python
def hold20_action(turn_total, legal_mask):
    should_hold = (turn_total >= 20) & legal_mask[1]
    return jnp.where(should_hold, 1, 0)
```

## MCTS Parameters

### `persist_tree` Parameter

- **`persist_tree=True`** (default): Reuses tree between `evaluate()` calls. After taking action A, the subtree at A becomes the new root. More efficient for gameplay.

- **`persist_tree=False`**: Rebuilds tree from scratch each call. Use when you create new tree objects with `mcts.init()` anyway.

### MCTS with Untrained Network

An untrained network will make poor decisions (e.g., holding at turn_total=2 instead of 20). This is expected - the network needs training to learn good policies.

## StochasticMCTS Training (2025-12-26)

### PGX `step_stochastic` Observation Bug (FIXED in pgx 3.1.2)

**Problem** (pgx <= 3.1.0): `env.step_stochastic()` updated internal state but did NOT recompute the `observation` field, causing the NN to see stale observations.

**Status**: **FIXED in pgx 3.1.2** - `step_stochastic` now properly updates observations. No workaround needed.

### Value Loss Must Be Masked for Chance Nodes

**Problem**: Value targets at chance nodes depend on which random dice outcome was sampled during training, creating noisy targets. The same chance node observation might have value +1 or -1 depending on the sampled dice roll.

**Fix**: Mask out value loss for chance node samples in `core/training/loss_fns.py`:
```python
if experience.is_chance_node is not None:
    is_decision_node = ~experience.is_chance_node
    num_decision_samples = jnp.sum(is_decision_node)
    value_loss = jnp.where(
        num_decision_samples > 0,
        jnp.sum(value_loss_per_sample * is_decision_node) / num_decision_samples,
        0.0
    )
```

### Training Data Collection Strategy

**Problem**: Using `make_stochastic_aware_step_fn` collects samples from both decision nodes AND chance nodes. Since ~50% of samples are chance nodes (which have policy loss masked), this dilutes the training signal.

**Solution**: Use regular `env.step` for training data collection, which auto-resolves stochasticity with random sampling. This means:
- All training samples are from decision nodes
- Policy loss is computed for all samples
- Value targets are cleaner (only for decision nodes)
- StochasticMCTS still handles chance nodes properly during its internal MCTS search

```python
# StochasticMCTS training - use regular step function
stoch_result = train_and_evaluate(
    "StochasticMCTS",
    stoch_evaluator,
    stoch_evaluator_test,
    env,
    mlp_stoch,
    step_fn=make_step_fn(env),  # Regular env.step, NOT stochastic_aware
    num_epochs=num_epochs
)
```

### Comparison Results (Pig, 300 epochs, with Dirichlet noise)

| Method | Win Rate vs Optimal | Training Time | Policy Accuracy |
|--------|---------------------|---------------|-----------------|
| **StochasticMCTS** | **10%** | **532s** (3x faster) | **99%** |
| AlphaZero(MCTS) | 0% | 1528s | 84% |

**Key finding**: Regular MCTS (even with AlphaZero Dirichlet noise) completely fails to learn Pig. The random dice sampling during MCTS search creates noisy, inconsistent policy targets that the network can't converge on. StochasticMCTS explores all dice outcomes deterministically, providing stable targets.

### NN Architecture Note: Zero Observation Problem

**Problem**: Pig starts with observation `[0, 0, 0, 0]`. With ReLU activation, this produces all-zero hidden activations, causing dead neurons.

**Fix**: Add a constant feature to break symmetry:
```python
class SimpleMLP(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool = True):
        # Add constant feature to avoid all-zero input
        ones = jnp.ones((x.shape[0], 1))
        x = jnp.concatenate([x, ones], axis=-1)
        # Use GELU instead of ReLU for smoother gradients
        for _ in range(num_layers):
            x = nn.Dense(features=hidden_size)(x)
            x = nn.gelu(x)
        ...
```

## Test Files

- `tests/jax_pig_eval.py` - Vectorized Hold-20 vs Optimal evaluation (GPU, ~seconds for 5000 games)
- `tests/jax_mcts_pig_eval.py` - MCTS agent evaluation vs strategies
- `tests/test_stochastic_pig.py` - StochasticMCTS correctness tests
- `scripts/compare_mcts_games.py` - Multi-game MCTS comparison (Connect4, 2048, Pig, Backgammon)

## Extracting Model Params from TrainState

When extracting params from `Trainer.train_loop()` output for inference/evaluation:

1. **Remove device axis**: Trainer uses `jax.pmap` so params have shape `(num_devices, ...)`. Remove the leading axis:
   ```python
   raw_params = jax.tree.map(lambda x: x[0], train_state.params)
   ```

2. **Wrap for nn.apply**: Flax's `module.apply(variables, ...)` expects `{'params': params}`:
   ```python
   params = {'params': raw_params}
   ```

3. **Handle batch_stats** (if using BatchNorm):
   ```python
   if hasattr(train_state, 'batch_stats') and train_state.batch_stats is not None:
       raw_batch_stats = jax.tree.map(lambda x: x[0], train_state.batch_stats)
       params = {'params': raw_params, 'batch_stats': raw_batch_stats}
   ```

**Common error if you skip step 1:**
```
flax.errors.ScopeParamShapeError: expected shape (3, 3, 2, 64), but got (1, 3, 3, 2, 64)
```

**Common error if you skip step 2:**
```
flax.errors.ScopeCollectionNotFound: Tried to access "kernel" from collection "params" but the collection is empty
```

## Gumbel MCTS Implementation (2025-12-27)

### Overview

Implemented Gumbel MCTS variants based on "Policy improvement by planning with Gumbel" (ICLR 2022). The key insight is achieving same performance with 50-100x fewer simulations by using Gumbel-Top-k sampling at the root.

### New Files

- `core/evaluators/mcts/gumbel.py` - Core Gumbel utilities:
  - `gumbel_top_k()` - Sample k actions without replacement using Gumbel-Max trick
  - `GumbelRootSelector` - Gumbel-based action selector for MCTS root
  - `GumbelActionScheduler` - Schedules which actions to search

- `core/evaluators/mcts/gumbel_mcts.py` - `GumbelMCTS` class for deterministic games
- `core/evaluators/mcts/gumbel_stochastic_mcts.py` - `GumbelStochasticMCTS` for stochastic games

### Usage

```python
from core.evaluators.mcts.gumbel_stochastic_mcts import GumbelStochasticMCTS

evaluator = GumbelStochasticMCTS(
    eval_fn=make_nn_eval_fn(network, state_to_nn_input),
    action_selector=PUCTSelector(),
    stochastic_action_probs=stochastic_probs,
    policy_size=env.num_actions,
    num_iterations=16,  # Far fewer than standard MCTS!
    max_nodes=66,
    gumbel_k=16,  # Number of actions to sample at root
    decision_step_fn=decision_step_fn,
    stochastic_step_fn=stochastic_step_fn,
)
```

### JAX Tracing Notes

**jax.lax.top_k for Gumbel selection**: Use `jax.lax.top_k` instead of manual slicing for JIT compatibility:
```python
# WRONG - dynamic slicing in JIT
sorted_indices = jnp.argsort(-perturbed)
selected_actions = sorted_indices[:k]  # Fails if k is traced

# CORRECT - use jax.lax.top_k
_, selected_actions = jax.lax.top_k(perturbed, k)
```

**Static vs traced values**: Use Python `min()` instead of `jnp.minimum()` for values that should be static:
```python
# WRONG - creates traced value
effective_k = jnp.minimum(self.gumbel_k, self.policy_size)

# CORRECT - static computation
effective_k = min(self.gumbel_k, self.policy_size)
```

### Comparison Results (2048, 30 epochs)

| Configuration | Time | Speed | Eval Score |
|---------------|------|-------|------------|
| Standard StochasticMCTS (50 sims) | 166s | 0.18 epochs/sec | ~4136 |
| Gumbel StochasticMCTS (16 sims) | 87s | 0.34 epochs/sec | ~4104 |
| Gumbel StochasticMCTS (8 sims) | ~60s | ~0.5 epochs/sec | ~4104 |

**Key findings**:
- Gumbel variant is ~2x faster due to fewer simulations
- Training score similar, but policy accuracy plateaus at ~55-60% (vs 97%+ for standard)
- The lower policy accuracy suggests the action cycling strategy may need tuning
- For now, recommend standard StochasticMCTS for production training

## UnifiedMCTS - Consolidated MCTS Implementation (2025-12-31)

### Overview

Consolidated all MCTS variants (MCTS, StochasticMCTS, GumbelMCTS, GumbelStochasticMCTS) into a single `UnifiedMCTS` class that handles all game types.

**File**: `core/evaluators/mcts/unified_mcts.py`

### Key Features

1. **Always Gumbel-Top-k at decision roots**: Efficient exploration by sampling k actions at root
2. **Always subtree persistence**: Tree is always reused between evaluate() calls
3. **Stochastic node support**: Handles dice rolls, tile spawns, etc.
4. **Temperature annealing**: Temperature can be a function of epoch for training schedules
5. **1-2 player support**: Correct Q-value handling with player perspective

### Usage

```python
from core.evaluators.mcts.unified_mcts import (
    UnifiedMCTS,
    linear_temp_schedule,
    exponential_temp_schedule,
)
from core.evaluators.mcts.action_selection import PUCTSelector

# Basic usage with constant temperature
mcts = UnifiedMCTS(
    eval_fn=nn_eval_fn,
    action_selector=PUCTSelector(),
    policy_size=env.num_actions,
    max_nodes=200,
    num_iterations=50,
    decision_step_fn=make_decision_step_fn(env),
    stochastic_step_fn=make_stochastic_step_fn(env),  # None for deterministic games
    stochastic_action_probs=env.stochastic_action_probs,  # None for deterministic
    gumbel_k=16,
    temperature=1.0,  # Or use a schedule (see below)
)

# Temperature annealing example (linear decay from 1.0 to 0.0 over 100 epochs)
mcts = UnifiedMCTS(
    ...,
    temperature=linear_temp_schedule(start_temp=1.0, end_temp=0.0, total_epochs=100),
)

# In training loop, update epoch for temperature annealing
for epoch in range(num_epochs):
    mcts.set_epoch(epoch)
    # ... training code
```

### Temperature Schedules

Three built-in schedules:

1. **Linear**: `linear_temp_schedule(start, end, total_epochs)` - Linear interpolation
2. **Exponential**: `exponential_temp_schedule(start, end, decay_rate)` - Exponential decay
3. **Step**: `step_temp_schedule(temps, boundaries)` - Discrete steps

Custom schedules: Pass any `Callable[[int], float]` that takes epoch and returns temperature.

### Configuration Examples

**Backgammon (Stochastic 2-Player)**:
```python
mcts = UnifiedMCTS(
    eval_fn=nn_eval_fn,
    action_selector=PUCTSelector(),
    policy_size=156,
    max_nodes=200,
    num_iterations=50,
    gumbel_k=16,
    stochastic_action_probs=env.stochastic_action_probs,
    decision_step_fn=make_bg_decision_step_fn(env),
    stochastic_step_fn=make_bg_stochastic_step_fn(env),
)
```

**2048 (Stochastic 1-Player)**:
```python
mcts = UnifiedMCTS(
    eval_fn=nn_eval_fn,
    action_selector=PUCTSelector(),
    policy_size=4,
    max_nodes=150,
    num_iterations=50,
    gumbel_k=4,
    stochastic_action_probs=tile_spawn_probs,
    decision_step_fn=make_2048_decision_step_fn(env),
    stochastic_step_fn=make_2048_stochastic_step_fn(env),
)
```

**TicTacToe (Deterministic 2-Player)**:
```python
mcts = UnifiedMCTS(
    eval_fn=nn_eval_fn,
    action_selector=PUCTSelector(),
    policy_size=9,
    max_nodes=100,
    num_iterations=25,
    gumbel_k=9,
    stochastic_action_probs=None,  # Deterministic
    decision_step_fn=ttt_step_fn,
    stochastic_step_fn=None,
)
```

### Root Embedding After step()

When calling `step(tree, action)`, the subtree rooted at the child becomes the new tree. The root embedding is preserved from when the child was expanded, so it correctly matches the game state.

**Important**: The caller should pass the matching `env_state` to the next `evaluate()` call.

## 2048 Symmetry Augmentation (2025-12-27)

### Overview

2048 has 8 symmetries (D4 group): 4 rotations × 2 reflections. Data augmentation multiplies training data by 8x, significantly improving sample efficiency.

### Implementation

**File**: `core/training/augmentation.py`

```python
from core.training.augmentation import make_2048_symmetry_transforms

# Get list of 7 transforms (excluding identity)
transforms = make_2048_symmetry_transforms()

# Pass to Trainer
trainer = Trainer(
    ...,
    data_transform_fns=transforms,
)
```

### Key Points

1. **Action transformation**: When rotating the board, actions must also rotate:
   - rot90: up→right, right→down, down→left, left→up
   - flip_h: left↔right

2. **pgx State API**: Use `state.replace(observation=new_obs)` not `state._replace()`

### Throughput Analysis

**Paper reference (Stochastic MuZero on V100)**:
- 10 billion frames in 8 days = ~14,500 frames/sec
- RTX 4090 expected: ~29,000 frames/sec (2x V100)

**Current implementation (RTX 4090)**:
- Without augmentation: ~4,400 frames/sec
- With 8x symmetry: ~35,300 frames/sec (8x effective training data)

**Time to 10B frames**:
- Without aug: ~26 days
- With 8x aug: ~3.3 days (meets paper target!)

### Note on Optimization

Current bottlenecks:
1. Sequential MCTS (not batched across games)
2. Small batch size (32 vs 1024+ in paper)
3. Python-level game loop

With full optimization, expect 10-50x additional speedup.
