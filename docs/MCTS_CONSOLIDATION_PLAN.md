# MCTS Consolidation Plan

## Goal
Collapse all MCTS variants into a single, simplified implementation:
- **One MCTS class**: `UnifiedMCTS`
- **Always persist_tree=True** (remove flag)
- **Always Gumbel-Top-k** at decision root nodes
- **Always support stochastic transitions** (non-stochastic uses identity function)
- **1 or 2 players only**
- **No Python if statements in JAX compiled code**

## Current State (Files to Remove/Consolidate)

```
core/evaluators/mcts/
├── mcts.py                    # Base MCTS - REMOVE
├── stochastic_mcts.py         # Stochastic MCTS - KEEP AS BASE
├── gumbel_mcts.py             # Gumbel MCTS - MERGE INTO unified
├── gumbel_stochastic_mcts.py  # Gumbel Stochastic - MERGE INTO unified
├── gumbel.py                  # Gumbel utilities - KEEP
├── action_selection.py        # PUCTSelector - KEEP
├── state.py                   # Tree state - KEEP
└── __init__.py                # Exports - UPDATE
```

## Target State

```
core/evaluators/mcts/
├── unified_mcts.py            # Single MCTS implementation
├── gumbel.py                  # Gumbel utilities (unchanged)
├── action_selection.py        # PUCTSelector for non-root nodes
├── state.py                   # Tree state (unchanged)
└── __init__.py                # Export UnifiedMCTS
```

## Key Design Decisions

### 1. Always Persist Tree
- Remove `persist_tree` parameter entirely
- Always call `get_subtree(action)` in `step()`
- Tree is always reused between evaluate() calls
- For stochastic games: caller passes correct `env_state` and we update root embedding

### 2. Always Use Gumbel at Decision Root
- Sample k actions using Gumbel-Top-k at decision node roots
- Use minimum-visit cycling (not Sequential Halving) for simplicity
- Non-root nodes use standard PUCT selection
- `gumbel_k` parameter defaults to 16

### 3. Stochastic Support Built-In
- Always have `decision_step_fn` and `stochastic_step_fn`
- Non-stochastic games pass a placeholder: `stochastic_step_fn = lambda s, a, k: (s, metadata)`
- `stochastic_action_probs` defaults to None (deterministic)
- JAX conditionals handle stochastic vs decision nodes

### 4. Two-Player Perspective Handling
- Keep `calculate_discount_factor()` for backpropagation
- Keep per-child discount in PUCTSelector
- Support current_player in embeddings

### 5. Root Update Strategy with persist_tree=True
**Critical change**: When tree is persisted, root may have stale embedding.

```python
# After step(tree, action):
# - New root is the child node
# - Child's embedding may be from opponent's expansion
# - We MUST update root embedding to match current env_state

# In evaluate():
# Always update root embedding with current env_state
# But preserve Q-values and visit counts from previous search
```

## Implementation Plan

### Phase 1: Create UnifiedMCTS Shell
1. Create `unified_mcts.py` with single class
2. Constructor parameters:
   ```python
   def __init__(
       self,
       eval_fn: EvalFn,
       action_selector: MCTSActionSelector,  # For non-root nodes
       policy_size: int,
       max_nodes: int,
       num_iterations: int,
       gumbel_k: int = 16,
       stochastic_action_probs: Optional[chex.Array] = None,  # None = deterministic
       decision_step_fn: DecisionStepFn,
       stochastic_step_fn: Optional[StochasticStepFn] = None,  # None = identity
       temperature: float = 1.0,
       dirichlet_alpha: float = 0.3,
       dirichlet_epsilon: float = 0.25,
   )
   ```

### Phase 2: Core Methods
1. `init()` - Initialize tree
2. `evaluate()` - Main search loop with Gumbel at root
3. `step()` - Always use get_subtree (no reset option)
4. `update_root()` - Always update embedding, preserve stats
5. `iterate()` - Single MCTS iteration
6. `backpropagate()` - With player perspective handling

### Phase 3: Stochastic Handling
1. `_is_chance_node()` - Check if node is stochastic
2. `_expand_decision_child()` - Expand decision node
3. `_expand_stochastic_child()` - Expand chance node
4. `cond_action_selector()` - JAX cond for stochastic vs decision

### Phase 4: Remove Old Files
1. Delete `mcts.py`
2. Delete `stochastic_mcts.py`
3. Delete `gumbel_mcts.py`
4. Delete `gumbel_stochastic_mcts.py`
5. Update imports everywhere

### Phase 5: Update Training Code
1. Update `Trainer` to use `UnifiedMCTS`
2. Update `StochasticTrainer` (or merge into single Trainer)
3. Update all training scripts

### Phase 6: Update Tests
1. Update all MCTS tests to use `UnifiedMCTS`
2. Remove tests for removed classes
3. Add tests for unified behavior

## JAX Compatibility Notes

### No Python if/else in JIT
Instead of:
```python
if state._is_stochastic:
    return stochastic_step(...)
else:
    return decision_step(...)
```

Use:
```python
return jax.lax.cond(
    state._is_stochastic,
    lambda: stochastic_step(...),
    lambda: decision_step(...)
)
```

### No Python for loops over dynamic ranges
Instead of:
```python
for i in range(num_iterations):
    tree = iterate(tree)
```

Use:
```python
def scan_body(carry, _):
    tree, key = carry
    key, iter_key = jax.random.split(key)
    return (iterate(iter_key, tree), key), None

(tree, _), _ = jax.lax.scan(scan_body, (tree, key), None, length=num_iterations)
```

### Shape consistency
All arrays must have consistent shapes regardless of branch taken:
```python
# Good: both branches return same shape
result = jax.lax.cond(
    condition,
    lambda: jnp.zeros(10),
    lambda: jnp.ones(10)
)

# Bad: different shapes
result = jax.lax.cond(
    condition,
    lambda: jnp.zeros(10),
    lambda: jnp.zeros(5)  # Different shape!
)
```

## persist_tree=True: Key Implementation Details

### Root Embedding Correctness
The root embedding will be correct because:
1. When we call `step(tree, action)`, we get subtree rooted at child
2. The child's embedding was set during expansion using the correct step function
3. The step functions (`decision_step_fn`, `stochastic_step_fn`) produce correct game states
4. Caller passes matching `env_state` to `evaluate()` which should match root embedding

**Tests needed to verify:**
- After `step()`, root embedding matches expected game state
- Root embedding has correct `current_player`
- Root embedding has correct `legal_action_mask`
- Two-player perspective is preserved correctly

### Step Method (Always Persist)
```python
def step(self, tree: MCTSTree, action: int) -> MCTSTree:
    """Update tree after taking action. Always persists subtree."""
    return tree.get_subtree(action)
```

### Root Update in evaluate()
```python
def update_root(self, key, tree, env_state, params, root_metadata):
    """Update root node. Only called if root has no visits (empty tree)."""
    # Root with visits already has correct embedding from expansion
    # Only update if tree is fresh (n == 0)
    root_n = tree.data.n[tree.ROOT_INDEX]

    return jax.lax.cond(
        root_n == 0,
        lambda: self._initialize_root(key, tree, env_state, params, root_metadata),
        lambda: tree  # Keep existing root with its embedding
    )
```

## Configuration Examples

### Backgammon (Stochastic 2-Player)
```python
mcts = UnifiedMCTS(
    eval_fn=nn_eval_fn,
    action_selector=PUCTSelector(),
    policy_size=156,  # Backgammon actions
    max_nodes=200,
    num_iterations=50,
    gumbel_k=16,
    stochastic_action_probs=env.stochastic_action_probs,  # 21 dice outcomes
    decision_step_fn=make_bg_decision_step_fn(env),
    stochastic_step_fn=make_bg_stochastic_step_fn(env),
)
```

### TicTacToe (Deterministic 2-Player)
```python
mcts = UnifiedMCTS(
    eval_fn=nn_eval_fn,
    action_selector=PUCTSelector(),
    policy_size=9,
    max_nodes=100,
    num_iterations=25,
    gumbel_k=9,  # All actions
    stochastic_action_probs=None,  # Deterministic
    decision_step_fn=ttt_step_fn,
    stochastic_step_fn=None,  # Not used
)
```

### 2048 (Stochastic 1-Player)
```python
mcts = UnifiedMCTS(
    eval_fn=nn_eval_fn,
    action_selector=PUCTSelector(),
    policy_size=4,  # Up, Down, Left, Right
    max_nodes=150,
    num_iterations=50,
    gumbel_k=4,
    stochastic_action_probs=tile_spawn_probs,  # 32 outcomes
    decision_step_fn=make_2048_decision_step_fn(env),
    stochastic_step_fn=make_2048_stochastic_step_fn(env),
)
```

## Test Cleanup Plan

### Tests to Remove (for deleted classes)
- Tests specifically for `MCTS` class (replaced by UnifiedMCTS)
- Tests for `persist_tree=False` behavior (no longer supported)
- Tests for `GumbelMCTS` and `GumbelStochasticMCTS` (merged into UnifiedMCTS)

### Tests to Update
- All `test_stochastic_mcts*.py` → use UnifiedMCTS
- Training tests → use UnifiedMCTS
- Integration tests → use UnifiedMCTS

### New Tests Needed
1. **Root Embedding Verification**
   ```python
   def test_root_embedding_after_step():
       """Verify root embedding matches expected state after step()."""
       # 1. Evaluate from initial state
       # 2. Take action, step tree
       # 3. Step game environment with same action
       # 4. Verify tree root embedding matches env state
   ```

2. **Two-Player Perspective**
   ```python
   def test_two_player_subtree_persistence():
       """Verify Q-values are correct from both players' perspectives."""
       # 1. Player 0 evaluates, takes action
       # 2. Step tree
       # 3. Player 1 evaluates from persisted tree
       # 4. Verify Q-values make sense for Player 1
   ```

3. **Stochastic Transition Handling**
   ```python
   def test_stochastic_root_after_step():
       """Verify tree handles stochastic roots correctly after step()."""
       # 1. Decision node → action → stochastic node
       # 2. Step tree
       # 3. New root should be stochastic
       # 4. Verify stochastic action selection works
   ```

4. **Gumbel Selection**
   ```python
   def test_gumbel_topk_at_root():
       """Verify Gumbel-Top-k is used at decision roots."""
       # 1. Evaluate with gumbel_k=4
       # 2. Verify only ~4 actions get visits
   ```

## Timeline Estimate
- Phase 1-2: Core implementation
- Phase 3: Stochastic handling
- Phase 4: Remove old files
- Phase 5: Update training
- Phase 6: Tests

## Sources
- [Policy improvement by planning with Gumbel (ICLR 2022)](https://openreview.net/forum?id=bERaNdoegnO)
- [Google DeepMind mctx library](https://github.com/google-deepmind/mctx)
- [LightZero Gumbel MuZero](https://github.com/opendilab/LightZero)
