# Proposal: Dynamic Stochastic Probability Functions in PGX Environments

## Summary

For stochastic games like 2048, Backgammon, and card games, the probability distribution over chance outcomes depends on the current game state. We propose adding a `stochastic_action_probs(state)` function to PGX environments to properly support MCTS algorithms that require accurate probability distributions at chance nodes.

## Problem Statement

### Current Situation

In games with stochastic elements, the probability of each random outcome often depends on the current state:

| Game | Stochastic Event | State Dependency |
|------|------------------|------------------|
| **2048** | New tile placement | Only empty cells can receive tiles |
| **Backgammon** | Dice roll | Usually uniform, but doubling cube affects options |
| **Poker/Card Games** | Card draw | Depends on remaining deck composition |
| **Pig (dice game)** | Dice roll | Uniform (state-independent) |

### The Issue

Currently, MCTS implementations must:
1. Access internal state fields (e.g., `state._board` in 2048) to compute probabilities
2. Hardcode game-specific logic outside the environment
3. Risk using incorrect probabilities when state structure changes

Example of current workaround in TurboZero:
```python
def compute_dynamic_stochastic_probs(state):
    """Hacky: accesses internal _board field"""
    board = state._board  # Internal field - may change!
    flat_board = board.flatten()
    empty_mask = (flat_board == 0).astype(jnp.float32)
    num_empty = jnp.maximum(jnp.sum(empty_mask), 1.0)

    probs = jnp.zeros(32)
    for pos in range(16):
        is_empty = empty_mask[pos]
        probs = probs.at[pos * 2].set(is_empty * 0.9 / num_empty)
        probs = probs.at[pos * 2 + 1].set(is_empty * 0.1 / num_empty)
    return probs
```

This is fragile and requires users to understand internal state representations.

## Proposed Solution

### Add `stochastic_action_probs` Method to Environments

```python
class Env:
    # Existing attributes
    num_actions: int          # Number of decision actions

    # New attributes for stochastic games
    is_stochastic: bool       # True if game has chance nodes
    num_stochastic_actions: int  # Number of possible stochastic outcomes

    def stochastic_action_probs(self, state: State) -> chex.Array:
        """Compute probability distribution over stochastic outcomes.

        Args:
            state: Current game state (at a chance node)

        Returns:
            Array of shape (num_stochastic_actions,) with probabilities
            summing to 1.0. Invalid outcomes have probability 0.

        Note:
            Only meaningful when state.is_stochastic is True.
        """
        ...
```

### Example Implementations

#### 2048
```python
# In pgx/2048.py
class Env2048(Env):
    is_stochastic = True
    num_stochastic_actions = 32  # 16 positions × 2 tile values

    def stochastic_action_probs(self, state: State) -> chex.Array:
        """Probability of (position, tile_value) outcomes."""
        board = state._board
        empty_mask = (board.flatten() == 0).astype(jnp.float32)
        num_empty = jnp.maximum(jnp.sum(empty_mask), 1.0)

        probs = jnp.zeros(32)
        for pos in range(16):
            # outcome = pos * 2 + tile_idx (0=tile2, 1=tile4)
            probs = probs.at[pos * 2].set(empty_mask[pos] * 0.9 / num_empty)
            probs = probs.at[pos * 2 + 1].set(empty_mask[pos] * 0.1 / num_empty)
        return probs
```

#### Backgammon
```python
class EnvBackgammon(Env):
    is_stochastic = True
    num_stochastic_actions = 21  # 21 unique dice combinations

    def stochastic_action_probs(self, state: State) -> chex.Array:
        """Uniform distribution over dice rolls."""
        # 6 doubles (1/36 each) + 15 non-doubles (2/36 each)
        probs = jnp.array([
            1/36, 1/36, 1/36, 1/36, 1/36, 1/36,  # doubles
            2/36, 2/36, 2/36, 2/36, 2/36,         # (1,2), (1,3), ...
            2/36, 2/36, 2/36, 2/36,               # (2,3), (2,4), ...
            2/36, 2/36, 2/36,                     # (3,4), (3,5), ...
            2/36, 2/36,                           # (4,5), (4,6)
            2/36                                   # (5,6)
        ])
        return probs
```

#### Pig (Dice Game)
```python
class EnvPig(Env):
    is_stochastic = True
    num_stochastic_actions = 6  # Die faces 1-6

    def stochastic_action_probs(self, state: State) -> chex.Array:
        """Uniform distribution - independent of state."""
        return jnp.ones(6) / 6
```

### Integration with MCTS

With this interface, MCTS implementations become cleaner:

```python
class UnifiedMCTS:
    def __init__(self, env, ...):
        self.env = env
        self.is_stochastic_game = env.is_stochastic
        self.stochastic_size = env.num_stochastic_actions

    def _get_stochastic_probs(self, state):
        """Get probabilities from environment."""
        return self.env.stochastic_action_probs(state)
```

## Additional Considerations

### 1. Stochastic Step Function

The environment should also provide a step function for stochastic actions:

```python
def stochastic_step(self, state: State, outcome: int, key: PRNGKey) -> State:
    """Apply stochastic outcome to state.

    Args:
        state: Current state (must be at chance node)
        outcome: Index in [0, num_stochastic_actions)
        key: Random key (usually unused since outcome is deterministic)

    Returns:
        New state after applying the stochastic outcome
    """
```

### 2. State Metadata

The existing `state._is_stochastic` flag already indicates whether the current state is a chance node. This should be exposed as a public attribute:

```python
@property
def is_stochastic(self) -> bool:
    """True if this state is a chance node awaiting a random outcome."""
    return self._is_stochastic
```

### 3. Outcome Encoding

Document the outcome encoding for each game:

| Game | Outcome Index | Meaning |
|------|---------------|---------|
| 2048 | `pos * 2 + tile_idx` | Position 0-15, tile_idx 0=2, 1=4 |
| Backgammon | 0-5: doubles, 6-20: non-doubles | Standard dice encoding |
| Pig | 0-5 | Die face minus 1 |

### 4. JAX Compatibility

All functions must be JAX-compatible (no Python control flow in traced paths):
- Use `jnp` operations instead of Python loops where possible
- Use `jax.lax.cond` instead of Python `if`
- Ensure vmappable for batched MCTS

## Benefits

1. **Encapsulation**: Game logic stays in the environment
2. **Correctness**: Environment authors know the correct probabilities
3. **Maintainability**: Changes to state representation don't break external code
4. **Usability**: MCTS implementations become game-agnostic
5. **Testing**: Can unit test probability functions independently

## Migration Path

1. Add `is_stochastic`, `num_stochastic_actions`, and `stochastic_action_probs()` to base `Env` class
2. Implement for existing stochastic games (2048, Backgammon, etc.)
3. Return uniform distribution or raise error for non-stochastic games
4. Update documentation with outcome encoding for each game

## Questions for PGX Team

1. Is the proposed API compatible with existing PGX design patterns?
2. Should `stochastic_action_probs` be a method or a standalone function like `step`?
3. Any concerns about JAX tracing with the proposed interface?
4. Timeline for adding this to pgx stochastic games?

---

*Prepared by TurboZero team for PGX integration*
