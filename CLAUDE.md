# TurboZero Development Notes

## Gumbel AlphaZero (UnifiedMCTS)

UnifiedMCTS implements the full Gumbel AlphaZero algorithm from ["Policy improvement by planning with Gumbel" (ICLR 2022)](https://openreview.net/forum?id=bERaNdoegnO).

### Key Features

1. **Gumbel-Top-k Sampling**: Replaces Dirichlet noise for exploration
   - Samples k actions without replacement using g(a) + log π(a)
   - Provides stochastic exploration while guaranteeing policy improvement

2. **Sequential Halving**: Progressive action elimination
   - Divides simulations into log₂(k) phases
   - Each phase visits active actions equally
   - Eliminates bottom half based on Gumbel scores at phase boundaries

3. **σ(q̂) Scaling**: Normalizes Q-values for action selection
   - `σ(q̂(a)) = c_scale * q̂(a) / (c_visit + max_b N(b))`
   - Default: c_visit=50, c_scale=1.0 (from paper)

4. **Improved Policy Target**: For training
   - `π'(a) ∝ exp(logits(a) + σ(q̂(a)))`
   - Completed Q-values: uses root_value for unvisited actions
   - Preserves information about unexplored actions even with small k

### Configuration

```python
mcts = UnifiedMCTS(
    eval_fn=nn_eval_fn,
    action_selector=PUCTSelector(),
    policy_size=env.num_actions,
    max_nodes=200,
    num_iterations=100,
    gumbel_k=16,        # Actions to sample (paper recommends 16-32)
    c_visit=50.0,       # σ scaling parameter
    c_scale=1.0,        # σ scaling parameter
    decision_step_fn=step_fn,
    temperature=1.0,    # For action sampling (0=greedy)
)
```

### Benefits over Standard AlphaZero

- Works with very few simulations (2-16)
- Theoretical policy improvement guarantee
- No need for Dirichlet noise tuning
- Better exploration/exploitation balance

## JIT Compilation Times (RTX 3090)

Typical first-epoch compilation times for different configurations:

| Game | MCTS Iters | Batch Size | Network | Compile Time |
|------|------------|------------|---------|--------------|
| TicTacToe | 25 | 64 | 2 blocks, 32ch | ~30s |
| TicTacToe | 100 | 128 | 3 blocks, 64ch | ~60s |
| 2048 (StochasticMCTS) | 50 | 128 | 6 blocks, 128ch | ~10-15 min |

**Note**: If compilation takes longer than a couple minutes for simple games, something is likely wrong (output buffering, stuck process, etc.).

## Training Benchmarks

### TicTacToe
- **Reference**: [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) uses 1000 iterations, 100 episodes, 25 MCTS sims
- **Target**: NN (without MCTS) should beat random ~95%+ after training
- **MCTS alone**: With 4000 simulations achieves perfect play (0 losses)

### 2048
- **Target**: >100k score
- **Known issues**: Train/test temperature mismatch can cause eval scores to decline while training metrics improve

## Common Issues

### Output Buffering
Python buffers stdout by default. Use `PYTHONUNBUFFERED=1` to see output immediately:
```bash
PYTHONUNBUFFERED=1 poetry run python scripts/train_xxx.py
```

### Train/Test Mismatch
If eval scores decline while training metrics improve:
- Check temperature: train temp=1.0 vs test temp=0.0 can cause issues
- Check MCTS iterations: test should use similar iterations as training
- Solution: Use moderate training temp (0.5) or temperature annealing

Read AGENTS.md
