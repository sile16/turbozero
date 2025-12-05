import os
# Force CPU-only mode to avoid Metal compatibility issues
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# Metal acceleration disabled due to compatibility issues with pgx
# os.environ['JAX_METAL_PLUGIN'] = '1'
# os.environ['JIT_METAL_DISABLE_COMPILATION_CACHE'] = '1'
# os.environ['METAL_DEVICE_WRAPPER_TYPE'] = 'buffer_loads'

import jax
import jax.numpy as jnp
import pgx.backgammon as bg
import chex
import time
from functools import partial

from core.types import StepMetadata
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.bgcommon import scalar_value_to_probs
import flax.linen as nn

# Print JAX device info
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")
print(f"JAX version: {jax.__version__}")

# Set up the environment globally to avoid reference before assignment
env = bg.Backgammon(simple_doubles=True)

# --- Define a Simple MLP Network ---
class SimpleMLP(nn.Module):
    num_actions: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x, train: bool = False):
        # Flatten input if it's not already flat
        x = x.reshape((x.shape[0], -1)) 
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Policy head
        policy_logits = nn.Dense(features=self.num_actions)(x)
        
        # Value head 
        value = nn.Dense(features=6)(x)
        
        return policy_logits, value


# --- Environment Interface Functions ---
def step_fn_with_key(state: bg.State, action: int, key: chex.PRNGKey):
    """Combined step function for backgammon environment that includes key parameter."""
    def stochastic_branch(operand):
        s, a, _ = operand  # state, action, key (key ignored for stochastic step)
        return env.stochastic_step(s, a)

    def deterministic_branch(operand):
        s, a, k = operand  # state, action, key
        return env.step(s, a, k)

    # Use conditional to route to the appropriate branch
    new_state = jax.lax.cond(
        state._is_stochastic,
        stochastic_branch,
        deterministic_branch,
        (state, action, key)
    )

    # Create standard metadata
    metadata = StepMetadata(
        rewards=new_state.rewards,
        action_mask=new_state.legal_action_mask,
        terminated=new_state.terminated,
        cur_player_id=new_state.current_player,
        step=new_state._step_count
    )

    return new_state, metadata


def state_to_nn_input(state):
    """Extracts observation from state for the neural network."""
    return state.observation


# --- Pip Count Eval Fn (doesn't require neural network parameters) ---
def backgammon_pip_count_eval(state: chex.ArrayTree, params: chex.ArrayTree, key: chex.PRNGKey):
    """Calculates value based on pip count difference. Ignores params/key."""
    board = state._board
    loc_player_0 = jnp.maximum(0, board[1:25])
    loc_player_1 = jnp.maximum(0, -board[1:25])
    points = jnp.arange(1, 25)
    pip_player_0 = jnp.sum(loc_player_0 * points)
    pip_player_1 = jnp.sum(loc_player_1 * (25 - points))
    pip_player_0 += jnp.maximum(0, board[0]) * 25
    pip_player_1 += jnp.maximum(0, -board[25]) * 25
    total_pips = pip_player_0 + pip_player_1 + 1e-6
    value_p0_perspective = (pip_player_1 - pip_player_0) / total_pips
    value = jnp.where(state.current_player == 0, value_p0_perspective, -value_p0_perspective)
    # Uniform policy over legal actions for greedy baseline
    policy_logits = jnp.where(state.legal_action_mask, 0.0, -jnp.inf)
    return policy_logits, scalar_value_to_probs(value)


def run_stochastic_mcts_benchmark(evaluator, env_state, num_runs=5, iterations_per_run=100, params=None):
    """Run benchmark for StochasticMCTS evaluator.
    
    Args:
        evaluator: StochasticMCTS evaluator
        env_state: Initial environment state
        num_runs: Number of benchmark runs
        iterations_per_run: Number of MCTS iterations per run
        params: Neural network parameters (if using neural network)
    """
    print(f"\nBenchmarking StochasticMCTS with {iterations_per_run} iterations per run")
    print(f"Evaluator settings: max_nodes={evaluator.max_nodes}, branching_factor={evaluator.branching_factor}")
    
    # Initialize evaluator state
    eval_state = evaluator.init(template_embedding=env_state)
    
    # Create metadata for root node
    metadata = StepMetadata(
        rewards=env_state.rewards,
        action_mask=env_state.legal_action_mask,
        terminated=env_state.terminated,
        cur_player_id=env_state.current_player,
        step=env_state._step_count
    )
    
    # Use empty dict if params is None
    if params is None:
        params = {}
    
    # Compile the evaluator's iterate function with key
    @jax.jit
    def iterate_fn(key, tree):
        return evaluator.iterate(key, tree, params, step_fn_with_key)
    
    # First run: warm-up to trigger compilation
    print("Performing warm-up run...")
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, iterations_per_run)
    tree = eval_state
    for i in range(iterations_per_run):
        tree = iterate_fn(keys[i], tree)
    
    # Make sure compilation is complete
    jax.block_until_ready(tree)
    print("Warm-up complete")
    
    # Benchmark runs
    total_time = 0
    results = []
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}:")
        
        key = jax.random.PRNGKey(run + 1)
        keys = jax.random.split(key, iterations_per_run)
        tree = eval_state
        
        # Force compilation to complete before timing
        jax.block_until_ready(tree)
        
        # Start timing
        start_time = time.time()
        
        # Run the iterations
        for i in range(iterations_per_run):
            tree = iterate_fn(keys[i], tree)
            
            # Real-time output every 10 iterations
            if (i + 1) % 10 == 0:
                # Force computation to complete
                jax.block_until_ready(tree)
                elapsed = time.time() - start_time
                iterations_per_sec = (i + 1) / elapsed
                print(f"  Iteration {i+1}/{iterations_per_run}: {iterations_per_sec:.2f} iterations/second")
        
        # Force computation to complete for final measurement
        jax.block_until_ready(tree)
        elapsed_time = time.time() - start_time
        iterations_per_second = iterations_per_run / elapsed_time
        
        print(f"  Run {run+1} completed in {elapsed_time:.4f} seconds")
        print(f"  Performance: {iterations_per_second:.2f} iterations per second")
        
        total_time += elapsed_time
        results.append(iterations_per_second)
    
    # Calculate and display final results
    avg_iterations_per_second = sum(results) / len(results)
    min_iterations_per_second = min(results)
    max_iterations_per_second = max(results)
    
    print("\n--- Final Results ---")
    print(f"Average: {avg_iterations_per_second:.2f} iterations per second")
    print(f"Min: {min_iterations_per_second:.2f} iterations per second")
    print(f"Max: {max_iterations_per_second:.2f} iterations per second")
    print(f"Total time: {total_time:.2f} seconds for {num_runs} runs of {iterations_per_run} iterations each")
    
    return avg_iterations_per_second


def benchmark_stochastic_mcts_with_different_nodes(max_nodes_list, iterations_per_run=100):
    """Benchmark StochasticMCTS with different max_nodes values.
    
    Args:
        max_nodes_list: List of max_nodes values to test
        iterations_per_run: Number of iterations per benchmark run
    """
    results = []
    
    # Initial environment state
    key = jax.random.PRNGKey(0)
    init_state = env.init(key)
    
    print("\n===== StochasticMCTS Performance Benchmark =====")
    print(f"Backgammon environment with simple_doubles={env.simple_doubles}")
    print(f"Running with {iterations_per_run} iterations per run")
    
    for max_nodes in max_nodes_list:
        print(f"\n----- Testing with max_nodes={max_nodes} -----")
        
        # Create StochasticMCTS evaluator with pip count eval (no parameters needed)
        stochastic_mcts = StochasticMCTS(
            eval_fn=backgammon_pip_count_eval,  # Use pip count eval which doesn't need parameters
            action_selector=PUCTSelector(),
            stochastic_action_probs=env.stochastic_action_probs,
            branching_factor=env.num_actions,
            max_nodes=max_nodes,
            num_iterations=50,
            discount=-1.0,
            temperature=1.0,
            noise_scale=0.05
        )
        
        # Run benchmark with the initial stochastic state
        perf = run_stochastic_mcts_benchmark(stochastic_mcts, init_state, num_runs=3, iterations_per_run=iterations_per_run)
        
        results.append({
            'max_nodes': max_nodes,
            'performance': perf
        })
    
    # Display comparison summary
    print("\n===== Performance Summary =====")
    print(f"{'Max Nodes':<10} | {'Performance (iterations/second)':<30}")
    print("-" * 45)
    
    for result in results:
        print(f"{result['max_nodes']:<10} | {result['performance']:<30.2f}")


if __name__ == "__main__":
    # Get environment details
    NUM_ACTIONS = env.num_actions
    print(f"Backgammon environment initialized with {NUM_ACTIONS} actions")
    
    # Get stochastic probabilities
    STOCHASTIC_PROBS = env.stochastic_action_probs
    print(f"Stochastic action probabilities: {STOCHASTIC_PROBS}")
    
    # Initialize the neural network (not used in this benchmark, using pip count instead)
    mlp_policy_value_net = SimpleMLP(num_actions=NUM_ACTIONS, hidden_dim=32)
    
    # Example of initializing parameters (if we wanted to use the network)
    # rng = jax.random.PRNGKey(0)
    # sample_input = jnp.zeros((1, 34))  # Backgammon observation shape
    # params = mlp_policy_value_net.init(rng, sample_input, train=False)
    
    # Run benchmark with different max_nodes values using pip count evaluator
    max_nodes_list = [100, 500, 1000, 2000]
    benchmark_stochastic_mcts_with_different_nodes(max_nodes_list, iterations_per_run=100) 
