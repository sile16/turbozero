import jax
import jax.numpy as jnp
import pgx
import pgx.backgammon as bg
import chex
import optax
from functools import partial
import flax.linen as nn
from typing import Tuple
import time
import pytest
import jax.tree_util

from core.types import StepMetadata
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
# Import regular MCTS for the test evaluator
from core.evaluators.mcts.mcts import MCTS
from core.memory.replay_memory import EpisodeReplayBuffer
from core.training.loss_fns import az_default_loss_fn
from core.training.stochastic_train import StochasticTrainer
from core.testing.two_player_baseline import TwoPlayerBaseline
from core.common import two_player_game_step # Need this for the game loop
from core.common import TwoPlayerGameState, GameFrame # Import the dataclasses
from core.evaluators.evaluator import Evaluator, EvalOutput # Make sure EvalOutput is imported
from core.types import EnvInitFn, EnvStepFn # Type hints
from typing import Dict, Tuple # Type hints

from functools import partial
from core.testing.utils import render_pgx_2p
render_fn = partial(render_pgx_2p, p1_label='Black', p2_label='White', duration=900)

# --- Environment Setup ---
env = bg.Backgammon(simple_doubles=True)
NUM_ACTIONS = env.num_actions
print(f"NUM_ACTIONS: {NUM_ACTIONS}")

# --- Get test observation ---
_key = jax.random.PRNGKey(0)
_init_state = env.init(_key)
OBSERVATION_SHAPE = _init_state.observation.shape
print(f"Detected Observation Shape: {OBSERVATION_SHAPE}")
STOCHASTIC_PROBS = env.stochastic_action_probs
print(f"STOCHASTIC_PROBS: {STOCHASTIC_PROBS}")

# --- Environment Interface Functions ---
def step_fn(state: bg.State, action: int, key: chex.PRNGKey) -> Tuple[bg.State, StepMetadata]:
    """Combined step function for backgammon environment that handles both deterministic and stochastic actions."""
    # print(f"[DEBUG-BG_STEP-{time.time()}] Called with state (stochastic={state.is_stochastic}), action={action}") # Optional debug

    # Handle stochastic vs deterministic branches
    def stochastic_branch(operand):
        s, a, _ = operand # state, action, key (key ignored for stochastic step)
        # Use env instance captured by closure (assuming env is accessible in this scope)
        return env.stochastic_step(s, a)

    def deterministic_branch(operand):
        s, a, k = operand # state, action, key
        # Use env instance captured by closure
        return env.step(s, a, k)

    # Use conditional to route to the appropriate branch
    # The key is only needed for the deterministic branch
    new_state = jax.lax.cond(
        state.is_stochastic,
        stochastic_branch,
        deterministic_branch,
        (state, action, key) # Pass all required operands
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

def init_fn(key):
    """Initializes a new environment state."""
    state = env.init(key)
    # No need to force non-stochastic, let the environment handle it
    return state, StepMetadata(
        rewards=state.rewards,
        action_mask=state.legal_action_mask,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state._step_count
    )

def state_to_nn_input(state):
    """Extracts observation from state for the neural network."""
    return state.observation

# --- Define a Simple MLP Network ---
class SimpleMLP(nn.Module):
    num_actions: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x, train: bool):
        # Flatten input if it's not already flat
        x = x.reshape((x.shape[0], -1)) 
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Policy head
        policy_logits = nn.Dense(features=self.num_actions)(x)
        
        # Value head 
        value = nn.Dense(features=1)(x)
        value = jnp.tanh(value)
        
        return policy_logits, jnp.squeeze(value, axis=-1)

# --- Pip Count Eval Fn (for test evaluator) ---
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
    return policy_logits, jnp.array(value)


# --- Neural Network ---
mlp_policy_value_net = SimpleMLP(num_actions=NUM_ACTIONS, hidden_dim=32)

# --- Random Evaluator ---
class RandomEvaluator(Evaluator):
    """An evaluator that selects actions randomly from legal moves."""

    def __init__(self, discount: float = -1.0):
        """Initializes the RandomEvaluator."""
        super().__init__(discount=discount)

    def evaluate(self, key: chex.PRNGKey, eval_state: chex.ArrayTree, env_state: chex.ArrayTree,
                 root_metadata: StepMetadata, params: chex.ArrayTree, env_step_fn: EnvStepFn, **kwargs) -> EvalOutput:
        """Chooses a random legal action."""
        action_mask = root_metadata.action_mask
        num_actions = action_mask.shape[-1]
        
        # Create uniform policy over legal actions
        legal_actions_count = jnp.sum(action_mask)
        uniform_prob = jnp.where(legal_actions_count > 0, 1.0 / legal_actions_count, 0.0)
        policy_weights = jnp.where(action_mask, uniform_prob, 0.0)
        
        # Sample a random action from the legal ones
        action = jax.random.choice(key, jnp.arange(num_actions), p=policy_weights)
        
        return EvalOutput(
            eval_state=eval_state, 
            action=action,
            policy_weights=policy_weights,
            value=jnp.array(0.0) 
        )

    def get_value(self, state: chex.ArrayTree) -> chex.Array:
        """Returns a zero value estimate."""
        return jnp.array(0.0)

    def init(self, template_embedding: chex.ArrayTree, *args, **kwargs) -> chex.ArrayTree:
        """Initializes the dummy state (can be empty)."""
        return jnp.array(0) # Return a minimal placeholder state

    def reset(self, state: chex.ArrayTree) -> chex.ArrayTree:
        """Resets the dummy state."""
        return state # Stateless, nothing to reset

    def step(self, state: chex.ArrayTree, action: int) -> chex.ArrayTree:
        """Updates the dummy state (no change needed)."""
        return state # Stateless, nothing to update based on action

# Instantiate the Random Evaluator
random_evaluator = RandomEvaluator()

# --- Evaluators ---
# Training evaluator: StochasticMCTS using NN
evaluator = StochasticMCTS(   #Explores new moves
    eval_fn=make_nn_eval_fn(mlp_policy_value_net, state_to_nn_input),
    stochastic_action_probs=STOCHASTIC_PROBS,
    num_iterations=200,  
    max_nodes=300,      
    branching_factor=NUM_ACTIONS,
    action_selector=PUCTSelector(),
    temperature=1.0,
)

evaluator_test = StochasticMCTS(   #Use optimized moves, temperature=0.0
    eval_fn=make_nn_eval_fn(mlp_policy_value_net, state_to_nn_input),
    stochastic_action_probs=STOCHASTIC_PROBS,
    num_iterations=200,  # Very few iterations
    max_nodes=300,      # Very small tree
    branching_factor=NUM_ACTIONS,
    action_selector=PUCTSelector(),
    temperature=0.0,
)

# Test evaluator: Regular MCTS using pip count
pip_count_mcts_evaluator_test = StochasticMCTS(  # optimizes for moves
    eval_fn=backgammon_pip_count_eval, # Use pip count eval fn
    stochastic_action_probs=STOCHASTIC_PROBS,
    num_iterations=30, # Give it slightly more iterations maybe
    max_nodes=100,
    branching_factor=NUM_ACTIONS,
    action_selector=PUCTSelector(),
    temperature=0.0 # Deterministic action selection for testing
)

# --- Replay Memory ---
replay_memory = EpisodeReplayBuffer(capacity=100)

# --- Custom BackgammonTwoPlayerBaseline ---
class BackgammonTwoPlayerBaseline(TwoPlayerBaseline):
    """TwoPlayerBaseline overridden to handle backgammon's 1D observation shape and rendering."""

    @partial(jax.pmap, axis_name='d', static_broadcasted_argnums=(0, 1, 2, 3, 4))
    def test(self, max_steps: int, env_step_fn: EnvStepFn, env_init_fn: EnvInitFn, evaluator: Evaluator,
             keys: chex.PRNGKey, state: chex.ArrayTree, params: chex.ArrayTree) -> Tuple[chex.ArrayTree, Dict, chex.ArrayTree, chex.Array]:
        """Test the agent against the baseline, handling 1D observations and collecting frames."""

        def game_fn(key):
            """Plays a single two-player game and collects frames."""
            # init envs
            key1, key2 = jax.random.split(key)
            env1_state, metadata1 = env_init_fn(key1)
            env2_state, metadata2 = env_init_fn(key2)

            # init evaluator states
            key1, key2 = jax.random.split(jax.random.fold_in(key, 1))
            eval_state1 = evaluator.init(template_embedding=env1_state)
            eval_state2 = self.baseline_evaluator.init(template_embedding=env2_state)

            # Initial game state using the dataclass
            init_val = TwoPlayerGameState(
                key=key,
                env_state=env1_state, 
                env_state_metadata=metadata1, 
                p1_eval_state=eval_state1,
                p2_eval_state=eval_state2,
                p1_value_estimate=jnp.array(0.0),
                p2_value_estimate=jnp.array(0.0),
                outcomes=jnp.zeros(2),
                completed=False 
            )
            
            # Capture initial frame
            initial_game_frame = GameFrame(
                env_state = init_val.env_state,
                p1_value_estimate = init_val.p1_value_estimate,
                p2_value_estimate = init_val.p2_value_estimate,
                completed = init_val.completed,
                outcomes = init_val.outcomes
            )

            # Modified step_step to return GameFrame using state *before* step for rendering
            def step_step(carry: TwoPlayerGameState, _) -> Tuple[TwoPlayerGameState, GameFrame]:
                needs_step = jnp.logical_and(~carry.completed, carry.env_state_metadata.step < max_steps)
                
                # Store state before step for frame capture
                state_before_step = carry 

                # Call the common game step logic
                next_state_if_step = two_player_game_step(
                    state=carry, 
                    p1_evaluator=evaluator, 
                    p2_evaluator=self.baseline_evaluator,
                    params=params, 
                    env_step_fn=env_step_fn,
                    env_init_fn=env_init_fn,
                    use_p1=True, # Simplified assumption for step call structure
                    max_steps=max_steps
                )
                
                next_carry = jax.tree.map(
                    lambda x, y: jnp.where(needs_step, x, y), 
                    next_state_if_step, 
                    carry
                )
                
                truncated = jnp.logical_and(needs_step, next_carry.env_state_metadata.step >= max_steps)
                next_carry = next_carry.replace(completed=jnp.logical_or(next_carry.completed, truncated))
                
                # Create frame using the state *before* the step was taken (state_before_step.env_state)
                # but use values/outcomes from *after* the step (next_carry)
                frame = GameFrame(
                    env_state=state_before_step.env_state, # Use state before step for rendering compatibility
                    p1_value_estimate=next_carry.p1_value_estimate,
                    p2_value_estimate=next_carry.p2_value_estimate,
                    completed=next_carry.completed,
                    outcomes=next_carry.outcomes
                )
                
                return next_carry, frame

            # Run scan, collecting frames
            final_state, collected_frames = jax.lax.scan(step_step, init_val, None, length=max_steps)

            # Combine initial frame with collected frames
            all_frames = jax.tree.map(
                lambda init, rest: jnp.concatenate([jnp.expand_dims(init, 0), rest]),
                initial_game_frame, collected_frames
            )

            p1_won = final_state.outcomes[0] > 0
            p2_won = final_state.outcomes[1] > 0
            draw = jnp.logical_and(final_state.outcomes[0] == 0, final_state.outcomes[1] == 0)

            results = {
                'p1_won': p1_won,
                'p2_won': p2_won,
                'draw': draw,
                'steps': final_state.env_state_metadata.step, 
                'outcome': jnp.select([p1_won, p2_won, draw], [1.0, -1.0, 0.0], default=0.0)
            }
            
            p_ids = jnp.array([0, 1]) # Assuming player 0 = agent, player 1 = baseline

            # Return results and the actual frames
            return results, all_frames, p_ids

        # Run vmap over game_fn
        results, frames, p_ids = jax.vmap(game_fn)(keys)

        # Calculate metrics
        avg = results['outcome'].mean()
        metrics = {
            f"{self.name}_avg_outcome": avg,
            f"{self.name}_win_rate": jnp.mean(results['p1_won']),
            f"{self.name}_loss_rate": jnp.mean(results['p2_won']),
            f"{self.name}_draw_rate": jnp.mean(results['draw']),
            f"{self.name}_avg_steps": jnp.mean(results['steps'])
        }
        
        # Select frames and p_ids from the first episode for rendering
        first_ep_frames = jax.tree.map(lambda x: x[0], frames)
        first_ep_p_ids = p_ids[0]

        # Return state, metrics, and *actual* frames
        return state, metrics, first_ep_frames, first_ep_p_ids


# --- Trainer ---
trainer = StochasticTrainer(
    batch_size=128,      # Minimal batch size
    train_batch_size=50,
    warmup_steps=0,
    collection_steps_per_epoch=50,  # Just 2 collection step
    train_steps_per_epoch=50,       # Just 2 training step
    nn=mlp_policy_value_net,
    loss_fn=partial(az_default_loss_fn, l2_reg_lambda=0.0),
    optimizer=optax.adam(1e-4),
    # Use the stochastic evaluator for training
    evaluator=evaluator, 
    memory_buffer=replay_memory,
    max_episode_steps=60,  # Super short episodes
    env_step_fn=step_fn,
    env_init_fn=init_fn,
    state_to_nn_input_fn=state_to_nn_input,
    testers=[
        # Use our custom BackgammonTwoPlayerBaseline
        BackgammonTwoPlayerBaseline(
            num_episodes=2,
            baseline_evaluator=pip_count_mcts_evaluator_test,
            #render_fn=render_fn,
            #render_dir='training_eval/pip_count_baseline',
            name='pip_count_baseline'
        ),
        # Add another tester using the RandomEvaluator
        BackgammonTwoPlayerBaseline(
            num_episodes=2, 
            baseline_evaluator=random_evaluator, # Use the random evaluator here
            # Optionally add rendering for this baseline too
            #render_fn=render_fn, 
            #render_dir='training_eval/random_baseline',
            name='random_baseline'
        )
    ],
    # Use the pip count MCTS evaluator for testing
    evaluator_test=evaluator_test, 
    data_transform_fns=[],  # No data transforms as requested
    wandb_project_name=None
)

# --- Main Execution ---
def test_backgammon_training_loop():
    """Runs a minimal training loop for Backgammon with StochasticMCTS."""
    print("Starting minimal Backgammon training test with StochasticMCTS...")
    try:
        print("Using minimal configuration with StochasticMCTS and Pip Count Test Evaluator")
        # Pass the trainer instance created in the global scope
        output = trainer.train_loop(seed=42, num_epochs=6, eval_every=3)
        print("Training loop completed successfully.")
        assert output is not None # Basic check to ensure it ran
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Training loop failed with exception: {e}")

    print("Test finished.") 