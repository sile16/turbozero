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
import traceback

from core.types import StepMetadata
from core.evaluators.evaluation_fns import make_nn_eval_fn
from core.evaluators.mcts.action_selection import PUCTSelector
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
# Import regular MCTS for the test evaluator
from core.evaluators.mcts.mcts import MCTS
from core.memory.replay_memory import EpisodeReplayBuffer
from core.training.loss_fns import az_default_loss_fn
from core.training.stochastic_train import StochasticTrainer
from core.training.train import Trainer
from core.testing.two_player_baseline import TwoPlayerBaseline
from core.common import two_player_game_step # Need this for the game loop
from core.common import TwoPlayerGameState, GameFrame # Import the dataclasses
from core.evaluators.evaluator import Evaluator, EvalOutput # Make sure EvalOutput is imported
from core.types import EnvInitFn, EnvStepFn # Type hints
from typing import Dict, Tuple # Type hints

from functools import partial
from core.testing.utils import render_pgx_2p
render_fn = partial(render_pgx_2p, p1_label='Black', p2_label='White', duration=900)

from bg.bgcommon import bg_step_fn, bg_pip_count_eval, BGRandomEvaluator

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

# --- Neural Network ---
mlp_policy_value_net = SimpleMLP(num_actions=NUM_ACTIONS, hidden_dim=32)



# Instantiate the Random Evaluator
random_evaluator = BGRandomEvaluator()

# --- Evaluators ---
# Training evaluator: StochasticMCTS using NN
evaluator = StochasticMCTS(   #Explores new moves
    eval_fn=make_nn_eval_fn(mlp_policy_value_net, state_to_nn_input),
    stochastic_action_probs=STOCHASTIC_PROBS,
    num_iterations=20,  
    max_nodes=300,      
    branching_factor=NUM_ACTIONS,
    action_selector=PUCTSelector(),
    temperature=1.0,
)

evaluator_test = StochasticMCTS(   #Use optimized moves, temperature=0.0
    eval_fn=make_nn_eval_fn(mlp_policy_value_net, state_to_nn_input),
    stochastic_action_probs=STOCHASTIC_PROBS,
    num_iterations=20,  # Very few iterations
    max_nodes=300,      # Very small tree
    branching_factor=NUM_ACTIONS,
    action_selector=PUCTSelector(),
    temperature=0.0,
)

# Test evaluator: Regular MCTS using pip count
pip_count_mcts_evaluator_test = StochasticMCTS(  # optimizes for moves
    eval_fn=bg_pip_count_eval, # Use pip count eval fn
    stochastic_action_probs=STOCHASTIC_PROBS,
    num_iterations=20, # Give it slightly more iterations maybe
    max_nodes=100,
    branching_factor=NUM_ACTIONS,
    action_selector=PUCTSelector(),
    temperature=0.0 # Deterministic action selection for testing
)

# --- Replay Memory ---
replay_memory = EpisodeReplayBuffer(capacity=100)


# --- Trainer ---
trainer = StochasticTrainer(
    batch_size=8,      # Minimal batch size
    train_batch_size=8,
    warmup_steps=0,
    collection_steps_per_epoch=300,  # Just 2 collection step
    train_steps_per_epoch=10,       # Just 2 training step
    nn=mlp_policy_value_net,
    loss_fn=partial(az_default_loss_fn, l2_reg_lambda=0.0),
    optimizer=optax.adam(1e-4),
    # Use the stochastic evaluator for training
    evaluator=evaluator, 
    memory_buffer=replay_memory,
    max_episode_steps=500,  
    env_step_fn=partial(bg_step_fn, env),
    env_init_fn=init_fn,
    state_to_nn_input_fn=state_to_nn_input,
    testers=[
        # Use our custom BackgammonTwoPlayerBaseline
        TwoPlayerBaseline(
            num_episodes=2,
            baseline_evaluator=pip_count_mcts_evaluator_test,
            #render_fn=render_fn,
            #render_dir='training_eval/pip_count_baseline',
            name='pip_count_baseline'
        ),
        # Add another tester using the RandomEvaluator
        TwoPlayerBaseline(
            num_episodes=2, 
            baseline_evaluator=random_evaluator, # Use the random evaluator here
            # Optionally add rendering for this baseline too
            #render_fn=render_fn, 
            #render_dir='training_eval/random_baseline',
            name='random_baseline'
        )
    ],
    
    evaluator_test=evaluator_test, 
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
        
        traceback.print_exc()
        pytest.fail(f"Training loop failed with exception: {e}")

    print("Test finished.") 