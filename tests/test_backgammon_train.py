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
from core.training.train import Trainer
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

from core.bgcommon import bg_step_fn, bg_pip_count_eval, BGRandomEvaluator, bg_hit2_eval, ResNetTurboZero

# --- Environment Setup ---
env = bg.Backgammon(simple_doubles=True, short_game=True)
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
        value = nn.Dense(features=6)(x)
        
        return policy_logits, value

# --- Neural Network ---
mlp_policy_value_net = SimpleMLP(num_actions=NUM_ACTIONS, hidden_dim=32)



# Instantiate the Random Evaluator
random_evaluator = BGRandomEvaluator()

def get_temperature(epoch: int, num_epochs: int, start_temp: float = 1.0, end_temp: float = 0.0) -> float:
    """Calculate temperature based on current epoch.
    
    Args:
        epoch: Current epoch number
        num_epochs: Total number of epochs
        start_temp: Starting temperature (default: 1.0)
        end_temp: Final temperature (default: 0.0)
        
    Returns:
        float: Temperature value for the current epoch
    """
    # Linear decay from start_temp to end_temp
    progress = epoch / (num_epochs - 1) if num_epochs > 1 else 1.0
    return start_temp - (start_temp - end_temp) * progress

# --- Evaluators ---
# Training evaluator: StochasticMCTS using NN
evaluator = StochasticMCTS(   #Explores new moves
    eval_fn=make_nn_eval_fn(mlp_policy_value_net, state_to_nn_input),
    stochastic_action_probs=STOCHASTIC_PROBS,
    num_iterations=20,  
    max_nodes=300,      
    branching_factor=NUM_ACTIONS,
    action_selector=PUCTSelector(),
    temperature=1.0,  # Initial temperature, will be updated during training
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

hit2_mcts_test = StochasticMCTS(  # optimizes for moves
    eval_fn=bg_hit2_eval, # Use pip count eval fn
    stochastic_action_probs=STOCHASTIC_PROBS,
    num_iterations=20, # Give it slightly more iterations maybe
    max_nodes=100,
    branching_factor=NUM_ACTIONS,
    action_selector=PUCTSelector(),
    temperature=0.0 # Deterministic action selection for testing
)

# --- Replay Memory ---
replay_memory = EpisodeReplayBuffer(capacity=500)




# --- Main Execution ---
@pytest.mark.slow
def test_backgammon_training_loop():
    # --- Trainer ---
    # Use minimal test configuration to keep test under 2 minutes
    trainer = StochasticTrainer(
        batch_size=2,      # Minimal batch size
        train_batch_size=2,
        warmup_steps=0,
        collection_steps_per_epoch=20,  # Reduced from 300 for faster test
        train_steps_per_epoch=2,
        nn=mlp_policy_value_net,
        loss_fn=partial(az_default_loss_fn, l2_reg_lambda=0.0),
        optimizer=optax.adam(1e-4),
        # Use the stochastic evaluator for training
        evaluator=evaluator,
        memory_buffer=replay_memory,
        max_episode_steps=100,  # Reduced from 1000 for faster test
        env_step_fn=partial(bg_step_fn, env),
        env_init_fn=init_fn,
        state_to_nn_input_fn=state_to_nn_input,
        testers=[
            # Use our custom BackgammonTwoPlayerBaseline
            TwoPlayerBaseline(
                num_episodes=5,  # Reduced from 50 for faster test
                baseline_evaluator=pip_count_mcts_evaluator_test,
                name='pip_count_baseline'
            ),
            # Add another tester using the Hit2 evaluator
            TwoPlayerBaseline(
                num_episodes=5,  # Reduced from 50 for faster test
                baseline_evaluator=hit2_mcts_test,
                name='hit2_baseline'
            )
        ],
        evaluator_test=evaluator_test
    )
    """Runs a minimal training loop for Backgammon with StochasticMCTS."""
    print("Starting minimal Backgammon training test with StochasticMCTS...")

    print("Using minimal configuration with StochasticMCTS and Pip Count Test Evaluator")

    num_epochs = 2  # Reduced from 6 for faster test
    output = None
    # Update temperature for each epoch
    for epoch in range(num_epochs):
        current_temp = get_temperature(epoch, num_epochs)
        evaluator.temperature = current_temp
        print(f"Epoch {epoch + 1}/{num_epochs}, Temperature: {current_temp:.2f}")

        # Run one epoch of training
        output = trainer.train_loop(seed=42, num_epochs=epoch, eval_every=1, initial_state=output)
        assert output is not None # Basic check to ensure it ran
    
    print("Training loop completed successfully.")
    print("Test finished.")


def test_train_step_test_large_nn():
    resnet_model = ResNetTurboZero(
        num_actions=env.num_actions,  # i.e. micro_steps*(micro_src + micro_die)
        hidden_dim=256,
        num_blocks=10
    )

    trainer2 = StochasticTrainer(
        batch_size=2,      # Minimal batch size
        train_batch_size=2,
        warmup_steps=0,
        collection_steps_per_epoch=10,  # needs to be higher than average game length, or better, 10x
        train_steps_per_epoch=2,       
        nn=resnet_model,
        loss_fn=partial(az_default_loss_fn, l2_reg_lambda=0.0),
        optimizer=optax.adam(1e-4),
        testers=[],
        # Use the stochastic evaluator for training
        evaluator=BGRandomEvaluator(), 
        memory_buffer=replay_memory,
        max_episode_steps=1000,  
        env_step_fn=partial(bg_step_fn, env),
        env_init_fn=init_fn,
        state_to_nn_input_fn=state_to_nn_input
    )
    

    trainer2.train_loop(seed=42, num_epochs=1)

    


