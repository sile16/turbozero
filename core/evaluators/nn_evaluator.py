"""Simple Neural Network Evaluator and Random Evaluator.

NNEvaluator is useful for:
1. Testing if the NN has learned the game (without MCTS assistance)
2. Fast evaluation during testing
3. Games simple enough that NN can solve them directly (e.g., TicTacToe)

RandomEvaluator is useful for:
1. Baseline comparisons
2. Testing against random play
"""

from typing import Optional
import chex
import jax
import jax.numpy as jnp
from chex import dataclass

from core.evaluators.evaluator import Evaluator, EvalOutput
from core.types import EvalFn, StepMetadata


@dataclass(frozen=True)
class RandomEvalState:
    """Minimal state for random evaluator."""
    dummy: int  # Just a placeholder


class RandomEvaluator(Evaluator):
    """Evaluator that selects random legal actions.

    Useful as a baseline for testing trained agents.
    """

    def __init__(self, policy_size: int):
        """
        Args:
            policy_size: Number of possible actions
        """
        super().__init__()
        self.policy_size = policy_size

    def init(self, template_embedding: chex.ArrayTree = None, *args, **kwargs) -> RandomEvalState:
        """Initialize evaluator state."""
        return RandomEvalState(dummy=0)

    def reset(self, state: RandomEvalState) -> RandomEvalState:
        """Reset evaluator state."""
        return RandomEvalState(dummy=0)

    def evaluate(self,
                 key: chex.PRNGKey,
                 eval_state: RandomEvalState,
                 env_state: chex.ArrayTree,
                 root_metadata: StepMetadata,
                 params: chex.ArrayTree,
                 **kwargs) -> EvalOutput:
        """Select a random legal action.

        Args:
            key: Random key
            eval_state: Current evaluator state
            env_state: Environment state (unused)
            root_metadata: Contains action_mask for legal moves
            params: Parameters (unused)

        Returns:
            EvalOutput with random action and uniform policy
        """
        action_mask = root_metadata.action_mask

        # Uniform distribution over legal actions
        num_legal = jnp.sum(action_mask)
        policy = jnp.where(action_mask, 1.0 / jnp.maximum(num_legal, 1), 0.0)

        # Sample random legal action
        action = jax.random.choice(key, self.policy_size, p=policy)

        return EvalOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=policy
        )

    def get_value(self, state: RandomEvalState) -> chex.Array:
        """Return 0 value estimate (random has no value function)."""
        return jnp.array(0.0)

    def get_config(self) -> dict:
        """Return evaluator configuration."""
        return {
            "type": "RandomEvaluator",
            "policy_size": self.policy_size,
        }


@dataclass(frozen=True)
class NNEvalState:
    """Minimal state for NN evaluator - just tracks the value estimate."""
    value: float


class NNEvaluator(Evaluator):
    """Evaluator that uses neural network directly without MCTS.

    Selects actions based on NN policy output (with optional temperature).
    Useful for testing whether the NN has learned to play well independently.
    """

    def __init__(self,
                 eval_fn: EvalFn,
                 policy_size: int,
                 temperature: float = 0.0,
                 tiebreak_noise: float = 1e-8):
        """
        Args:
            eval_fn: Neural network evaluation function (state, params, key) -> (policy_logits, value)
            policy_size: Number of possible actions
            temperature: Temperature for action selection (0 = greedy, >0 = sample)
            tiebreak_noise: Small noise for breaking ties in greedy selection
        """
        super().__init__()
        self.eval_fn = eval_fn
        self.policy_size = policy_size
        self.temperature = temperature
        self.tiebreak_noise = tiebreak_noise

    def init(self, template_embedding: chex.ArrayTree = None, *args, **kwargs) -> NNEvalState:
        """Initialize evaluator state."""
        return NNEvalState(value=0.0)

    def reset(self, state: NNEvalState) -> NNEvalState:
        """Reset evaluator state."""
        return NNEvalState(value=0.0)

    def evaluate(self,
                 key: chex.PRNGKey,
                 eval_state: NNEvalState,
                 env_state: chex.ArrayTree,
                 root_metadata: StepMetadata,
                 params: chex.ArrayTree,
                 **kwargs) -> EvalOutput:
        """Evaluate state using neural network directly.

        Args:
            key: Random key
            eval_state: Current evaluator state
            env_state: Environment state (used as NN input)
            root_metadata: Contains action_mask for legal moves
            params: Neural network parameters

        Returns:
            EvalOutput with action, policy weights, and updated state
        """
        # Get NN prediction
        policy_logits, value = self.eval_fn(env_state, params, key)

        # Apply action mask
        action_mask = root_metadata.action_mask
        masked_logits = jnp.where(action_mask, policy_logits, -1e9)

        # Get policy probabilities
        policy = jax.nn.softmax(masked_logits)

        # Select action based on temperature
        if self.temperature == 0:
            # Greedy selection with tiebreaking
            noise = jax.random.uniform(key, shape=policy.shape, maxval=self.tiebreak_noise)
            action = jnp.argmax(policy + noise)
        else:
            # Sample with temperature
            tempered_logits = masked_logits / self.temperature
            tempered_policy = jax.nn.softmax(tempered_logits)
            action = jax.random.choice(key, self.policy_size, p=tempered_policy)

        # Update state with value estimate
        new_state = NNEvalState(value=value)

        return EvalOutput(
            eval_state=new_state,
            action=action,
            policy_weights=policy
        )

    def get_value(self, state: NNEvalState) -> chex.Array:
        """Get value estimate from state."""
        return state.value

    def get_config(self) -> dict:
        """Return evaluator configuration."""
        return {
            "type": "NNEvaluator",
            "policy_size": self.policy_size,
            "temperature": self.temperature,
        }
