"""UnifiedMCTS - Single MCTS implementation for all games.

Combines:
- Gumbel-Top-k for efficient root action selection at decision nodes
- Stochastic node support for games with chance events (dice, tile spawns)
- Subtree persistence (always on) for tree reuse
- Two-player perspective handling

Supports:
- Deterministic games (TicTacToe, Chess, Go)
- Stochastic games (2048, Backgammon, Pig)
- 1 or 2 player games
- Games where root can be stochastic or deterministic
"""

from typing import Dict, Optional, Tuple, Callable, NamedTuple, Union
import math
import jax
import jax.numpy as jnp
import chex

from core.evaluators.evaluator import Evaluator
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.state import (
    MCTSOutput, TraversalState,
    StochasticMCTSNode, StochasticMCTSTree
)
from core.evaluators.mcts.gumbel import (
    gumbel_top_k,
    sigma_q,
    compute_gumbel_score,
    get_action_to_simulate,
    eliminate_half,
    select_action_after_halving,
    compute_improved_policy,
    sequential_halving_phases,
)
from core.types import EvalFn, StepMetadata
from core.trees.tree import init_tree


def _value_to_scalar(value: chex.Array) -> chex.Array:
    """Ensure value is a scalar.

    For scalar values: returns as-is.
    For array values: returns first element (assumes scalar or at least first element is meaningful).
    """
    if value.ndim > 0:
        return value.reshape(-1)[0]  # Flatten and take first element
    return value


# Type aliases for step functions
DecisionStepFn = Callable[[chex.ArrayTree, int, Optional[chex.PRNGKey]], Tuple[chex.ArrayTree, StepMetadata]]
StochasticStepFn = Callable[[chex.ArrayTree, int, Optional[chex.PRNGKey]], Tuple[chex.ArrayTree, StepMetadata]]

# Temperature can be a constant or a function of epoch
TemperatureFn = Callable[[int], float]  # (epoch) -> temperature
Temperature = Union[float, TemperatureFn]


def linear_temp_schedule(start_temp: float, end_temp: float, total_epochs: int) -> TemperatureFn:
    """Create a linear temperature annealing schedule.

    Args:
        start_temp: Temperature at epoch 0
        end_temp: Temperature at epoch total_epochs
        total_epochs: Number of epochs over which to anneal

    Returns:
        Function (epoch) -> temperature
    """
    def schedule(epoch: int) -> float:
        progress = min(1.0, epoch / max(1, total_epochs))
        return start_temp + progress * (end_temp - start_temp)
    return schedule


def exponential_temp_schedule(start_temp: float, end_temp: float, decay_rate: float) -> TemperatureFn:
    """Create an exponential temperature annealing schedule.

    Args:
        start_temp: Temperature at epoch 0
        end_temp: Minimum temperature (asymptote)
        decay_rate: Decay rate per epoch (e.g., 0.95 for 5% decay each epoch)

    Returns:
        Function (epoch) -> temperature
    """
    def schedule(epoch: int) -> float:
        return max(end_temp, start_temp * math.pow(decay_rate, epoch))
    return schedule


def step_temp_schedule(temps: list, epoch_boundaries: list) -> TemperatureFn:
    """Create a step temperature schedule.

    Args:
        temps: List of temperatures for each phase
        epoch_boundaries: List of epoch boundaries (e.g., [10, 50, 100] means:
            epochs 0-9 use temps[0], 10-49 use temps[1], 50-99 use temps[2], 100+ use temps[3])

    Returns:
        Function (epoch) -> temperature
    """
    def schedule(epoch: int) -> float:
        for i, boundary in enumerate(epoch_boundaries):
            if epoch < boundary:
                return temps[i]
        return temps[-1]
    return schedule


class BackpropState(NamedTuple):
    """State for backpropagation loop."""
    node_idx: int
    value: float
    tree: StochasticMCTSTree


class ExpandResult(NamedTuple):
    """Result of node expansion."""
    tree: StochasticMCTSTree
    child_idx: int
    value: float


class UnifiedMCTS(Evaluator):
    """Unified MCTS implementation for all game types.

    Key features:
    - Always uses Gumbel-Top-k at decision node roots
    - Always persists tree between evaluate() calls
    - Handles stochastic transitions (chance nodes)
    - Supports 1-2 player games
    - JAX-compatible (no Python control flow in compiled paths)

    For deterministic games, pass stochastic_action_probs=None.
    For stochastic games, pass the probability distribution over outcomes.
    """

    def __init__(
        self,
        eval_fn: EvalFn,
        action_selector: MCTSActionSelector,
        policy_size: int,
        max_nodes: int,
        num_iterations: int,
        decision_step_fn: DecisionStepFn,
        stochastic_step_fn: Optional[StochasticStepFn] = None,
        stochastic_action_probs: Optional[chex.Array] = None,
        gumbel_k: int = 16,
        temperature: Temperature = 1.0,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
        tiebreak_noise: float = 1e-8,
    ):
        """Initialize UnifiedMCTS.

        Uses Gumbel AlphaZero algorithm from "Policy improvement by planning with Gumbel"
        (ICLR 2022) https://openreview.net/forum?id=bERaNdoegnO

        Key features:
        - Gumbel-Top-k sampling for exploration (replaces Dirichlet noise)
        - Sequential Halving progressively eliminates actions
        - σ(q̂) scaling normalizes Q-values for action selection
        - Improved policy from g(a) + logits(a) + σ(q̂(a))

        Args:
            eval_fn: Neural network evaluation function (state, params, key) -> (policy_logits, value)
            action_selector: Action selector for non-root nodes (e.g., PUCTSelector)
            policy_size: Size of the policy output (number of decision actions)
            max_nodes: Maximum number of nodes in the tree
            num_iterations: Number of MCTS iterations per evaluate() call
            decision_step_fn: Step function for decision nodes (state, action, key) -> (new_state, metadata)
            stochastic_step_fn: Step function for chance nodes. None for deterministic games.
            stochastic_action_probs: Probability distribution over stochastic outcomes. None for deterministic.
            gumbel_k: Number of actions to sample at decision roots (default: 16)
            temperature: Temperature for final action selection. Can be:
                - float: constant temperature (0 = greedy)
                - callable: function (epoch: int) -> float for temperature annealing
            c_visit: Visit count offset for σ(q̂) scaling (default: 50, from paper)
            c_scale: Scale factor for σ(q̂) (default: 1.0, from paper)
            tiebreak_noise: Small noise for breaking ties
        """
        self.eval_fn = eval_fn
        self.action_selector = action_selector
        self.policy_size = policy_size
        self.max_nodes = max_nodes
        self.num_iterations = num_iterations
        self.decision_step_fn = decision_step_fn
        self.c_visit = c_visit
        self.c_scale = c_scale
        self.tiebreak_noise = tiebreak_noise
        self.gumbel_k = min(gumbel_k, policy_size)  # Can't sample more than available

        # Temperature can be a constant or a function of epoch
        if callable(temperature):
            self._temperature_fn = temperature
            self._temperature_const = None
        else:
            self._temperature_fn = None
            self._temperature_const = float(temperature)
        self._current_epoch = 0

        # Stochastic game support
        self.is_stochastic_game = stochastic_action_probs is not None

        if self.is_stochastic_game:
            self.stochastic_action_probs = stochastic_action_probs
            self.stochastic_step_fn = stochastic_step_fn
            self.stochastic_size = len(stochastic_action_probs)
            # Total branching factor = decision actions + stochastic outcomes
            self.branching_factor = policy_size + self.stochastic_size
        else:
            # For deterministic games, provide dummy values for JAX tracing
            # jax.lax.cond traces both branches, so stochastic code paths need valid arrays
            # Use single-element dummy array so operations like jax.random.choice work during tracing
            self.stochastic_action_probs = jnp.array([1.0])  # Dummy single-element for tracing
            self.stochastic_step_fn = lambda s, a, k: (s, StepMetadata(
                rewards=jnp.zeros(2), action_mask=jnp.ones(policy_size, dtype=bool),
                terminated=jnp.array(False), cur_player_id=jnp.array(0), step=jnp.array(0),
                is_stochastic=jnp.array(False)
            ))
            self.stochastic_size = 1  # Match dummy array size
            self.branching_factor = policy_size + 1  # Extra slot for dummy stochastic

    @property
    def temperature(self) -> float:
        """Get current temperature based on epoch."""
        if self._temperature_fn is not None:
            return self._temperature_fn(self._current_epoch)
        return self._temperature_const

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set temperature directly (overrides schedule)."""
        self._temperature_const = value
        self._temperature_fn = None  # Clear any schedule

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for temperature annealing.

        Args:
            epoch: Current epoch number (0-indexed)
        """
        self._current_epoch = epoch

    @property
    def handles_chance_nodes(self) -> bool:
        """Whether this evaluator explicitly handles chance nodes (stochastic states).

        For stochastic games, UnifiedMCTS handles chance nodes internally and
        returns placeholder policy weights for them. The trainer should skip
        policy loss for these samples.
        """
        return self.is_stochastic_game

    def get_config(self) -> Dict:
        """Returns configuration for logging."""
        return {
            "type": "UnifiedMCTS",
            "policy_size": self.policy_size,
            "max_nodes": self.max_nodes,
            "num_iterations": self.num_iterations,
            "gumbel_k": self.gumbel_k,
            "temperature": self.temperature,
            "current_epoch": self._current_epoch,
            "has_temperature_schedule": self._temperature_fn is not None,
            "is_stochastic_game": self.is_stochastic_game,
            "stochastic_size": self.stochastic_size,
            "branching_factor": self.branching_factor,
            "c_visit": self.c_visit,
            "c_scale": self.c_scale,
            "action_selector": self.action_selector.get_config(),
        }

    def init(self, template_embedding: chex.ArrayTree) -> StochasticMCTSTree:
        """Initialize an empty MCTS tree.

        Args:
            template_embedding: Template state for shape inference

        Returns:
            Empty StochasticMCTSTree ready for evaluation
        """
        return init_tree(
            self.max_nodes,
            self.branching_factor,
            self._new_node(
                policy=jnp.zeros((self.branching_factor,)),
                value=0.0,
                embedding=template_embedding,
                terminated=False,
                nn_value_estimate=0.0,
                is_chance_node=False
            )
        )

    def evaluate(
        self,
        key: chex.PRNGKey,
        eval_state: StochasticMCTSTree,
        env_state: chex.ArrayTree,
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
        **kwargs,  # Accept extra args like env_step_fn for compatibility
    ) -> MCTSOutput:
        """Perform MCTS evaluation from the given state.

        Handles both stochastic and decision roots:
        - Decision root: Uses Gumbel-Top-k to select actions to explore
        - Stochastic root: Samples from stochastic_action_probs

        Args:
            key: JAX random key
            eval_state: Current tree state
            env_state: Current game state
            root_metadata: Metadata for root node (action_mask, is_stochastic, etc.)
            params: Neural network parameters

        Returns:
            MCTSOutput with tree state, selected action, and policy weights
        """
        # Check if root is stochastic
        is_root_stochastic = getattr(root_metadata, 'is_stochastic', jnp.array(False))

        # Use jax.lax.cond to handle both cases
        return jax.lax.cond(
            is_root_stochastic,
            lambda: self._evaluate_stochastic_root(key, eval_state, env_state, root_metadata, params),
            lambda: self._evaluate_decision_root(key, eval_state, env_state, root_metadata, params)
        )

    def _evaluate_decision_root(
        self,
        key: chex.PRNGKey,
        eval_state: StochasticMCTSTree,
        env_state: chex.ArrayTree,
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
    ) -> MCTSOutput:
        """Evaluate when root is a decision node (player's turn).

        Implements full Gumbel AlphaZero algorithm:
        1. Gumbel-Top-k samples k actions without replacement
        2. Sequential Halving progressively eliminates actions
        3. σ(q̂) scaling normalizes Q-values for action selection
        4. Improved policy computed from g(a) + logits(a) + σ(q̂(a))
        """
        key, root_key, gumbel_key, sample_key = jax.random.split(key, 4)

        # Update root if tree is empty (first call or after stepping to unexplored child)
        root_n = eval_state.data.n[eval_state.ROOT_INDEX]
        eval_state = jax.lax.cond(
            root_n == 0,
            lambda: self._initialize_root(root_key, eval_state, env_state, params, root_metadata),
            lambda: eval_state
        )

        # Get root policy for Gumbel sampling
        root_policy = eval_state.data.p[eval_state.ROOT_INDEX][:self.policy_size]
        root_logits = jnp.log(jnp.maximum(root_policy, 1e-8))

        # Get legal mask and pad if needed
        legal_mask = root_metadata.action_mask
        mask_size = legal_mask.shape[-1]
        legal_mask = jax.lax.cond(
            mask_size < self.policy_size,
            lambda: jnp.zeros(self.policy_size, dtype=bool).at[:mask_size].set(legal_mask),
            lambda: legal_mask[:self.policy_size]
        )

        # Sample k actions using Gumbel-Top-k (this replaces Dirichlet noise for exploration)
        selected_actions, gumbel_logits = gumbel_top_k(gumbel_key, root_logits, self.gumbel_k, legal_mask)

        # Initialize Sequential Halving state
        # All k actions start as active
        active_mask = jnp.ones(self.gumbel_k, dtype=bool)

        # Compute number of phases and iterations per phase
        num_phases = sequential_halving_phases(self.num_iterations, self.gumbel_k)
        iters_per_phase = self.num_iterations // max(num_phases, 1)

        # Run MCTS iterations with Sequential Halving
        def scan_body(carry, iteration):
            state, k, active = carry
            k, iter_key = jax.random.split(k)

            # Run one iteration
            new_state = self._gumbel_iterate(
                iter_key, state, iteration, params, selected_actions, active, gumbel_logits, legal_mask
            )

            # Check if we should eliminate half at phase boundary
            # Phase boundary occurs every iters_per_phase iterations (but not at iteration 0)
            is_phase_boundary = jnp.logical_and(
                iteration > 0,
                (iteration + 1) % iters_per_phase == 0
            )

            # Get Q-values and visit counts for elimination decision
            raw_q = new_state.get_child_data('q', new_state.ROOT_INDEX)[:self.policy_size]
            root_visits = new_state.get_child_data('n', new_state.ROOT_INDEX)[:self.policy_size]

            # Flip Q-values for two-player games (same logic as final selection)
            root_node = new_state.data_at(new_state.ROOT_INDEX)
            root_player = root_node.embedding.current_player
            child_indices = new_state.edge_map[new_state.ROOT_INDEX, :self.policy_size]
            safe_child_indices = jnp.maximum(child_indices, 0)
            child_players = new_state.data.embedding.current_player[safe_child_indices]
            child_exists = child_indices != new_state.NULL_INDEX
            player_diff = jnp.abs(root_player - child_players)
            per_child_discount = 1.0 - 2.0 * player_diff
            per_child_discount = jnp.where(child_exists, per_child_discount, 1.0)
            root_q = raw_q * per_child_discount

            # Eliminate half of active actions based on Gumbel scores
            new_active = jax.lax.cond(
                is_phase_boundary,
                lambda: eliminate_half(
                    selected_actions, active, gumbel_logits, root_q, root_visits,
                    self.c_visit, self.c_scale
                ),
                lambda: active
            )

            return (new_state, k, new_active), None

        iterations = jnp.arange(self.num_iterations)
        (eval_state, _, final_active), _ = jax.lax.scan(
            scan_body, (eval_state, key, active_mask), iterations
        )

        # Get final Q-values and visit counts
        raw_root_q = eval_state.get_child_data('q', eval_state.ROOT_INDEX)[:self.policy_size]
        root_visits = eval_state.get_child_data('n', eval_state.ROOT_INDEX)[:self.policy_size]

        # CRITICAL: Flip Q-values for two-player games
        # Child Q-values are from child's perspective; we need parent's perspective
        # For two-player games: if child is opponent, negate Q-value
        root_node = eval_state.data_at(eval_state.ROOT_INDEX)
        root_player = root_node.embedding.current_player
        child_indices = eval_state.edge_map[eval_state.ROOT_INDEX, :self.policy_size]
        safe_child_indices = jnp.maximum(child_indices, 0)
        child_players = eval_state.data.embedding.current_player[safe_child_indices]
        child_exists = child_indices != eval_state.NULL_INDEX

        # Discount: 1.0 if same player, -1.0 if different player
        player_diff = jnp.abs(root_player - child_players)
        per_child_discount = 1.0 - 2.0 * player_diff
        per_child_discount = jnp.where(child_exists, per_child_discount, 1.0)

        # Apply discount to get Q-values from root player's perspective
        root_q = raw_root_q * per_child_discount

        # Get root value for completing unvisited Q-values
        root_value = eval_state.data.q[eval_state.ROOT_INDEX]

        # Select action using Gumbel scores (g(a) + logits(a) + σ(q̂(a)))
        action = select_action_after_halving(
            gumbel_logits, root_q, root_visits, selected_actions, legal_mask,
            self.c_visit, self.c_scale
        )

        # Compute improved policy for training
        # Uses logits (not gumbel_logits) and completed Q-values per the paper:
        # π'(a) ∝ exp(logits(a) + σ(q̂(a)))
        policy_weights = compute_improved_policy(
            root_logits, root_q, root_visits, legal_mask, root_value,
            self.c_visit, self.c_scale
        )

        # Apply temperature for action sampling (if not greedy)
        action = self._apply_temperature_to_action(sample_key, action, policy_weights, legal_mask)

        # Compute MCTS stats
        tree_size = eval_state.next_free_idx
        root_visits_total = eval_state.data.n[eval_state.ROOT_INDEX]

        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=policy_weights[:self.policy_size],
            tree_size=tree_size,
            root_visits=root_visits_total,
        )

    def _apply_temperature_to_action(
        self,
        key: chex.PRNGKey,
        greedy_action: int,
        policy_weights: chex.Array,
        legal_mask: chex.Array,
    ) -> int:
        """Apply temperature to action selection.

        If temperature > 0, sample from policy_weights raised to 1/temperature.
        If temperature == 0, return greedy action.
        """
        safe_temp = jnp.maximum(self.temperature, 1e-8)

        def sample_with_temp():
            tempered = jnp.power(jnp.maximum(policy_weights, 1e-10), 1.0 / safe_temp)
            tempered = jnp.where(legal_mask, tempered, 0.0)
            tempered = tempered / jnp.maximum(jnp.sum(tempered), 1e-8)
            return jax.random.choice(key, self.policy_size, p=tempered)

        return jax.lax.cond(self.temperature > 0, sample_with_temp, lambda: greedy_action)

    def _evaluate_stochastic_root(
        self,
        key: chex.PRNGKey,
        eval_state: StochasticMCTSTree,
        env_state: chex.ArrayTree,
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
    ) -> MCTSOutput:
        """Evaluate when root is a stochastic node (chance event needed)."""
        key, root_key, sample_key = jax.random.split(key, 3)

        # Update root if empty
        root_n = eval_state.data.n[eval_state.ROOT_INDEX]
        eval_state = jax.lax.cond(
            root_n == 0,
            lambda: self._initialize_stochastic_root(root_key, eval_state, env_state, params, root_metadata),
            lambda: eval_state
        )

        # Run MCTS iterations (stochastic selection at root)
        def scan_body(carry, _):
            state, k = carry
            k, iter_key = jax.random.split(k)
            new_state = self._stochastic_root_iterate(iter_key, state, params)
            return (new_state, k), None

        (eval_state, _), _ = jax.lax.scan(scan_body, (eval_state, key), None, length=self.num_iterations)

        # Sample stochastic action from probabilities
        action = jax.random.choice(sample_key, self.stochastic_size, p=self.stochastic_action_probs)

        # Policy weights for stochastic root are just the probabilities
        policy_weights = jnp.zeros(self.policy_size)  # No decision policy for stochastic root

        # Compute MCTS stats
        tree_size = eval_state.next_free_idx
        root_visits = eval_state.data.n[eval_state.ROOT_INDEX]

        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=policy_weights,
            tree_size=tree_size,
            root_visits=root_visits,
        )

    def _initialize_root(
        self,
        key: chex.PRNGKey,
        tree: StochasticMCTSTree,
        env_state: chex.ArrayTree,
        params: chex.ArrayTree,
        root_metadata: StepMetadata,
    ) -> StochasticMCTSTree:
        """Initialize root node for decision state.

        Uses clean policy without Dirichlet noise. Exploration is handled by
        Gumbel-Top-k sampling in _evaluate_decision_root.
        """
        policy_logits, value = self.eval_fn(env_state, params, key)

        # Convert 4-way value to scalar if needed
        scalar_value = _value_to_scalar(value)

        # Mask illegal actions
        policy_logits = jnp.where(
            root_metadata.action_mask[:self.policy_size],
            policy_logits,
            jnp.finfo(jnp.float32).min
        )
        policy = jax.nn.softmax(policy_logits)

        # Pad policy to branching_factor (no Dirichlet noise - Gumbel handles exploration)
        full_policy = jnp.zeros(self.branching_factor)
        full_policy = full_policy.at[:self.policy_size].set(policy)

        root_node = self._new_node(
            policy=full_policy,
            value=scalar_value,
            embedding=env_state,
            terminated=root_metadata.terminated,
            nn_value_estimate=scalar_value,
            is_chance_node=False  # Decision root is not a chance node
        )
        return tree.set_root(root_node)

    def _initialize_stochastic_root(
        self,
        key: chex.PRNGKey,
        tree: StochasticMCTSTree,
        env_state: chex.ArrayTree,
        params: chex.ArrayTree,
        root_metadata: StepMetadata,
    ) -> StochasticMCTSTree:
        """Initialize root node for stochastic state."""
        # For stochastic root, we use the stochastic_action_probs as policy
        _, value = self.eval_fn(env_state, params, key)

        # Convert 4-way value to scalar if needed
        scalar_value = _value_to_scalar(value)

        # Policy for stochastic root: probabilities in stochastic slots
        full_policy = jnp.zeros(self.branching_factor)
        full_policy = full_policy.at[self.policy_size:self.policy_size + self.stochastic_size].set(
            self.stochastic_action_probs
        )

        root_node = self._new_node(
            policy=full_policy,
            value=scalar_value,
            embedding=env_state,
            terminated=root_metadata.terminated,
            nn_value_estimate=scalar_value,
            is_chance_node=True  # Stochastic root IS a chance node
        )
        return tree.set_root(root_node)

    def _gumbel_iterate(
        self,
        key: chex.PRNGKey,
        tree: StochasticMCTSTree,
        iteration: int,
        params: chex.ArrayTree,
        selected_actions: chex.Array,
        active_mask: chex.Array,
        gumbel_logits: chex.Array,
        legal_mask: chex.Array,
    ) -> StochasticMCTSTree:
        """Single MCTS iteration with Sequential Halving at root.

        Uses get_action_to_simulate to select which action to explore,
        prioritizing actions with minimum visits among active actions.
        """
        traverse_key, expand_key, backprop_key = jax.random.split(key, 3)

        # Get current visit counts for selected actions
        root_child_visits = tree.get_child_data('n', tree.ROOT_INDEX)[:self.policy_size]

        # Select action using Sequential Halving (minimum visits among active legal actions)
        root_action = get_action_to_simulate(
            iteration, selected_actions, root_child_visits, active_mask, traverse_key, legal_mask
        )

        # Traverse tree from root
        traversal_state = self._traverse(traverse_key, tree, root_action)
        parent, action = traversal_state.parent, traversal_state.action

        # Expand leaf node
        is_parent_stochastic = self._is_chance_node(tree, parent)

        result = jax.lax.cond(
            is_parent_stochastic,
            lambda: self._expand_stochastic_child(expand_key, tree, parent, action, params),
            lambda: self._expand_decision_child(expand_key, tree, parent, action, params)
        )

        # Backpropagate
        return self._backpropagate(backprop_key, result.tree, parent, result.child_idx, result.value)

    def _stochastic_root_iterate(
        self,
        key: chex.PRNGKey,
        tree: StochasticMCTSTree,
        params: chex.ArrayTree,
    ) -> StochasticMCTSTree:
        """Single iteration when root is stochastic."""
        traverse_key, expand_key, backprop_key = jax.random.split(key, 3)

        # Use delta-based selection to keep visits proportional to probabilities
        root_action = self._stochastic_action_selector(traverse_key, tree, tree.ROOT_INDEX)

        # Traverse from root
        traversal_state = self._traverse(traverse_key, tree, root_action)
        parent, action = traversal_state.parent, traversal_state.action

        # Expand
        is_parent_stochastic = self._is_chance_node(tree, parent)
        result = jax.lax.cond(
            is_parent_stochastic,
            lambda: self._expand_stochastic_child(expand_key, tree, parent, action, params),
            lambda: self._expand_decision_child(expand_key, tree, parent, action, params)
        )

        return self._backpropagate(backprop_key, result.tree, parent, result.child_idx, result.value)

    def _traverse(
        self,
        key: chex.PRNGKey,
        tree: StochasticMCTSTree,
        root_action: int,
    ) -> TraversalState:
        """Traverse tree from root to leaf."""
        root_child_exists = tree.is_edge(tree.ROOT_INDEX, root_action)
        root_child_idx = tree.edge_map[tree.ROOT_INDEX, root_action]

        def traverse_deeper():
            safe_child_idx = jnp.maximum(root_child_idx, 0)
            child_terminated = tree.data.terminated[safe_child_idx]

            def do_traverse():
                init_action = self._cond_action_selector(key, tree, root_child_idx)
                init_state = TraversalState(parent=root_child_idx, action=init_action)

                def cond_fn(s):
                    child_idx = tree.edge_map[s.parent, s.action]
                    safe_idx = jnp.maximum(child_idx, 0)
                    return jnp.logical_and(
                        tree.is_edge(s.parent, s.action),
                        ~tree.data.terminated[safe_idx]
                    )

                def body_fn(s):
                    node_idx = tree.edge_map[s.parent, s.action]
                    iter_key = jax.random.fold_in(key, node_idx)
                    next_action = self._cond_action_selector(iter_key, tree, node_idx)
                    return TraversalState(parent=node_idx, action=next_action)

                return jax.lax.while_loop(cond_fn, body_fn, init_state)

            return jax.lax.cond(
                child_terminated,
                lambda: TraversalState(parent=tree.ROOT_INDEX, action=root_action),
                do_traverse
            )

        def start_from_root():
            return TraversalState(parent=tree.ROOT_INDEX, action=root_action)

        return jax.lax.cond(root_child_exists, traverse_deeper, start_from_root)

    def _cond_action_selector(self, key: chex.PRNGKey, tree: StochasticMCTSTree, node_idx: int) -> int:
        """Select action based on node type using branchless arithmetic.

        Why: Replaces control flow with simple math, allowing XLA to fuse the entire
        traversal loop into a single efficient GPU kernel.
        """
        # 1. Cast boolean flag to float (0.0 or 1.0)
        # is_chance = 1.0 if stochastic, 0.0 if decision
        is_chance = self._is_chance_node(tree, node_idx).astype(jnp.float32)

        # 2. Compute BOTH candidates unconditionally
        # This is safe because both selectors rely on array lookups that are valid for any node
        stochastic_action = self._stochastic_action_selector(key, tree, node_idx)
        decision_action = self.action_selector(tree, node_idx)

        # 3. Blend results using the mask
        # If is_chance=1.0: (1.0 * stoch) + (0.0 * decis) = stoch
        # If is_chance=0.0: (0.0 * stoch) + (1.0 * decis) = decis
        final_action = (is_chance * stochastic_action) + ((1.0 - is_chance) * decision_action)

        # Cast back to int for indexing
        return final_action.astype(jnp.int32)

    def _stochastic_action_selector(self, key: chex.PRNGKey, tree: StochasticMCTSTree, node_idx: int) -> int:
        """Select action at a stochastic node.

        All outcomes are expanded when the chance node is first created.
        Selection is by greatest delta (expected - actual visits) to keep
        visit distribution proportional to outcome probabilities.
        """
        stoch_start = self.policy_size
        stoch_end = self.policy_size + self.stochastic_size

        # Get child indices and visit counts
        child_indices = tree.edge_map[node_idx, stoch_start:stoch_end]
        safe_indices = jnp.maximum(child_indices, 0)
        child_visits = tree.data.n[safe_indices]

        # Select by greatest delta (most under-visited relative to probability)
        total_visits = jnp.sum(child_visits)
        expected_visits = self.stochastic_action_probs * total_visits
        delta = expected_visits - child_visits

        # Add small noise for tiebreaking
        noise = jax.random.uniform(key, delta.shape) * 1e-6
        outcome = jnp.argmax(delta + noise)

        return self.policy_size + outcome

    def _is_chance_node(self, tree: StochasticMCTSTree, node_idx: int) -> chex.Array:
        """Check if node is a chance node (stochastic).

        Uses tree.data.is_chance_node field from StochasticMCTSNode.
        """
        return tree.data.is_chance_node[node_idx]

    def _expand_decision_child(
        self,
        key: chex.PRNGKey,
        tree: StochasticMCTSTree,
        parent: int,
        action: int,
        params: chex.ArrayTree,
    ) -> ExpandResult:
        """Expand a child from a decision node.

        If the child is a chance node, immediately expands all stochastic outcomes.
        """
        step_key, eval_key, expand_key = jax.random.split(key, 3)

        # Get parent embedding
        parent_embedding = jax.tree_util.tree_map(lambda x: x[parent], tree.data.embedding)

        # Step environment
        new_embedding, metadata = self.decision_step_fn(parent_embedding, action, step_key)

        # Check if node already exists
        node_exists = tree.is_edge(parent, action)
        node_idx = tree.edge_map[parent, action]

        is_next_stochastic = getattr(metadata, 'is_stochastic', jnp.array(False))

        def update_existing():
            # Node exists - just evaluate and update stats
            _, value = self.eval_fn(new_embedding, params, eval_key)
            scalar_value = _value_to_scalar(value)
            player_reward = metadata.rewards[metadata.cur_player_id]
            scalar_value = jnp.where(metadata.terminated, player_reward, scalar_value)
            updated_tree = self._update_node_stats(tree, node_idx, scalar_value)
            return ExpandResult(tree=updated_tree, child_idx=node_idx, value=scalar_value)

        def add_new():
            # Evaluate new state
            policy_logits, value = self.eval_fn(new_embedding, params, eval_key)
            scalar_value = _value_to_scalar(value)

            # Handle terminal state
            player_reward = metadata.rewards[metadata.cur_player_id]
            scalar_value = jnp.where(metadata.terminated, player_reward, scalar_value)

            # Mask illegal actions
            policy_logits = jnp.where(
                metadata.action_mask[:self.policy_size],
                policy_logits,
                jnp.finfo(jnp.float32).min
            )
            policy = jax.nn.softmax(policy_logits)

            # Build full policy
            full_policy = jnp.zeros(self.branching_factor)
            full_policy = jax.lax.cond(
                is_next_stochastic,
                lambda: full_policy.at[self.policy_size:self.policy_size + self.stochastic_size].set(
                    self.stochastic_action_probs if self.is_stochastic_game else jnp.zeros(max(1, self.stochastic_size))
                ),
                lambda: full_policy.at[:self.policy_size].set(policy)
            )

            # Create the child node
            node_data = self._new_node(
                policy=full_policy,
                value=scalar_value,
                embedding=new_embedding,
                terminated=metadata.terminated,
                nn_value_estimate=scalar_value,
                is_chance_node=is_next_stochastic
            )
            new_tree = tree.add_node(parent_index=parent, edge_index=action, data=node_data)
            child_idx = new_tree.edge_map[parent, action]

            # If this is a chance node, immediately expand all stochastic outcomes
            def expand_all_outcomes():
                return self._expand_all_stochastic_children(
                    expand_key, new_tree, child_idx, new_embedding, params
                )

            def keep_single():
                return ExpandResult(tree=new_tree, child_idx=child_idx, value=scalar_value)

            return jax.lax.cond(
                jnp.logical_and(is_next_stochastic, self.is_stochastic_game),
                expand_all_outcomes,
                keep_single
            )

        return jax.lax.cond(node_exists, update_existing, add_new)

    def _expand_all_stochastic_children(
        self,
        key: chex.PRNGKey,
        tree: StochasticMCTSTree,
        chance_node_idx: int,
        chance_node_embedding: chex.ArrayTree,
        params: chex.ArrayTree,
    ) -> ExpandResult:
        """Expand ALL stochastic outcomes from a chance node at once using vmap.

        This is called immediately when a chance node is created, ensuring all
        outcomes are evaluated before any selection happens.

        Uses vmap to process all outcomes in parallel:
        1. Batch step: vmap over stochastic_step_fn for all outcomes
        2. Batch eval: vmap over eval_fn for all resulting states
        3. Sequential update: Add nodes to tree (inherently sequential)

        This reduces latency from O(K) to O(1) for the expensive NN evaluations.

        Returns weighted average value based on outcome probabilities.
        """
        # Generate all keys upfront for parallel processing
        outcome_indices = jnp.arange(self.stochastic_size)
        step_keys = jax.random.split(key, self.stochastic_size)
        eval_keys = jax.random.split(jax.random.fold_in(key, 1), self.stochastic_size)

        # Step 1: Batch step - vmap over all stochastic outcomes
        # This steps the environment for all outcomes in parallel
        def step_one_outcome(outcome_idx, step_key):
            return self.stochastic_step_fn(chance_node_embedding, outcome_idx, step_key)

        all_embeddings, all_metadata = jax.vmap(step_one_outcome)(outcome_indices, step_keys)

        # Step 2: Batch eval - vmap over all resulting states
        # This runs the neural network on all states in a single batched call
        def eval_one_state(embedding, eval_key):
            return self.eval_fn(embedding, params, eval_key)

        all_policy_logits, all_values = jax.vmap(eval_one_state)(all_embeddings, eval_keys)

        # Convert values to scalars (handle 4-way value heads)
        all_scalar_values = jax.vmap(_value_to_scalar)(all_values)

        # Handle terminal states - use rewards if terminated
        all_player_rewards = jax.vmap(
            lambda rewards, player_id: rewards[player_id]
        )(all_metadata.rewards, all_metadata.cur_player_id)
        all_scalar_values = jnp.where(
            all_metadata.terminated, all_player_rewards, all_scalar_values
        )

        # Build policies for all outcomes in parallel
        def build_policy(policy_logits, action_mask, is_next_stochastic):
            # Mask illegal actions
            masked_logits = jnp.where(
                action_mask[:self.policy_size],
                policy_logits,
                jnp.finfo(jnp.float32).min
            )
            policy = jax.nn.softmax(masked_logits)

            # Build full policy based on whether next state is stochastic
            full_policy = jnp.zeros(self.branching_factor)
            stoch_policy = full_policy.at[self.policy_size:self.policy_size + self.stochastic_size].set(
                self.stochastic_action_probs
            )
            decision_policy = full_policy.at[:self.policy_size].set(policy)

            # Branchless selection between stochastic and decision policy
            is_stoch_float = is_next_stochastic.astype(jnp.float32)
            return is_stoch_float * stoch_policy + (1.0 - is_stoch_float) * decision_policy

        all_is_next_stochastic = getattr(all_metadata, 'is_stochastic', jnp.zeros(self.stochastic_size, dtype=bool))
        all_policies = jax.vmap(build_policy)(all_policy_logits, all_metadata.action_mask, all_is_next_stochastic)

        # Step 3: Sequential update - add all nodes to tree
        # Tree updates are inherently sequential, but the expensive computation is done
        def add_one_node(current_tree, idx):
            # Extract data for this outcome
            embedding = jax.tree_util.tree_map(lambda x: x[idx], all_embeddings)
            policy = all_policies[idx]
            value = all_scalar_values[idx]
            terminated = all_metadata.terminated[idx]
            is_chance = all_is_next_stochastic[idx]

            node_data = self._new_node(
                policy=policy,
                value=value,
                embedding=embedding,
                terminated=terminated,
                nn_value_estimate=value,
                is_chance_node=is_chance
            )

            # Add to tree at action = policy_size + outcome_idx
            action = self.policy_size + idx
            updated_tree = current_tree.add_node(
                parent_index=chance_node_idx,
                edge_index=action,
                data=node_data
            )
            return updated_tree, None

        final_tree, _ = jax.lax.scan(add_one_node, tree, outcome_indices)

        # Compute weighted average value
        weighted_value = jnp.sum(all_scalar_values * self.stochastic_action_probs)

        return ExpandResult(tree=final_tree, child_idx=chance_node_idx, value=weighted_value)

    def _expand_stochastic_child(
        self,
        key: chex.PRNGKey,
        tree: StochasticMCTSTree,
        parent: int,
        action: int,
        params: chex.ArrayTree,
    ) -> ExpandResult:
        """Expand a child from a stochastic node."""
        step_key, eval_key = jax.random.split(key)

        # Get parent embedding
        parent_embedding = jax.tree_util.tree_map(lambda x: x[parent], tree.data.embedding)

        # Convert action to stochastic outcome
        outcome = action - self.policy_size

        # Step with stochastic outcome
        new_embedding, metadata = self.stochastic_step_fn(parent_embedding, outcome, step_key)

        # Evaluate new state
        policy_logits, value = self.eval_fn(new_embedding, params, eval_key)

        # Convert 4-way value to scalar if needed
        scalar_value = _value_to_scalar(value)

        # Handle terminal
        player_reward = metadata.rewards[metadata.cur_player_id]
        scalar_value = jnp.where(metadata.terminated, player_reward, scalar_value)

        # Mask and build policy
        policy_logits = jnp.where(
            metadata.action_mask[:self.policy_size],
            policy_logits,
            jnp.finfo(jnp.float32).min
        )
        policy = jax.nn.softmax(policy_logits)

        full_policy = jnp.zeros(self.branching_factor)
        is_next_stochastic = getattr(metadata, 'is_stochastic', jnp.array(False))

        full_policy = jax.lax.cond(
            is_next_stochastic,
            lambda: full_policy.at[self.policy_size:self.policy_size + self.stochastic_size].set(
                self.stochastic_action_probs if self.is_stochastic_game else jnp.zeros(max(1, self.stochastic_size))
            ),
            lambda: full_policy.at[:self.policy_size].set(policy)
        )

        node_exists = tree.is_edge(parent, action)
        node_idx = tree.edge_map[parent, action]

        def update_existing():
            updated_tree = self._update_node_stats(tree, node_idx, scalar_value)
            return ExpandResult(tree=updated_tree, child_idx=node_idx, value=scalar_value)

        def add_new():
            node_data = self._new_node(
                policy=full_policy,
                value=scalar_value,
                embedding=new_embedding,
                terminated=metadata.terminated,
                nn_value_estimate=scalar_value,
                is_chance_node=is_next_stochastic  # Child is chance node if next state is stochastic
            )
            new_tree = tree.add_node(parent_index=parent, edge_index=action, data=node_data)
            child_idx = new_tree.edge_map[parent, action]
            return ExpandResult(tree=new_tree, child_idx=child_idx, value=scalar_value)

        return jax.lax.cond(node_exists, update_existing, add_new)

    def _backpropagate(
        self,
        key: chex.PRNGKey,
        tree: StochasticMCTSTree,
        parent: int,
        child: int,
        value: float,
    ) -> StochasticMCTSTree:
        """Backpropagate value through tree with player perspective handling."""
        # Initial discount from parent to child
        initial_discount = self._calculate_discount_factor(tree, parent, child)
        value = value * initial_discount

        def body_fn(state: BackpropState) -> BackpropState:
            node_idx, val, t = state.node_idx, state.value, state.tree
            t = self._update_node_stats(t, node_idx, val)
            parent_idx = t.parents[node_idx]
            discount = self._calculate_discount_factor(t, parent_idx, node_idx)
            return BackpropState(node_idx=parent_idx, value=val * discount, tree=t)

        state = jax.lax.while_loop(
            lambda s: s.node_idx != s.tree.ROOT_INDEX,
            body_fn,
            BackpropState(node_idx=parent, value=value, tree=tree)
        )

        # Update root
        return self._update_node_stats(state.tree, state.tree.ROOT_INDEX, state.value)

    def _calculate_discount_factor(self, tree: StochasticMCTSTree, node_idx: int, other_idx: int) -> float:
        """Calculate discount factor for two-player perspective."""
        current_player = tree.data.embedding.current_player[node_idx]
        other_player = tree.data.embedding.current_player[other_idx]
        player_diff = jnp.abs(current_player - other_player)
        return 1.0 - 2.0 * player_diff

    def _update_node_stats(self, tree: StochasticMCTSTree, node_idx: int, value: float) -> StochasticMCTSTree:
        """Update visit count and Q-value for a node."""
        old_n = tree.data.n[node_idx]
        old_q = tree.data.q[node_idx]
        new_n = old_n + 1
        new_q = (old_q * old_n + value) / new_n

        new_data = tree.data.replace(
            n=tree.data.n.at[node_idx].set(new_n),
            q=tree.data.q.at[node_idx].set(new_q)
        )
        return tree.replace(data=new_data)

    def _sample_root_action(
        self,
        key: chex.PRNGKey,
        tree: StochasticMCTSTree,
        action_mask: chex.Array,
    ) -> Tuple[int, chex.Array]:
        """Sample action from root visit counts."""
        action_visits = tree.get_child_data('n', tree.ROOT_INDEX)[:self.policy_size]

        # Mask illegal actions
        mask_size = action_mask.shape[-1]
        padded_mask = jax.lax.cond(
            mask_size < self.policy_size,
            lambda: jnp.zeros(self.policy_size, dtype=bool).at[:mask_size].set(action_mask),
            lambda: action_mask[:self.policy_size]
        )
        action_visits = jnp.where(padded_mask, action_visits, 0)

        # Compute policy weights
        total_visits = jnp.sum(action_visits)
        policy_weights = action_visits / jnp.maximum(total_visits, 1)

        # Handle zero visits (uniform over legal)
        policy_weights = jnp.where(
            total_visits > 0,
            policy_weights,
            padded_mask / jnp.maximum(jnp.sum(padded_mask), 1)
        )

        # Sample with temperature (use max(temp, 1e-8) to avoid division by zero in JAX tracing)
        safe_temp = jnp.maximum(self.temperature, 1e-8)

        def sample_with_temp():
            tempered = jnp.power(policy_weights, 1.0 / safe_temp)
            tempered = tempered / jnp.maximum(jnp.sum(tempered), 1e-8)
            return jax.random.choice(key, self.policy_size, p=tempered)

        def greedy_select():
            noise = jax.random.uniform(key, shape=(self.policy_size,), maxval=self.tiebreak_noise)
            return jnp.argmax(policy_weights + noise)

        action = jax.lax.cond(self.temperature > 0, sample_with_temp, greedy_select)

        return action, policy_weights

    def step(self, tree: StochasticMCTSTree, action: int) -> StochasticMCTSTree:
        """Update tree after taking action. Always persists subtree.

        Args:
            tree: Current tree state
            action: Action taken in the environment

        Returns:
            Updated tree with subtree rooted at action's child
        """
        return tree.get_subtree(action)

    def get_value(self, tree: StochasticMCTSTree) -> float:
        """Get value estimate from root node."""
        return tree.data.q[tree.ROOT_INDEX]

    def reset(self, tree: StochasticMCTSTree) -> StochasticMCTSTree:
        """Reset tree to empty state."""
        return tree.reset()

    def get_tree_stats(self, tree: StochasticMCTSTree) -> dict:
        """Get statistics about the MCTS tree.

        Args:
            tree: Current tree state

        Returns:
            Dictionary with tree statistics:
            - mcts/tree_size: number of nodes in tree
            - mcts/root_visits: number of visits to root node
            - mcts/max_nodes: maximum nodes allowed
            - mcts/num_iterations: number of MCTS iterations per move
        """
        tree_size = tree.next_free_idx
        root_visits = tree.data.n[tree.ROOT_INDEX]
        return {
            "mcts/tree_size": float(tree_size),
            "mcts/root_visits": float(root_visits),
            "mcts/max_nodes": float(self.max_nodes),
            "mcts/num_iterations": float(self.num_iterations),
        }

    def _new_node(
        self,
        policy: chex.Array,
        value: float,
        embedding: chex.ArrayTree,
        terminated: bool,
        nn_value_estimate: float = 0.0,
        is_chance_node: bool = False,
    ) -> StochasticMCTSNode:
        """Create a new MCTS node with stochastic game support."""
        # Determine outcome probs size
        outcome_size = max(1, self.stochastic_size) if self.is_stochastic_game else 1

        return StochasticMCTSNode(
            n=jnp.array(1, dtype=jnp.int32),
            p=policy,
            q=jnp.array(value, dtype=jnp.float32),
            terminated=jnp.array(terminated, dtype=jnp.bool_),
            embedding=embedding,
            is_chance_node=jnp.array(is_chance_node, dtype=jnp.bool_),
            nn_value_estimate=jnp.array(nn_value_estimate, dtype=jnp.float32),
            expanded_outcomes=jnp.zeros(outcome_size, dtype=jnp.bool_),
            outcome_probs=self.stochastic_action_probs if self.is_stochastic_game else jnp.ones(1)
        )
