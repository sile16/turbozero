from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
import chex
from chex import dataclass
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.state import (
    BackpropState, MCTSNode, MCTSTree, TraversalState, MCTSOutput,
    StochasticMCTSNode, StochasticMCTSTree
)
from core.evaluators.mcts.mcts import MCTS
from core.evaluators.alphazero import AlphaZero
from core.trees.tree import init_tree
from core.types import EvalFn, StepMetadata

# Type aliases for the two step functions (pgx 3.1.0 API - no key needed)
# Decision step: player chooses action
DecisionStepFn = Callable[[chex.ArrayTree, int], Tuple[chex.ArrayTree, StepMetadata]]
# Stochastic step: environment samples outcome
StochasticStepFn = Callable[[chex.ArrayTree, int], Tuple[chex.ArrayTree, StepMetadata]]


@dataclass(frozen=True)
class ExpandResult:
    """Result of expanding child node(s)."""
    tree: MCTSTree
    child_idx: int
    value: float


class StochasticMCTS(AlphaZero(MCTS)):
    """MCTS for games with stochastic transitions (e.g., 2048, Pig, backgammon).

    This class handles games where `is_stochastic=True` indicates a CHANCE NODE - a state
    where the environment determines the next outcome randomly, not the player.

    Key Design: Separate step functions for decisions vs chance outcomes
    =====================================================================
    Uses pgx 3.1.0 API (no random keys needed for step functions):

    - `decision_step_fn`: Called at decision nodes (player chooses action)
      Signature: (state, action) -> (new_state, metadata)
      Example: env.step_deterministic(state, action)

    - `stochastic_step_fn`: Called at chance nodes (environment samples outcome)
      Signature: (state, outcome) -> (new_state, metadata)
      Example: env.step_stochastic(state, outcome)

    Progressive Expansion at Chance Nodes
    =====================================
    Instead of expanding all stochastic children at once, we use progressive
    expansion based on the formula: parent_visits * child_prob > threshold

    This means:
    - High probability outcomes are expanded early
    - Low probability outcomes are expanded only when parent has many visits
    - Unexpanded children contribute to value via blended backprop

    Blended Value Backpropagation
    =============================
    At chance nodes, the value used for backprop is:
        blended_value = expanded_prob * expanded_avg_value + (1 - expanded_prob) * nn_estimate

    This allows proper value estimation even when not all children are expanded.

    Example for pgx 3.1.0 2048:
    ```python
    env = pgx.make('2048')

    # Decision step: player slides tiles (4 actions)
    decision_step_fn = lambda state, action: (
        env.step_deterministic(state, action),
        StepMetadata(action_mask=state.legal_action_mask, ...)
    )

    # Stochastic step: tile spawns (32 outcomes)
    stochastic_step_fn = lambda state, outcome: (
        env.step_stochastic(state, outcome),
        StepMetadata(...)
    )

    evaluator = StochasticMCTS(
        policy_size=4,  # Neural network outputs 4 logits
        stochastic_action_probs=tile_spawn_probs,  # 32 outcomes
        decision_step_fn=decision_step_fn,
        stochastic_step_fn=stochastic_step_fn,
        progressive_threshold=1.0,  # Expand when parent_visits * prob > 1.0
        ...
    )
    ```

    When the root node is a chance node (is_stochastic=True), this class:
    - Skips MCTS search (no player decision to make)
    - Samples an action from `stochastic_action_probs`
    - Returns a dummy policy (zeros) - not used for training at chance nodes

    When traversing the tree:
    - At decision nodes: uses neural network policy with PUCT selection
    - At chance nodes: explores outcomes based on progressive expansion
    - Backpropagation uses blended expectimax at chance nodes

    Compatible with `jax.vmap`, `jax.pmap`, `jax.jit`, etc.
    """
    def __init__(self,
        eval_fn: EvalFn,
        action_selector: MCTSActionSelector,
        stochastic_action_probs: chex.Array,
        policy_size: int,
        max_nodes: int,
        num_iterations: int,
        decision_step_fn: Optional[DecisionStepFn] = None,
        stochastic_step_fn: Optional[StochasticStepFn] = None,
        temperature: float = 1.0,
        tiebreak_noise: float = 1e-8,
        persist_tree: bool = True,
        noise_scale: float = 0.05,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        min_num_iterations: int = 300,
        progressive_threshold: float = 1.0
    ):
        """
        Args:
        - `eval_fn`: leaf node evaluation function (env_state -> (policy_logits, value))
        - `action_selector`: action selection function (tree, node_idx -> action)
        - `stochastic_action_probs`: probability distribution over stochastic outcomes
        - `policy_size`: size of neural network policy output (number of decision actions)
        - `max_nodes`: allocated size of MCTS tree
        - `num_iterations`: number of MCTS iterations per evaluate call
        - `decision_step_fn`: step function for decision nodes (pgx step_deterministic)
        - `stochastic_step_fn`: step function for chance nodes (pgx step_stochastic)
        - `temperature`: temperature for root action selection (default: 1.0)
        - `tiebreak_noise`: noise magnitude for tie-breaking (default: 1e-8)
        - `persist_tree`: whether to persist tree between evaluate calls (default: True)
        - `noise_scale`: noise scale for stochastic action selection (default: 0.05)
        - `dirichlet_alpha`: Dirichlet noise magnitude (default: 0.3)
        - `dirichlet_epsilon`: proportion of Dirichlet noise in root policy (default: 0.25)
        - `progressive_threshold`: threshold for progressive expansion at chance nodes.
            Expand child when parent_visits * child_prob > threshold (default: 1.0)
        """
        # Tree branching factor = max of decision and stochastic action spaces
        stochastic_size = len(stochastic_action_probs)
        tree_branching_factor = max(policy_size, stochastic_size)

        super().__init__(
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            policy_size=policy_size,  # Pass to parent for Dirichlet noise sizing
            eval_fn=eval_fn,
            action_selector=action_selector,
            branching_factor=tree_branching_factor,
            max_nodes=max_nodes,
            num_iterations=num_iterations,
            temperature=temperature,
            tiebreak_noise=tiebreak_noise,
            persist_tree=persist_tree
        )

        self.stochastic_action_probs = stochastic_action_probs
        # policy_size is now set by parent, but also store stochastic_size
        self.stochastic_size = stochastic_size
        self.noise_scale = noise_scale
        self.max_num_iterations = self.num_iterations
        self.min_num_iterations = min(min_num_iterations, self.max_num_iterations)
        self.progressive_threshold = progressive_threshold

        # Store step functions if provided (can also be passed to evaluate)
        self._decision_step_fn = decision_step_fn
        self._stochastic_step_fn = stochastic_step_fn

    @property
    def handles_chance_nodes(self) -> bool:
        """StochasticMCTS explicitly handles chance nodes.

        At chance nodes, we use expectimax backprop and return placeholder policy weights.
        The trainer should skip policy loss for these samples.
        """
        return True

    def init(self, template_embedding: chex.ArrayTree, *args, **kwargs) -> StochasticMCTSTree:
        """Initialize a StochasticMCTSTree with proper node structure for progressive expansion.

        Args:
            template_embedding: Template environment state embedding

        Returns:
            StochasticMCTSTree with StochasticMCTSNode data structure
        """
        return init_tree(self.max_nodes, self.branching_factor, self.new_stochastic_node(
            policy=jnp.zeros((self.branching_factor,)),
            value=0.0,
            embedding=template_embedding,
            terminated=False,
            is_chance_node=False,
            nn_value_estimate=0.0,
            outcome_probs=self.stochastic_action_probs
        ))

    @staticmethod
    def new_stochastic_node(
        policy: chex.Array,
        value: float,
        embedding: chex.ArrayTree,
        terminated: bool,
        is_chance_node: bool,
        nn_value_estimate: float,
        outcome_probs: chex.Array
    ) -> StochasticMCTSNode:
        """Create a new StochasticMCTSNode for progressive expansion.

        Args:
            policy: Policy probabilities over decision actions
            value: Initial value estimate
            embedding: Environment state embedding
            terminated: Whether state is terminal
            is_chance_node: Whether this is a chance node (stochastic state)
            nn_value_estimate: NN value estimate (stored for blended backprop)
            outcome_probs: Probability distribution over stochastic outcomes

        Returns:
            StochasticMCTSNode with progressive expansion fields initialized
        """
        num_outcomes = outcome_probs.shape[0]
        return StochasticMCTSNode(
            n=jnp.array(1, dtype=jnp.int32),
            p=policy,
            q=jnp.array(value, dtype=jnp.float32),
            terminated=jnp.array(terminated, dtype=jnp.bool_),
            embedding=embedding,
            is_chance_node=jnp.array(is_chance_node, dtype=jnp.bool_),
            nn_value_estimate=jnp.array(nn_value_estimate, dtype=jnp.float32),
            expanded_outcomes=jnp.zeros(num_outcomes, dtype=jnp.bool_),
            outcome_probs=outcome_probs
        )

    def evaluate(self,
        key: chex.PRNGKey,
        eval_state: StochasticMCTSTree,
        env_state: chex.ArrayTree,
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
        env_step_fn: Optional[DecisionStepFn] = None,
        decision_step_fn: Optional[DecisionStepFn] = None,
        stochastic_step_fn: Optional[StochasticStepFn] = None,
        **kwargs
    ) -> MCTSOutput:
        """Performs MCTS iterations and samples an action.

        Args:
        - `eval_state`: StochasticMCTSTree to evaluate
        - `env_state`: current environment state
        - `root_metadata`: metadata for the root node
        - `params`: neural network parameters
        - `env_step_fn`: (deprecated) combined step function, use decision_step_fn instead
        - `decision_step_fn`: step function for decision nodes (pgx step_deterministic)
        - `stochastic_step_fn`: step function for chance nodes (pgx step_stochastic)

        Returns:
        - MCTSOutput: tree state, selected action, and policy weights
        """
        # Resolve step functions: parameter > instance > env_step_fn fallback
        dec_fn = decision_step_fn or self._decision_step_fn or env_step_fn
        stoch_fn = stochastic_step_fn or self._stochastic_step_fn or env_step_fn

        if dec_fn is None or stoch_fn is None:
            raise ValueError(
                "StochasticMCTS requires both decision_step_fn and stochastic_step_fn. "
                "Pass them to __init__ or evaluate()."
            )

        tree = eval_state
        # Use stored is_chance_node flag instead of metadata lookup
        is_root_stochastic = self._is_chance_node(tree, tree.ROOT_INDEX)

        # Use jax.lax.cond to select between stochastic and decision evaluation
        # This avoids shape mismatches from tree_map when internal tree states differ
        return jax.lax.cond(
            is_root_stochastic,
            lambda: self.stochastic_evaluate(
                key, eval_state, env_state, root_metadata, params, dec_fn, stoch_fn
            ),
            lambda: self._decision_evaluate(
                key, eval_state, env_state, root_metadata, params, dec_fn, stoch_fn
            )
        )

    def update_root(self, key: chex.PRNGKey, tree: StochasticMCTSTree, root_embedding: chex.ArrayTree,
                    params: chex.ArrayTree, root_metadata: StepMetadata = None, **kwargs) -> StochasticMCTSTree:
        """Populates the root node of a StochasticMCTSTree.

        Args:
            key: rng
            tree: StochasticMCTSTree to update
            root_embedding: root environment state
            params: nn parameters
            root_metadata: metadata containing is_stochastic flag

        Returns:
            StochasticMCTSTree with root node populated
        """
        # Evaluate root state
        root_policy_logits, root_value = self.eval_fn(root_embedding, params, key)
        root_policy = jax.nn.softmax(root_policy_logits)

        # Pad policy to branching factor if needed
        if root_policy.shape[0] < self.branching_factor:
            padded_policy = jnp.zeros(self.branching_factor)
            padded_policy = padded_policy.at[:root_policy.shape[0]].set(root_policy)
            root_policy = padded_policy

        # Determine if root is a chance node
        is_chance = root_metadata.is_stochastic if root_metadata is not None else False

        # Create root node with stochastic fields
        root_node = self.new_stochastic_node(
            policy=root_policy,
            value=root_value,
            embedding=root_embedding,
            terminated=False,
            is_chance_node=is_chance,
            nn_value_estimate=root_value,
            outcome_probs=self.stochastic_action_probs
        )

        # Check if root already visited
        visited = tree.data.n[tree.ROOT_INDEX] > 0

        # Update fields conditionally
        def get_updated_root():
            existing_node = tree.data_at(tree.ROOT_INDEX)
            return existing_node.replace(
                p=jnp.where(visited, existing_node.p, root_policy),
                q=jnp.where(visited, existing_node.q, root_value),
                n=jnp.where(visited, existing_node.n, 1),
                is_chance_node=jnp.where(visited, existing_node.is_chance_node, is_chance),
                nn_value_estimate=jnp.where(visited, existing_node.nn_value_estimate, root_value),
                embedding=jax.lax.cond(visited, lambda _: existing_node.embedding, lambda _: root_embedding, operand=None)
            )

        updated_root = get_updated_root()
        return tree.set_root(updated_root)

    def _decision_evaluate(self,
        key: chex.PRNGKey,
        eval_state: StochasticMCTSTree,
        env_state: chex.ArrayTree,
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
        decision_step_fn: DecisionStepFn,
        stochastic_step_fn: StochasticStepFn,
    ) -> MCTSOutput:
        """Standard MCTS evaluation for decision nodes."""
        key, root_key = jax.random.split(key)

        # Reset tree if persist_tree=False
        root_n = eval_state.data.n[eval_state.ROOT_INDEX]
        should_reset = jnp.logical_and(
            not self.persist_tree,
            root_n > 0
        )
        eval_state = jax.lax.cond(
            should_reset,
            lambda: eval_state.reset(),
            lambda: eval_state
        )

        # Update root if needed
        root_n = eval_state.data.n[eval_state.ROOT_INDEX]
        should_update_root = jnp.logical_or(
            not self.persist_tree,
            root_n == 0
        )
        eval_state = jax.lax.cond(
            should_update_root,
            lambda: self.update_root(root_key, eval_state, env_state, params, root_metadata=root_metadata),
            lambda: eval_state
        )

        # Perform MCTS iterations
        iterate_fn = lambda k, tree: self.iterate(k, tree, params, decision_step_fn, stochastic_step_fn)

        def scan_body(carry, _):
            state, k = carry
            k, iter_key = jax.random.split(k)
            new_state = iterate_fn(iter_key, state)
            return (new_state, k), None

        (eval_state, _), _ = jax.lax.scan(scan_body, (eval_state, key), None, length=self.num_iterations)

        # Sample action based on root visit counts
        action, policy_weights = self.sample_root_action(key, eval_state, action_mask=root_metadata.action_mask)
        # Truncate policy_weights to policy_size (tree uses branching_factor for edges)
        policy_weights = policy_weights[:self.policy_size]
        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=policy_weights
        )

    def stochastic_evaluate(self,
        key: chex.PRNGKey,
        eval_state: StochasticMCTSTree,
        env_state: chex.ArrayTree,
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
        decision_step_fn: DecisionStepFn,
        stochastic_step_fn: StochasticStepFn,
    ) -> MCTSOutput:
        """Evaluation when root is a chance node - just sample from stochastic probs."""
        key, root_key = jax.random.split(key)

        # Reset tree if persist_tree=False
        root_n = eval_state.data.n[eval_state.ROOT_INDEX]
        should_reset = jnp.logical_and(
            not self.persist_tree,
            root_n > 0
        )
        eval_state = jax.lax.cond(
            should_reset,
            lambda: eval_state.reset(),
            lambda: eval_state
        )

        # Update root if needed
        root_n = eval_state.data.n[eval_state.ROOT_INDEX]
        should_update_root = jnp.logical_or(
            not self.persist_tree,
            root_n == 0
        )
        eval_state = jax.lax.cond(
            should_update_root,
            lambda: self.update_root(root_key, eval_state, env_state, params, root_metadata=root_metadata),
            lambda: eval_state
        )

        # Sample stochastic action
        action = self.stochastic_action_sample(key, eval_state, eval_state.ROOT_INDEX)

        # Return uniform policy at chance nodes to avoid NaN in cross-entropy loss
        # (zeros would cause log(0) = -inf in training)
        # Ideally, training would skip policy loss at chance nodes, but uniform is stable
        policy_weights = jnp.ones(self.policy_size) / self.policy_size

        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=policy_weights
        )


    def stochastic_action_sample(self, key: chex.PRNGKey, tree, node_idx):
        """Sample an action from stochastic_action_probs (for chance nodes)."""
        action = jax.random.choice(key, self.stochastic_size, p=self.stochastic_action_probs)
        return jnp.array(action, dtype=jnp.int32)


    def deterministic_action_selector(self, key: chex.PRNGKey, tree: MCTSTree, node_idx: int) -> int:
        """Action selection at decision nodes using PUCT."""
        return self.action_selector(tree, node_idx)

    def stochastic_action_selector(self, key: chex.PRNGKey, tree, node_idx) -> int:
        """Action selection at chance nodes during tree traversal.

        Strategy:
        - Expand unexpanded outcomes first (prefer high-probability ones)
        - Once all expanded, select based on visit count discrepancy vs target probs
        """
        num_stochastic = self.stochastic_size

        # Get child indices for stochastic actions
        child_indices = tree.edge_map[node_idx, :num_stochastic]

        # Check which children exist
        child_exists_mask = child_indices != tree.NULL_INDEX
        unexpanded_mask = ~child_exists_mask
        has_unexpanded = jnp.any(unexpanded_mask)

        k1, k2 = jax.random.split(key)

        # Expand unexpanded: prefer higher probability outcomes
        masked_probs_expand = jnp.where(unexpanded_mask, self.stochastic_action_probs, -jnp.inf)
        noise1 = jax.random.normal(k1, masked_probs_expand.shape) * self.noise_scale
        expand_action = jnp.argmax(masked_probs_expand + noise1)

        # Probability matching: select action with largest visit deficit
        safe_indices = jnp.maximum(child_indices, 0)
        child_visits = jnp.where(child_exists_mask, tree.data.n[safe_indices], 0)
        total_visits = jnp.sum(child_visits)
        normalized_visits = jnp.where(
            total_visits > 0,
            child_visits / total_visits,
            jnp.ones_like(child_visits, dtype=jnp.float32) / num_stochastic
        )
        delta = self.stochastic_action_probs - normalized_visits
        noise2 = jax.random.normal(k2, delta.shape) * self.noise_scale
        match_action = jnp.argmax(delta + noise2)

        return jnp.where(has_unexpanded, expand_action, match_action)


    def cond_action_selector(self, key: chex.PRNGKey, tree: StochasticMCTSTree, node_idx: int) -> int:
        """Select action based on node type (decision vs chance)."""
        is_stochastic = self._is_chance_node(tree, node_idx)
        stochastic_action = self.stochastic_action_selector(key, tree, node_idx)
        deterministic_action = self.deterministic_action_selector(key, tree, node_idx)
        return jnp.where(is_stochastic, stochastic_action, deterministic_action)


    def value_policy(self, embedding, params, eval_key, metadata, player_reward) -> Tuple[float, chex.Array]:
        """Get value and policy for a node.

        The neural network outputs policy_size logits. We pad to tree_branching_factor
        for storage in the tree (extra slots remain zero/invalid).
        """
        policy_logits, value = self.eval_fn(embedding, params, eval_key)

        # Apply action mask (mask size = policy_size for decision nodes)
        action_mask = metadata.action_mask
        # Handle case where mask might be smaller than policy_logits
        mask_size = action_mask.shape[-1]
        policy_size = policy_logits.shape[-1]

        if mask_size < policy_size:
            # Pad mask to match policy size
            padded_mask = jnp.zeros(policy_size, dtype=bool)
            padded_mask = padded_mask.at[:mask_size].set(action_mask)
            action_mask = padded_mask
        elif mask_size > policy_size:
            # Truncate mask (shouldn't happen with proper setup)
            action_mask = action_mask[:policy_size]

        policy_logits = jnp.where(action_mask, policy_logits, jnp.finfo(policy_logits).min)
        policy = jax.nn.softmax(policy_logits)
        value = jnp.where(metadata.terminated, player_reward, value)

        # Pad policy to tree_branching_factor for node storage
        if policy_size < self.branching_factor:
            padded_policy = jnp.zeros(self.branching_factor)
            padded_policy = padded_policy.at[:policy_size].set(policy)
            policy = padded_policy

        return value, policy


    def iterate(self,
        key: chex.PRNGKey,
        tree: StochasticMCTSTree,
        params: chex.ArrayTree,
        decision_step_fn: DecisionStepFn,
        stochastic_step_fn: StochasticStepFn
    ) -> StochasticMCTSTree:
        """Performs one iteration of MCTS with stochastic expansion.

        Uses decision_step_fn for decision node expansion and
        stochastic_step_fn for chance node expansion.

        Key optimization: expands only ONE child per iteration (like regular MCTS).
        The stochastic_action_selector picks which outcome to expand/visit.
        """
        traverse_key, expand_key, backprop_key = jax.random.split(key, 3)

        # Traverse from root to leaf
        traversal_state = self.traverse(traverse_key, tree)
        parent, action = traversal_state.parent, traversal_state.action

        # Check if parent is stochastic (chance node)
        is_parent_stochastic = self._is_chance_node(tree, parent)

        # Compute BOTH expansion strategies (for vmap compatibility)
        # Decision expansion: single child using decision_step_fn
        decision_result = self._expand_single_child(
            expand_key, tree, parent, action, params, decision_step_fn
        )
        # Stochastic expansion: single child using stochastic_step_fn
        stochastic_result = self._expand_single_stochastic_child(
            expand_key, tree, parent, action, params, stochastic_step_fn
        )

        # Select result based on node type
        final_tree = jax.tree_util.tree_map(
            lambda a, b: jnp.where(is_parent_stochastic, a, b),
            stochastic_result.tree, decision_result.tree
        )
        final_value = jnp.where(is_parent_stochastic, stochastic_result.value, decision_result.value)
        final_child = jnp.where(is_parent_stochastic, stochastic_result.child_idx, decision_result.child_idx)

        return self.backpropagate(backprop_key, final_tree, parent, final_child, final_value)

    def _is_chance_node(self, tree: StochasticMCTSTree, node_idx: int) -> bool:
        """Check if a node is a chance node using the stored flag."""
        return tree.data.is_chance_node[node_idx]

    def _expand_single_child(self,
        key: chex.PRNGKey,
        tree: StochasticMCTSTree,
        parent: int,
        action: int,
        params: chex.ArrayTree,
        decision_step_fn: DecisionStepFn
    ) -> ExpandResult:
        """Expand a single child node using decision_step_fn (for decision nodes)."""
        eval_key = key  # Key only used for NN evaluation, not step

        embedding = jax.tree_util.tree_map(lambda x: x[parent], tree.data.embedding)
        # pgx 3.1.0: step_deterministic doesn't need key
        new_embedding, metadata = decision_step_fn(embedding, action)
        player_reward = metadata.rewards[metadata.cur_player_id]

        value, policy = self.value_policy(new_embedding, params, eval_key, metadata, player_reward)

        node_exists = tree.is_edge(parent, action)
        node_idx = tree.edge_map[parent, action]

        # Get is_stochastic flag from new state's metadata
        is_chance_node = metadata.is_stochastic

        def update_existing_node():
            return StochasticMCTS._update_node_stats(tree, node_idx, value)

        def add_new_node():
            node_data = self.new_stochastic_node(
                policy=policy,
                value=value,
                embedding=new_embedding,
                terminated=metadata.terminated,
                is_chance_node=is_chance_node,
                nn_value_estimate=value,  # Store NN estimate for blended backprop
                outcome_probs=self.stochastic_action_probs
            )
            return tree.add_node(parent_index=parent, edge_index=action, data=node_data)

        updated_tree = jax.lax.cond(node_exists, update_existing_node, add_new_node)
        child_idx = updated_tree.edge_map[parent, action]

        return ExpandResult(tree=updated_tree, child_idx=child_idx, value=value)

    def _expand_single_stochastic_child(self,
        key: chex.PRNGKey,
        tree: StochasticMCTSTree,
        parent: int,
        outcome: int,
        params: chex.ArrayTree,
        stochastic_step_fn: StochasticStepFn
    ) -> ExpandResult:
        """Expand a single child node using stochastic_step_fn (for chance nodes).

        Similar to _expand_single_child but uses stochastic_step_fn.
        The outcome is selected by stochastic_action_selector during traversal.
        """
        eval_key = key

        embedding = jax.tree_util.tree_map(lambda x: x[parent], tree.data.embedding)
        # pgx 3.1.0: step_stochastic doesn't need key
        new_embedding, metadata = stochastic_step_fn(embedding, outcome)
        player_reward = metadata.rewards[metadata.cur_player_id]

        value, policy = self.value_policy(new_embedding, params, eval_key, metadata, player_reward)

        node_exists = tree.is_edge(parent, outcome)
        node_idx = tree.edge_map[parent, outcome]

        # Get is_stochastic flag from new state's metadata
        is_chance_node = metadata.is_stochastic

        def update_existing_node():
            return StochasticMCTS._update_node_stats(tree, node_idx, value)

        def add_new_node():
            node_data = self.new_stochastic_node(
                policy=policy,
                value=value,
                embedding=new_embedding,
                terminated=metadata.terminated,
                is_chance_node=is_chance_node,
                nn_value_estimate=value,
                outcome_probs=self.stochastic_action_probs
            )
            return tree.add_node(parent_index=parent, edge_index=outcome, data=node_data)

        updated_tree = jax.lax.cond(node_exists, update_existing_node, add_new_node)
        child_idx = updated_tree.edge_map[parent, outcome]

        return ExpandResult(tree=updated_tree, child_idx=child_idx, value=value)

    def _expand_stochastic_children_progressive(self,
        key: chex.PRNGKey,
        tree: StochasticMCTSTree,
        parent: int,
        params: chex.ArrayTree,
        stochastic_step_fn: StochasticStepFn
    ) -> ExpandResult:
        """Progressively expand stochastic children based on threshold.

        Progressive expansion: expand child when parent_visits * child_prob > threshold

        This allows efficient handling of stochastic games by:
        - Expanding high-probability outcomes early
        - Deferring low-probability outcomes until parent has enough visits
        - Using blended value (expanded + NN estimate for unexpanded)
        """
        num_stochastic = self.stochastic_size
        parent_visits = tree.data.n[parent]
        parent_embedding = jax.tree_util.tree_map(lambda x: x[parent], tree.data.embedding)
        parent_nn_estimate = tree.data.nn_value_estimate[parent]

        # Determine which outcomes should be expanded based on threshold
        probs = self.stochastic_action_probs
        should_expand = parent_visits * probs > self.progressive_threshold

        # Also check which are already expanded
        child_indices = tree.edge_map[parent, :num_stochastic]
        already_expanded = child_indices != tree.NULL_INDEX

        # Need to expand = should_expand AND NOT already_expanded
        needs_expansion = jnp.logical_and(should_expand, ~already_expanded)

        keys = jax.random.split(key, num_stochastic + 1)
        child_keys = keys[1:]

        def maybe_expand_child(carry, inputs):
            tree_state, parent_emb = carry
            outcome, child_key, should_exp = inputs

            # pgx 3.1.0: step_stochastic doesn't need key
            new_embedding, metadata = stochastic_step_fn(parent_emb, outcome)
            player_reward = metadata.rewards[metadata.cur_player_id]

            value, policy = self.value_policy(new_embedding, params, child_key, metadata, player_reward)

            is_chance_node = metadata.is_stochastic

            node_data = self.new_stochastic_node(
                policy=policy,
                value=value,
                embedding=new_embedding,
                terminated=metadata.terminated,
                is_chance_node=is_chance_node,
                nn_value_estimate=value,
                outcome_probs=self.stochastic_action_probs
            )

            # Add node only if should_exp is True
            tree_with_new = tree_state.add_node(parent_index=parent, edge_index=outcome, data=node_data)

            # Select tree based on should_exp
            new_tree = jax.tree_util.tree_map(
                lambda a, b: jnp.where(should_exp, a, b),
                tree_with_new, tree_state
            )

            return (new_tree, parent_emb), value

        outcomes = jnp.arange(num_stochastic, dtype=jnp.int32)
        (final_tree, _), child_values = jax.lax.scan(
            maybe_expand_child,
            (tree, parent_embedding),
            (outcomes, child_keys, needs_expansion)
        )

        # Update expanded_outcomes mask in parent node
        new_expanded_mask = jnp.logical_or(already_expanded, needs_expansion)
        final_tree = self._update_expanded_outcomes(final_tree, parent, new_expanded_mask)

        # Compute blended value for backprop
        # blended_value = sum(expanded_prob * expanded_q) + (1 - sum_expanded_prob) * nn_estimate
        child_indices_final = final_tree.edge_map[parent, :num_stochastic]
        expanded_mask = child_indices_final != final_tree.NULL_INDEX

        safe_indices = jnp.maximum(child_indices_final, 0)
        child_q_values = final_tree.data.q[safe_indices]
        expanded_q = jnp.where(expanded_mask, child_q_values, 0.0)

        expanded_prob = jnp.where(expanded_mask, probs, 0.0)
        sum_expanded_prob = jnp.sum(expanded_prob)

        # Weighted sum of expanded children
        expanded_value = jnp.sum(expanded_prob * expanded_q)

        # Blended value: expanded contribution + unexpanded contribution (NN estimate)
        blended_value = expanded_value + (1.0 - sum_expanded_prob) * parent_nn_estimate

        # Return first expanded child index (or 0 if none)
        first_expanded_idx = jnp.where(
            jnp.any(expanded_mask),
            safe_indices[jnp.argmax(expanded_mask)],
            0
        )

        return ExpandResult(tree=final_tree, child_idx=first_expanded_idx, value=blended_value)

    @staticmethod
    def _update_expanded_outcomes(tree: StochasticMCTSTree, node_idx: int, expanded_mask: chex.Array) -> StochasticMCTSTree:
        """Update the expanded_outcomes mask for a node."""
        new_data = tree.data.replace(
            expanded_outcomes=tree.data.expanded_outcomes.at[node_idx].set(expanded_mask)
        )
        return tree.replace(data=new_data)


    def traverse(self, key: chex.PRNGKey, tree: StochasticMCTSTree) -> TraversalState:
        """Traverse from root to leaf using appropriate action selector per node type."""
        root_key, body_key = jax.random.split(key)

        def cond_fn(state: TraversalState) -> bool:
            child_idx = tree.edge_map[state.parent, state.action]
            safe_child_idx = jnp.maximum(child_idx, 0)
            child_terminated = tree.data.terminated[safe_child_idx]
            return jnp.logical_and(
                tree.is_edge(state.parent, state.action),
                ~child_terminated
            )

        def body_fn(state: TraversalState) -> TraversalState:
            node_idx = tree.edge_map[state.parent, state.action]
            iteration_key = jax.random.fold_in(body_key, node_idx * tree.branching_factor + state.action)
            action = self.cond_action_selector(iteration_key, tree, node_idx)
            return TraversalState(parent=node_idx, action=action)

        root_action = self.cond_action_selector(root_key, tree, tree.ROOT_INDEX)
        return jax.lax.while_loop(
            cond_fn, body_fn,
            TraversalState(parent=tree.ROOT_INDEX, action=root_action)
        )


    def calculate_discount_factor(self, tree: StochasticMCTSTree, node_idx: int, other_idx: int) -> float:
        """Calculate discount factor between two nodes based on current player."""
        current_player = tree.data.embedding.current_player[node_idx]
        other_player = tree.data.embedding.current_player[other_idx]
        player_diff = jnp.abs(current_player - other_player)
        return 1.0 - 2.0 * player_diff

    def compute_blended_value(self, tree: StochasticMCTSTree, node_idx: int) -> float:
        """Compute blended value at a chance node using progressive expansion.

        Blended value = sum(expanded_prob * expanded_q) + (1 - sum_expanded_prob) * nn_estimate

        This properly handles partially expanded chance nodes by using the
        NN value estimate as a surrogate for unexpanded children.
        """
        child_indices = tree.edge_map[node_idx, :self.stochastic_size]
        expanded_mask = child_indices != tree.NULL_INDEX

        safe_indices = jnp.maximum(child_indices, 0)
        child_q_values = tree.data.q[safe_indices]

        # Weighted value from expanded children
        probs = self.stochastic_action_probs
        expanded_probs = jnp.where(expanded_mask, probs, 0.0)
        sum_expanded_prob = jnp.sum(expanded_probs)

        expanded_q = jnp.where(expanded_mask, child_q_values, 0.0)
        expanded_value = jnp.sum(expanded_probs * expanded_q)

        # Get NN estimate for unexpanded portion
        nn_estimate = tree.data.nn_value_estimate[node_idx]

        # Blended value: expanded contribution + unexpanded contribution
        blended = expanded_value + (1.0 - sum_expanded_prob) * nn_estimate

        return blended

    @staticmethod
    def _update_node_stats(tree: MCTSTree, node_idx: int, value: float) -> MCTSTree:
        """Update only n and q for a node (optimized)."""
        old_n = tree.data.n[node_idx]
        old_q = tree.data.q[node_idx]
        new_n = old_n + 1
        new_q = (old_q * old_n + value) / new_n

        new_data = tree.data.replace(
            n=tree.data.n.at[node_idx].set(new_n),
            q=tree.data.q.at[node_idx].set(new_q)
        )
        return tree.replace(data=new_data)

    def backpropagate(self, key: chex.PRNGKey, tree: StochasticMCTSTree, parent: int, child: int, value: float) -> StochasticMCTSTree:
        """Backpropagate with blended values for chance nodes.

        At chance nodes, we use blended value = expanded_value + (1 - expanded_prob) * nn_estimate
        This properly handles progressive expansion where not all children are expanded.
        """
        compute_blended = self.compute_blended_value
        is_chance = self._is_chance_node
        root_index = tree.ROOT_INDEX

        def body_fn(state: BackpropState) -> BackpropState:
            node_idx, value, tree = state.node_idx, state.value, state.tree
            tree = StochasticMCTS._update_node_stats(tree, node_idx, value)
            parent_idx = tree.parents[node_idx]
            discount = self.calculate_discount_factor(tree, parent_idx, node_idx)

            # At chance nodes, use blended value instead of raw value
            is_stochastic = is_chance(tree, node_idx)
            blended_value = compute_blended(tree, node_idx)
            value_to_propagate = jnp.where(is_stochastic, blended_value, value)
            value_to_propagate *= discount

            return BackpropState(node_idx=parent_idx, value=value_to_propagate, tree=tree)

        initial_discount = self.calculate_discount_factor(tree, parent, child)
        value *= initial_discount
        state = BackpropState(node_idx=parent, value=value, tree=tree)

        state = jax.lax.while_loop(
            lambda s: s.node_idx != root_index,
            body_fn,
            state
        )

        # Handle root node
        is_root_stochastic = is_chance(state.tree, state.node_idx)
        blended_root = compute_blended(state.tree, state.node_idx)
        final_value = jnp.where(is_root_stochastic, blended_root, state.value)
        tree = StochasticMCTS._update_node_stats(state.tree, state.node_idx, final_value)

        return tree

    @staticmethod
    def is_node_stochastic(node: MCTSNode):
        """Check if a node is stochastic."""
        is_stochastic = getattr(node.embedding, '_is_stochastic', None)
        if is_stochastic is None:
            return jnp.array(False)
        return is_stochastic

    @staticmethod
    def is_node_idx_stochastic(tree: MCTSTree, node_idx: int):
        """Check if a node at given index is stochastic."""
        is_stochastic = getattr(tree.data.embedding, '_is_stochastic', None)
        if is_stochastic is None:
            return jnp.array(False)
        return is_stochastic[node_idx]
