
import chex
from chex import dataclass
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class BaseExperience:
    """Experience data structure. Stores a training sample.
    - `reward`: value target for training (discounted return or game outcome)
    - `step_reward`: instant reward received at this step (used to compute discounted returns)
    - `policy_weights`: policy weights
    - `policy_mask`: mask for policy weights (mask out invalid/illegal actions)
    - `observation_nn`: observation for neural network input
    - `cur_player_id`: current player id
    - `is_chance_node`: True if this sample is from a chance node (skip policy loss)
    """
    reward: chex.Array  # Value target (filled at episode end)
    step_reward: chex.Array  # Instant reward at this step
    policy_weights: chex.Array
    policy_mask: chex.Array
    observation_nn: chex.Array
    cur_player_id: chex.Array
    is_chance_node: chex.Array = None  # Optional: True = skip policy loss for this sample


@dataclass(frozen=True)
class ReplayBufferState:
    """State of the replay buffer. Stores objects stored in the buffer
    and metadata used to determine where to store the next object, as well as
    which objects are valid to sample from.
    - `next_idx`: index where the next experience will be stored
    - `episode_start_idx`: index where the current episode started, samples are placed in order
    - `buffer`: buffer of experiences
    - `populated`: mask for populated buffer indices
    - `has_reward`: mask for buffer indices that have been assigned a reward
        - we store samples from in-progress episodes, but don't want to be able to sample them
        until the episode is complete
    - `games_completed_count`: total number of games completed in this buffer
    - `total_completed_game_steps`: total steps across all completed games
    - `total_episode_reward_sum`: sum of all episode rewards (for computing avg)
    - `max_episode_reward`: max single episode reward
    - `min_episode_reward`: min single episode reward
    - `max_episode_length`: max episode length
    - `min_episode_length`: min episode length
    - `original_episode_start_idx`: original episode start (for tracking full episode length)
    """
    next_idx: int
    episode_start_idx: int  # Current segment start (may be updated by bootstrap)
    original_episode_start_idx: int  # True episode start (for length stats)
    buffer: BaseExperience
    populated: chex.Array
    has_reward: chex.Array
    games_completed_count: int
    total_completed_game_steps: int
    total_episode_reward_sum: float = 0.0
    max_episode_reward: float = 0.0
    min_episode_reward: float = 1e9  # Start high so first episode becomes min
    max_episode_length: int = 0
    min_episode_length: int = 1000000  # Start high so first episode becomes min


class EpisodeReplayBuffer:
    """Replay buffer, stores trajectories from episodes for training.

    Compatible with `jax.jit`, `jax.vmap`, and `jax.pmap`."""

    def __init__(self,
        capacity: int,
        discount_factor: float = 0.99,
        reward_scale: float = 1e-4,
    ):
        """
        Args:
        - `capacity`: number of experiences to store in the buffer
        - `discount_factor`: gamma for discounted returns (default: 0.99)
        - `reward_scale`: scale factor to normalize rewards (default: 1e-4 for 2048-scale games)
        """
        self.capacity = capacity
        self.discount_factor = discount_factor
        self.reward_scale = reward_scale


    def get_config(self):
        """Returns the configuration of the replay buffer. Used for logging."""
        return {
            'capacity': self.capacity,
            'discount_factor': self.discount_factor,
            'reward_scale': self.reward_scale,
        }


    def add_experience(self, state: ReplayBufferState, experience: BaseExperience) -> ReplayBufferState:
        """Adds an experience to the replay buffer.
        
        Args:
        - `state`: replay buffer state
        - `experience`: experience to add
        
        Returns:
        - (ReplayBufferState): updated replay buffer state"""
        return state.replace(
            buffer = jax.tree_util.tree_map(
                lambda x, y: x.at[state.next_idx].set(y),
                state.buffer,
                experience
            ),
            next_idx = (state.next_idx + 1) % self.capacity,
            populated = state.populated.at[state.next_idx].set(True),
            has_reward = state.has_reward.at[state.next_idx].set(False)
        )
    

    def assign_rewards(self, state: ReplayBufferState, reward: chex.Array) -> ReplayBufferState:
        """Assign discounted returns to the current episode.

        Computes backwards: V[t] = r[t] + γ * V[t+1]
        where r[t] is the step reward and γ is the discount factor.

        Args:
        - `state`: replay buffer state
        - `reward`: terminal reward (used as bootstrap value, typically 0 for terminal states)

        Returns:
        - (ReplayBufferState): updated replay buffer state with discounted returns
        """
        # Calculate episode length using ORIGINAL start (handle wraparound)
        # This ensures we track full episode length even after bootstrap updates
        episode_length = state.next_idx - state.original_episode_start_idx
        episode_length = jnp.where(
            episode_length < 0,
            episode_length + self.capacity,
            episode_length
        )

        # Compute discounted returns backwards through the episode
        # We need to iterate from the last step to the first
        # V[t] = step_reward[t] + gamma * V[t+1]
        gamma = self.discount_factor
        scale = self.reward_scale

        # Get step rewards for this episode (only unassigned samples)
        # step_reward has shape (capacity, num_players)
        step_rewards = state.buffer.step_reward  # (capacity, num_players)

        # Create mask for current episode samples
        episode_mask = ~state.has_reward  # True for samples in current episode

        # Compute discounted returns using a scan
        # We process indices in reverse order from (next_idx - 1) back to episode_start_idx
        def compute_returns_for_player(player_id):
            """Compute discounted returns for a single player."""
            player_step_rewards = step_rewards[:, player_id]  # (capacity,)

            # Bootstrap value (terminal reward, typically 0)
            terminal_value = reward[player_id] * scale

            # We need to compute returns backwards through the episode
            # Using a scan that processes in reverse order

            # Create index array for the episode (handling wraparound)
            # indices go from episode_start_idx to next_idx - 1
            all_indices = jnp.arange(self.capacity)

            # Compute returns using reverse scan
            def scan_fn(carry, idx):
                """Scan function for backward pass."""
                next_return = carry
                # Get step reward at this index (scaled)
                step_r = player_step_rewards[idx] * scale
                # Check if this index is in the current episode
                is_in_episode = episode_mask[idx]
                # Compute return: r + gamma * V_next
                current_return = step_r + gamma * next_return
                # Only update if in episode, otherwise keep next_return for chaining
                new_return = jnp.where(is_in_episode, current_return, next_return)
                # Output the return for this position
                output_return = jnp.where(is_in_episode, current_return, 0.0)
                return new_return, output_return

            # Scan backwards from next_idx - 1 to 0, then from capacity - 1 to next_idx
            # To handle wraparound properly, we scan the full capacity in reverse

            # Create reverse indices
            reverse_indices = jnp.flip(jnp.arange(self.capacity))

            # But we need to start from next_idx - 1 and go backwards
            # Shift indices so that position 0 in the scan corresponds to next_idx - 1
            shifted_indices = (state.next_idx - 1 - jnp.arange(self.capacity)) % self.capacity

            _, returns = jax.lax.scan(scan_fn, terminal_value, shifted_indices)

            # returns is in reverse order (from newest to oldest), flip back
            returns = jnp.flip(returns)

            # Re-order to match buffer positions
            # shifted_indices tells us which buffer position each scan output corresponds to
            # We need to scatter the returns back to their correct positions
            buffer_returns = jnp.zeros(self.capacity)
            # Use scatter to place returns at correct indices
            flipped_shifted = jnp.flip(shifted_indices)
            buffer_returns = buffer_returns.at[flipped_shifted].set(returns)

            return buffer_returns

        # Compute returns for each player
        num_players = reward.shape[0]
        all_returns = jax.vmap(compute_returns_for_player)(jnp.arange(num_players))  # (num_players, capacity)
        all_returns = all_returns.T  # (capacity, num_players)

        # Update rewards only for current episode samples
        new_rewards = jnp.where(
            episode_mask[..., None],
            all_returns,
            state.buffer.reward
        )

        # Compute episode total reward (sum of step rewards for player 0)
        # For single-player games like 2048, this is the game score
        episode_step_rewards = jnp.where(episode_mask, step_rewards[:, 0], 0.0)
        episode_reward = jnp.sum(episode_step_rewards)

        # Update min/max statistics
        new_max_reward = jnp.maximum(state.max_episode_reward, episode_reward)
        new_min_reward = jnp.minimum(state.min_episode_reward, episode_reward)
        new_max_length = jnp.maximum(state.max_episode_length, episode_length)
        new_min_length = jnp.minimum(state.min_episode_length, episode_length)

        return state.replace(
            episode_start_idx = state.next_idx,
            original_episode_start_idx = state.next_idx,  # Reset for next episode
            has_reward = jnp.full_like(state.has_reward, True),
            games_completed_count = state.games_completed_count + 1,
            total_completed_game_steps = state.total_completed_game_steps + episode_length,
            total_episode_reward_sum = state.total_episode_reward_sum + episode_reward,
            max_episode_reward = new_max_reward,
            min_episode_reward = new_min_reward,
            max_episode_length = new_max_length,
            min_episode_length = new_min_length,
            buffer = state.buffer.replace(reward=new_rewards)
        )
    

    def assign_bootstrap_ongoing(self, state: ReplayBufferState, bootstrap_value: chex.Array) -> ReplayBufferState:
        """Assign discounted returns to ongoing episode samples using bootstrap value.

        Unlike assign_rewards, this does NOT end the episode - it just makes
        current samples trainable while the episode continues. This enables
        n-step returns for long-running games like 2048.

        Args:
        - `state`: replay buffer state
        - `bootstrap_value`: value estimate for current state (per player), used as V(s_T)

        Returns:
        - (ReplayBufferState): updated state with rewards assigned to ongoing samples
        """
        # Only process if there are unassigned samples
        episode_mask = ~state.has_reward  # True for unassigned samples

        # Compute discounted returns same as assign_rewards
        gamma = self.discount_factor
        scale = self.reward_scale
        step_rewards = state.buffer.step_reward

        def compute_returns_for_player(player_id):
            player_step_rewards = step_rewards[:, player_id]
            terminal_value = bootstrap_value[player_id] * scale

            shifted_indices = (state.next_idx - 1 - jnp.arange(self.capacity)) % self.capacity

            def scan_fn(carry, idx):
                next_return = carry
                step_r = player_step_rewards[idx] * scale
                is_in_episode = episode_mask[idx]
                current_return = step_r + gamma * next_return
                new_return = jnp.where(is_in_episode, current_return, next_return)
                output_return = jnp.where(is_in_episode, current_return, 0.0)
                return new_return, output_return

            _, returns = jax.lax.scan(scan_fn, terminal_value, shifted_indices)
            unshifted_returns = jnp.zeros(self.capacity)
            unshifted_returns = unshifted_returns.at[shifted_indices].set(returns)
            return unshifted_returns

        num_players = bootstrap_value.shape[0]
        all_returns = jax.vmap(compute_returns_for_player)(jnp.arange(num_players))
        all_returns = all_returns.T

        new_rewards = jnp.where(
            episode_mask[..., None],
            all_returns,
            state.buffer.reward
        )

        # Mark samples as having rewards, but DON'T reset episode_start_idx
        # The episode continues, we just made these samples trainable
        return state.replace(
            has_reward = jnp.full_like(state.has_reward, True),
            episode_start_idx = state.next_idx,  # New samples start fresh
            buffer = state.buffer.replace(reward=new_rewards)
        )

    def truncate(self,
        state: ReplayBufferState,
    ) -> ReplayBufferState:
        """Truncates the replay buffer, removing all experiences from the current episode.
        Use this if we want to discard all experiences from the current episode.

        Args:
        - `state`: replay buffer state

        Returns:
        - (ReplayBufferState): updated replay buffer state
        """
        # un-assigned trajectory indices have populated set to False
        # so their buffer contents will be overwritten (eventually)
        # and cannot be sampled
        # so there's no need to overwrite them with zeros here
        return state.replace(
            next_idx = state.episode_start_idx,
            original_episode_start_idx = state.episode_start_idx,  # Reset for next episode
            has_reward = jnp.full_like(state.has_reward, True),
            populated = jnp.where(
                ~state.has_reward,
                False,
                state.populated
            )
        )
    
    # assumes input is batched!! (dont vmap/pmap)
    def sample(self,
        state: ReplayBufferState,
        key: jax.random.PRNGKey,
        sample_size: int
    ) -> chex.ArrayTree:
        """Samples experiences from the replay buffer.

        Assumes the buffer has two batch dimensions, so shape = (devices, batch_size, capacity, ...)
        Perhaps there is a dimension-agnostic way to do this?

        Samples across all batch dimensions, not per-batch/device.

        Args:
        - `state`: replay buffer state
        - `key`: rng
        - `sample_size`: size of minibatch to sample

        Returns:
        - (chex.ArrayTree): minibatch of size (sample_size, ...)
        """
        masked_weights = jnp.logical_and(
            state.populated,
            state.has_reward
        ).reshape(-1)

        num_partitions = state.populated.shape[0]
        num_batches = state.populated.shape[1]

        indices = jax.random.choice(
            key,
            self.capacity * num_partitions * num_batches,
            shape=(sample_size,),
            replace=False,
            p = masked_weights / masked_weights.sum()
        )

        partition_indices, batch_indices, item_indices = jnp.unravel_index(
            indices,
            (num_partitions, num_batches, self.capacity)
        )
        
        sampled_buffer_items = jax.tree_util.tree_map(
            lambda x: x[partition_indices, batch_indices, item_indices],
            state.buffer
        )

        return sampled_buffer_items
    
    
    def init(self, batch_size: int, template_experience: BaseExperience) -> ReplayBufferState:
        """Initializes the replay buffer state.
        
        Args:
        - `batch_size`: number of parallel environments
        - `template_experience`: template experience data structure
            - just used to determine the shape of the replay buffer data

        Returns:
        - (ReplayBufferState): initialized replay buffer state
        """
        return ReplayBufferState(
            next_idx = jnp.zeros((batch_size,), dtype=jnp.int32),
            episode_start_idx = jnp.zeros((batch_size,), dtype=jnp.int32),
            original_episode_start_idx = jnp.zeros((batch_size,), dtype=jnp.int32),
            buffer = jax.tree_util.tree_map(
                lambda x: jnp.zeros((batch_size, self.capacity, *x.shape), dtype=x.dtype),
                template_experience
            ),
            populated = jnp.full((batch_size, self.capacity,), fill_value=False, dtype=jnp.bool_),
            has_reward = jnp.full((batch_size, self.capacity,), fill_value=True, dtype=jnp.bool_),
            games_completed_count = jnp.zeros((batch_size,), dtype=jnp.int32),
            total_completed_game_steps = jnp.zeros((batch_size,), dtype=jnp.int32),
            total_episode_reward_sum = jnp.zeros((batch_size,), dtype=jnp.float32),
            max_episode_reward = jnp.zeros((batch_size,), dtype=jnp.float32),
            min_episode_reward = jnp.full((batch_size,), fill_value=1e9, dtype=jnp.float32),
            max_episode_length = jnp.zeros((batch_size,), dtype=jnp.int32),
            min_episode_length = jnp.full((batch_size,), fill_value=1000000, dtype=jnp.int32),
        )
