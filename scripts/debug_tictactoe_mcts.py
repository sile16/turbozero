"""Debug TicTacToe MCTS to find why training isn't working.

Investigates:
1. Are policy targets (improved policy) sensible?
2. Are Q-values being computed correctly?
3. Are visit counts reasonable?
4. Is Gumbel action selection working?
"""

import jax
import jax.numpy as jnp
import pgx

from core.evaluators.mcts import UnifiedMCTS, PUCTSelector
from core.evaluators.mcts.gumbel import compute_improved_policy, sigma_q
from core.networks.azresnet import AZResnet, AZResnetConfig
from core.types import StepMetadata


def make_tictactoe_decision_step_fn(env):
    """Create decision step function for TicTacToe."""
    def step_fn(state, action, key):
        action = jnp.asarray(action, dtype=jnp.int32)
        new_state = env.step(state, action)
        metadata = StepMetadata(
            rewards=new_state.rewards,
            action_mask=new_state.legal_action_mask,
            terminated=new_state.terminated,
            cur_player_id=new_state.current_player,
            step=new_state._step_count,
            is_stochastic=jnp.array(False, dtype=jnp.bool_),
        )
        return new_state, metadata
    return step_fn


def make_nn_eval_fn(network, state_to_nn_input_fn):
    """Create NN evaluation function for MCTS."""
    def eval_fn(state, params, key):
        obs = state_to_nn_input_fn(state)
        obs_batch = jnp.expand_dims(obs, 0)
        policy_logits, value = network.apply(params, obs_batch, train=False)
        return policy_logits[0], value[0]
    return eval_fn


def state_to_nn_input(state):
    """Convert TicTacToe state to NN input."""
    return state.observation


def print_board(state):
    """Print TicTacToe board."""
    # observation shape is (3, 3, 2) - two planes for X and O
    obs = state.observation
    board = []
    for i in range(3):
        row = []
        for j in range(3):
            if obs[i, j, 0] == 1:
                row.append('X')
            elif obs[i, j, 1] == 1:
                row.append('O')
            else:
                row.append('.')
        board.append(' '.join(row))
    print('\n'.join(board))
    print(f"Current player: {state.current_player}")
    print(f"Legal actions: {jnp.where(state.legal_action_mask)[0].tolist()}")


def main():
    print("=" * 60)
    print("Debug TicTacToe MCTS")
    print("=" * 60)

    # Create environment
    env = pgx.make("tic_tac_toe")

    # Create network
    network = AZResnet(AZResnetConfig(
        policy_head_out_size=env.num_actions,
        num_blocks=4,
        num_channels=64,
        value_head_type="default",
    ))

    # Initialize network
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    dummy_state = env.init(init_key)
    dummy_obs = jnp.expand_dims(dummy_state.observation, 0)
    params = network.init(init_key, dummy_obs, train=False)

    # Create MCTS
    decision_step_fn = make_tictactoe_decision_step_fn(env)
    nn_eval_fn = make_nn_eval_fn(network, state_to_nn_input)

    # Critical fix: Strong c_scale for low-sim regime
    # Paper uses c_visit=50, c_scale=1 for Go/Chess with thousands of sims.
    # For TicTacToe with max_visits ≈ 47:
    #   σ(Q) = c_scale * Q / (c_visit + max_N)
    #   c_visit=1, c_scale=20: σ(Q) ≈ 0.42 * Q (strong signal)
    C_VISIT = 1.0
    C_SCALE = 20.0

    mcts = UnifiedMCTS(
        eval_fn=nn_eval_fn,
        action_selector=PUCTSelector(),
        policy_size=env.num_actions,
        max_nodes=200,
        num_iterations=100,
        gumbel_k=9,
        decision_step_fn=decision_step_fn,
        temperature=1.0,
        c_visit=C_VISIT,
        c_scale=C_SCALE,
    )

    # Helper to run MCTS
    def run_mcts(state, params, key):
        # Create metadata
        metadata = StepMetadata(
            rewards=state.rewards,
            action_mask=state.legal_action_mask,
            terminated=state.terminated,
            cur_player_id=state.current_player,
            step=0,
            is_stochastic=jnp.array(False, dtype=jnp.bool_),
        )
        # Initialize eval state
        eval_state = mcts.init(template_embedding=state)
        # Run MCTS
        return mcts.evaluate(key, eval_state, state, metadata, params)

    # Test on initial state
    print("\n" + "=" * 60)
    print("TEST 1: Initial empty board")
    print("=" * 60)

    key, state_key = jax.random.split(key)
    state = env.init(state_key)
    print_board(state)

    # Get raw NN output
    policy_logits, value = nn_eval_fn(state, params, key)
    policy_probs = jax.nn.softmax(policy_logits)

    print(f"\nRaw NN output:")
    print(f"  Value shape: {value.shape}, value: {value}")
    value_scalar = value[0] if value.ndim > 0 else value
    print(f"  Value (scalar): {float(value_scalar):.4f}")
    print(f"  Policy logits: {policy_logits}")
    print(f"  Policy probs: {policy_probs}")
    print(f"  Max prob action: {int(jnp.argmax(policy_probs))} (prob={float(jnp.max(policy_probs)):.4f})")

    # Run MCTS
    key, mcts_key = jax.random.split(key)
    result = run_mcts(state, params, mcts_key)

    # Check eval_state details
    eval_state = result.eval_state
    root_idx = eval_state.ROOT_INDEX

    # Use get_child_data() to get per-action Q-values and visit counts
    root_visits = eval_state.get_child_data('n', root_idx)[:9]  # 9 actions for TicTacToe
    root_q = eval_state.get_child_data('q', root_idx)[:9]
    root_logits = eval_state.data.p[root_idx]  # Prior logits at root

    # Root value is the Q-value of the root node itself
    root_value = eval_state.data.q[root_idx]
    legal_mask = state.legal_action_mask

    print(f"\nMCTS result:")
    print(f"  Root value: {float(root_value):.4f}")
    print(f"  Policy weights (improved): {result.policy_weights}")
    print(f"  Max policy action: {int(jnp.argmax(result.policy_weights))} (weight={float(jnp.max(result.policy_weights)):.4f})")
    print(f"  Selected action: {int(result.action)}")

    print(f"\nMCTS tree details (root node):")
    print(f"  Per-action visit counts: {root_visits}")
    print(f"  Per-action Q-values: {root_q}")

    # Prior logits stored might have extra slot - slice to match
    root_logits_9 = root_logits[:9]
    print(f"  Prior logits (9 actions): {root_logits_9}")

    # Manually compute improved policy
    max_visits = jnp.max(root_visits)
    is_visited = root_visits > 0
    completed_q = jnp.where(is_visited, root_q, root_value)
    scaled_q = sigma_q(completed_q, max_visits, c_visit=C_VISIT, c_scale=C_SCALE)

    print(f"\nImproved policy computation:")
    print(f"  Max visits: {int(max_visits)}")
    print(f"  Visited actions: {jnp.where(is_visited)[0].tolist()}")
    print(f"  Completed Q: {completed_q}")
    print(f"  Scaled Q (sigma) with c_visit={C_VISIT}, c_scale={C_SCALE}: {scaled_q}")

    # Show what σ(Q) would be with different hyperparameters
    scaled_q_v50 = sigma_q(completed_q, max_visits, c_visit=50.0, c_scale=1.0)
    scaled_q_v1 = sigma_q(completed_q, max_visits, c_visit=1.0, c_scale=1.0)
    scaled_q_strong = sigma_q(completed_q, max_visits, c_visit=1.0, c_scale=10.0)
    print(f"  For comparison:")
    print(f"    c_visit=50, c_scale=1 (paper default): σ(Q) = {scaled_q_v50}")
    print(f"    c_visit=1, c_scale=1:  σ(Q) = {scaled_q_v1}")
    print(f"    c_visit=1, c_scale=10: σ(Q) = {scaled_q_strong}")

    improved_logits = root_logits_9 + scaled_q
    improved_logits_masked = jnp.where(legal_mask, improved_logits, -jnp.inf)
    manual_improved_policy = jax.nn.softmax(improved_logits_masked)

    print(f"  Improved logits: {improved_logits}")
    print(f"  Manual improved policy: {manual_improved_policy}")
    print(f"  Returned improved policy: {result.policy_weights}")

    # Check if they match
    diff = jnp.abs(manual_improved_policy - result.policy_weights)
    print(f"  Policy diff (should be ~0): {float(jnp.max(diff)):.6f}")

    # Test 2: After one move
    print("\n" + "=" * 60)
    print("TEST 2: After X plays center (action 4)")
    print("=" * 60)

    state = env.step(state, 4)  # X plays center
    print_board(state)

    # Get raw NN output
    policy_logits, value = nn_eval_fn(state, params, key)
    policy_probs = jax.nn.softmax(policy_logits)

    print(f"\nRaw NN output:")
    value_scalar = value[0] if value.ndim > 0 else value
    print(f"  Value: {float(value_scalar):.4f}")
    print(f"  Policy probs (corners higher?): {policy_probs}")

    # Run MCTS
    key, mcts_key = jax.random.split(key)
    result = run_mcts(state, params, mcts_key)

    # Check Q-values for each action
    eval_state = result.eval_state
    root_idx = eval_state.ROOT_INDEX
    root_visits = eval_state.get_child_data('n', root_idx)[:9]
    root_q = eval_state.get_child_data('q', root_idx)[:9]
    root_value = eval_state.data.q[root_idx]

    print(f"\nMCTS result:")
    print(f"  Root value: {float(root_value):.4f}")
    print(f"  Policy weights: {result.policy_weights}")
    print(f"  Selected action: {int(result.action)}")

    print(f"\nQ-values per action (O's perspective, should prefer corners):")
    for action in range(9):
        if state.legal_action_mask[action]:
            print(f"  Action {action}: visits={int(root_visits[action])}, Q={float(root_q[action]):.4f}")

    # Test 3: Check if MCTS finds winning move
    print("\n" + "=" * 60)
    print("TEST 3: X has winning threat - O must block")
    print("=" * 60)

    # Create position: X has two in a row
    # X . .
    # . X .
    # . . O
    key, state_key = jax.random.split(key)
    state = env.init(state_key)
    state = env.step(state, 0)  # X top-left
    state = env.step(state, 8)  # O bottom-right
    state = env.step(state, 4)  # X center
    # Now O's turn, X threatens 0-4-8 diagonal... wait, that's already done
    # Let me create a better position

    key, state_key = jax.random.split(key)
    state = env.init(state_key)
    state = env.step(state, 0)  # X top-left
    state = env.step(state, 4)  # O center
    state = env.step(state, 1)  # X top-middle
    # X X .
    # . O .
    # . . .
    # X threatens to win with action 2. O must play 2 to block.

    print_board(state)
    print("X threatens to win at position 2. O MUST play 2 to block.")

    # Run MCTS
    key, mcts_key = jax.random.split(key)
    result = run_mcts(state, params, mcts_key)

    print(f"\nMCTS result:")
    print(f"  Policy weights: {result.policy_weights}")
    print(f"  Selected action: {int(result.action)}")
    print(f"  Did MCTS find blocking move (2)? {bool(result.action == 2)}")

    eval_state = result.eval_state
    root_idx = eval_state.ROOT_INDEX
    root_visits = eval_state.get_child_data('n', root_idx)[:9]
    root_q = eval_state.get_child_data('q', root_idx)[:9]

    print(f"\nQ-values (action 2 should be much better for O):")
    for action in range(9):
        if state.legal_action_mask[action]:
            print(f"  Action {action}: visits={int(root_visits[action])}, Q={float(root_q[action]):.4f}, policy_weight={float(result.policy_weights[action]):.4f}")

    # Test 4: Check value learning - who wins?
    print("\n" + "=" * 60)
    print("TEST 4: Value interpretation for two-player game")
    print("=" * 60)

    # Create a position where X is about to win
    key, state_key = jax.random.split(key)
    state = env.init(state_key)
    state = env.step(state, 0)  # X
    state = env.step(state, 3)  # O
    state = env.step(state, 1)  # X
    state = env.step(state, 4)  # O
    # X X .
    # O O .
    # . . .
    # X's turn - X can win with action 2

    print("Position where X can win with action 2:")
    print_board(state)

    policy_logits, value = nn_eval_fn(state, params, key)
    value_scalar = value[0] if value.ndim > 0 else value
    print(f"\nNN value (X's perspective): {float(value_scalar):.4f}")
    print("(Positive should mean X is winning)")

    key, mcts_key = jax.random.split(key)
    result = run_mcts(state, params, mcts_key)

    root_idx = result.eval_state.ROOT_INDEX
    root_q = result.eval_state.get_child_data('q', root_idx)[:9]
    root_visits = result.eval_state.get_child_data('n', root_idx)[:9]
    root_value = result.eval_state.data.q[root_idx]

    print(f"MCTS root value: {float(root_value):.4f}")
    print(f"Selected action: {int(result.action)} (should be 2 to win)")
    print(f"Q-value for action 2: {float(root_q[2]):.4f} (should be ~1.0)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key things to check:
1. Are Q-values for obviously good moves higher?
2. Does improved policy put more weight on better moves?
3. Is the value network predicting correctly (positive = current player winning)?
4. Does MCTS find forcing moves (blocks, wins)?

If MCTS isn't finding good moves with 100 iterations, check:
- PUCT exploration vs exploitation balance
- Whether Q-values are being backed up correctly
- Whether the search is expanding properly
""")


if __name__ == "__main__":
    main()
