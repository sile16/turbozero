import jax
import jax.numpy as jnp
from flax.struct import dataclass


from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.evaluators.mcts.state import MCTSNode, MCTSTree
from core.evaluators.mcts.action_selection import PUCTSelector
from core.trees.tree import init_tree



def dummy_eval_fn(state, params, key):
    """Dummy evaluation function."""
    return jnp.zeros((10,)), jnp.zeros((6,))


@dataclass
class DummyEmbedding:
    """JAX-compatible embedding with _is_stochastic flag for testing."""
    _is_stochastic: bool = False


def test_explore_stochastic_action_selector_manual():
    """Simple manual test to demonstrate the current behavior of explore_stochastic_action_selector.
    
    This test manually shows that the implementation selects actions that are under-visited
    relative to their expected probabilities.
    """
    # Set up parameters
    stochastic_probs = jnp.array([0.2, 0.5, 0.3])
    
    # Create visit counts for children
    child_visits = jnp.array([5, 15, 10], dtype=jnp.int32)  # Total: 30
    
    # Calculate the normalized visit counts
    visit_count_percents = child_visits / jnp.sum(child_visits)
    print(f"Normalized visit counts: {visit_count_percents}")
    print(f"Stochastic probabilities: {stochastic_probs}")
    
    # Calculate deltas (stochastic_probs - normalized_visits)
    # Positive values indicate under-visited nodes
    # Negative values indicate over-visited nodes
    deltas = stochastic_probs - visit_count_percents
    print(f"Deltas (stochastic_probs - normalized_visits): {deltas}")
    
    # Find index of largest positive delta (most under-visited)
    action = jnp.argmax(deltas)
    print(f"Action with largest positive delta (most under-visited): {action}")
    
    # In this example:
    # visit_count_percents = [0.167, 0.5, 0.333]
    # stochastic_probs = [0.2, 0.5, 0.3]
    # deltas = [0.033, 0.0, -0.033]
    # The action should be 0, as it has the highest positive delta, meaning it's the most under-visited
    
    print("\nDemonstration of Corrected Logic:")
    print("We now calculate stochastic_probs - normalized_visits")
    print("Positive values = under-visited (select these)")
    print("Negative values = over-visited")
    print("Zero values = perfectly visited according to their probability")


def test_explore_stochastic_action_selector():
    """Test that explore_stochastic_action_selector correctly selects under-visited children."""
    # Set up a small branching factor for easier testing
    branching_factor = 6
    
    # Set up stochastic probabilities 
    stochastic_probs = jnp.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
    assert jnp.isclose(jnp.sum(stochastic_probs), 1.0), "Probabilities should sum to 1"
    
    # Create a StochasticMCTS instance
    evaluator = StochasticMCTS(
        eval_fn=dummy_eval_fn,
        action_selector=PUCTSelector(),
        branching_factor=branching_factor,
        max_nodes=10,
        num_iterations=10,
        discount=-1.0,
        stochastic_action_probs=stochastic_probs,
        noise_scale=0.0  # Set noise to 0 for deterministic testing
    )
    
    # Create a mock tree
    key = jax.random.PRNGKey(42)
    
    # Create a template embedding
    template_embedding = DummyEmbedding(_is_stochastic=True)
    
    # Initialize a tree
    tree = evaluator.init(template_embedding=template_embedding)
    
    # The tree starts empty, so let's add some nodes
    # First, set the root node
    root_node = MCTSNode(
        n=jnp.array(10, dtype=jnp.int32),  # Root visit count
        p=jnp.zeros((branching_factor,)),  # Policy 
        q=jnp.array(0.5, dtype=jnp.float32),  # Value
        value_probs=jnp.zeros((6,)),
        terminated=jnp.array(False, dtype=jnp.bool_),
        embedding=template_embedding
    )
    tree = tree.set_root(root_node)
    
    # Now add some children to the root with varying visit counts
    # These visit counts are intentionally uneven compared to stochastic_probs
    # Child 0: 10% visits but 10% stochastic prob -> balanced (delta = 0)
    # Child 1: 10% visits but 20% stochastic prob -> under-visited (delta = +0.1)
    # Child 2: 40% visits but 30% stochastic prob -> over-visited (delta = -0.1)
    # Child 3: 10% visits but 20% stochastic prob -> under-visited (delta = +0.1)
    # Child 4: 30% visits but 10% stochastic prob -> over-visited (delta = -0.2)
    # Child 5: 0% visits but 10% stochastic prob -> under-visited (delta = +0.1)
    
    # Create children data
    child_visit_counts = [5, 5, 20, 5, 15, 0]  # Total: 50
    
    # Add children to the tree
    for i in range(branching_factor):
        # Skip the last child to test non-existent edge
        if i < branching_factor - 1:
            child_node = MCTSNode(
                n=jnp.array(child_visit_counts[i], dtype=jnp.int32),
                p=jnp.zeros((branching_factor,)),
                q=jnp.array(0.0, dtype=jnp.float32),
                value_probs=jnp.zeros((6,)),
                terminated=jnp.array(False, dtype=jnp.bool_),
                embedding=DummyEmbedding(_is_stochastic=False)
            )
            tree = tree.add_node(parent_index=tree.ROOT_INDEX, edge_index=i, data=child_node)
    
    # Now call explore_stochastic_action_selector
    action = evaluator.stochastic_action_selector(key, tree, tree.ROOT_INDEX)
    
    # Calculate expected deltas for verification
    child_visit_percents = jnp.array([0.1, 0.1, 0.4, 0.1, 0.3, 0.0])
    expected_deltas = stochastic_probs - child_visit_percents
    print(f"Expected deltas (stochastic_probs - normalized_visits): {expected_deltas}")
    
    # The action with the largest positive delta should be chosen (most under-visited)
    # In this case, it's Child 5 with delta = +0.1 (has 0% visits but 10% probability)
    expected_action = jnp.argmax(expected_deltas)
    print(f"Expected action (most under-visited): {expected_action}")
    
    assert action == expected_action, f"Expected action {expected_action}, got {action}"
    
    
def test_explore_stochastic_action_selector_with_missing_children():
    """Test that explore_stochastic_action_selector correctly handles missing children."""
    # Set up a small branching factor for easier testing
    branching_factor = 6
    
    # Set up stochastic probabilities 
    stochastic_probs = jnp.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
    
    # Create a StochasticMCTS instance
    evaluator = StochasticMCTS(
        eval_fn=dummy_eval_fn,
        action_selector=PUCTSelector(),
        branching_factor=branching_factor,
        max_nodes=10,
        num_iterations=10,
        discount=-1.0,
        stochastic_action_probs=stochastic_probs,
        noise_scale=0.0  # Set noise to 0 for deterministic testing
    )
    
    # Create a mock tree
    key = jax.random.PRNGKey(42)
    
    # Create a template embedding
    template_embedding = DummyEmbedding(_is_stochastic=True)
    
    # Initialize a tree
    tree = evaluator.init(template_embedding=template_embedding)
    
    # Set the root node
    root_node = MCTSNode(
        n=jnp.array(10, dtype=jnp.int32),
        p=jnp.zeros((branching_factor,)),
        q=jnp.array(0.5, dtype=jnp.float32),
        value_probs=jnp.zeros((6,)),
        terminated=jnp.array(False, dtype=jnp.bool_),
        embedding=template_embedding
    )
    tree = tree.set_root(root_node)
    
    # Add only a few children, leaving some missing
    # Child 0: Added with visit count 5
    # Child 1: Missing (should be treated as visit count 0)
    # Child 2: Added with visit count 10
    # Child 3-5: Missing (should be treated as visit count 0)
    
    child_data = [
        (0, 5),   # (edge_index, visit_count)
        (2, 10)   # (edge_index, visit_count)
    ]
    
    for edge_idx, visit_count in child_data:
        child_node = MCTSNode(
            n=jnp.array(visit_count, dtype=jnp.int32),
            p=jnp.zeros((branching_factor,)),
            q=jnp.array(0.0, dtype=jnp.float32),
            value_probs=jnp.zeros((6,)),
            terminated=jnp.array(False, dtype=jnp.bool_),
            embedding=DummyEmbedding(_is_stochastic=False)
        )
        tree = tree.add_node(parent_index=tree.ROOT_INDEX, edge_index=edge_idx, data=child_node)
    
    # Now call explore_stochastic_action_selector
    action = evaluator.stochastic_action_selector(key, tree, tree.ROOT_INDEX)
    
    # Calculate expected visit percentages, treating missing children as 0 visits
    total_visits = 15  # Sum of all visit counts (5 + 0 + 10 + 0 + 0 + 0)
    child_visit_percents = jnp.array([5/total_visits, 0.0, 10/total_visits, 0.0, 0.0, 0.0])
    expected_deltas = stochastic_probs - child_visit_percents
    print(f"Expected deltas (stochastic_probs - normalized_visits): {expected_deltas}")
    
    # The action with the largest positive delta should be chosen
    # Child 1 with 20% probability but 0% visits should be selected
    expected_action = jnp.argmax(expected_deltas)
    print(f"Expected action (most under-visited): {expected_action}")
    
    assert action == expected_action, f"Expected action {expected_action}, got {action}"


def demonstrate_fix():
    """Demonstrate how the fixed version works."""
    # Let's say we have a tree structure like this:
    #
    # Root (idx 0)
    #  |
    #  |-- Child 0 (idx 1) - visits: 5
    #  |-- Child 1 (idx 2) - visits: 15
    #  |-- Child 2 (idx 3) - visits: 10
    #
    # With stochastic_probs = [0.2, 0.5, 0.3]
    
    # Let's manually implement the selector
    
    print("\nDemonstration of Fixed Logic:")
    
    # 1. Define the stochastic probabilities
    stochastic_probs = jnp.array([0.2, 0.5, 0.3])
    
    # 2. Create a scenario with child visit counts
    children_indices = jnp.array([1, 2, 3])  # Indices of child nodes
    children_visits = jnp.array([5, 15, 10])  # Visit counts of child nodes
    
    # 3. Calculate the normalized visit counts
    total_visits = jnp.sum(children_visits)
    children_visit_percents = children_visits / total_visits
    print(f"Child normalized visit counts: {children_visit_percents}")
    print(f"Stochastic probabilities: {stochastic_probs}")
    
    # 4. Calculate deltas (stochastic_probs - normalized_visits)
    deltas = stochastic_probs - children_visit_percents
    print(f"Deltas (stochastic_probs - normalized_visits): {deltas}")
    
    # 5. Find index of largest positive delta (most under-visited)
    action_idx = jnp.argmax(deltas)
    action = action_idx  # In this case, action_idx directly maps to action
    print(f"Action with largest positive delta (most under-visited): {action}")
    
    # The correct implementation should show:
    # Child normalized visit counts = [0.167, 0.5, 0.333]
    # stochastic_probs = [0.2, 0.5, 0.3]
    # deltas = [0.033, 0.0, -0.033]
    # Action should be 0 (since it has delta +0.033, meaning it's under-visited)
    
    print("\nThe fixed implementation now:")
    print("1. Gets the child node indices from tree.edge_map[node_idx]")
    print("2. Gets the visit counts for existing children, using 0 for non-existent children")
    print("3. Normalizes the child visit counts")
    print("4. Calculates stochastic_probs - normalized_visits")
    print("5. Returns the action with the largest positive delta (most under-visited)")


if __name__ == "__main__":
    test_explore_stochastic_action_selector_manual()
    demonstrate_fix() 
