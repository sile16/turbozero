import chex
from chex import dataclass
import graphviz
import jax
import jax.numpy as jnp

from core.evaluators.evaluator import EvalOutput
from core.trees.tree import Tree


@dataclass(frozen=True)
class MCTSNode:
    """Base MCTS node data strucutre.
    - `n`: visit count
    - `p`: policy vector
    - `value_probs`: distribution over terminal outcomes (e.g., backgammon result buckets)
    - `q`: cumulative value estimate / visit count
    - `terminated`: whether the environment state is terminal
    - `embedding`: environment state
    """
    n: jnp.number
    p: chex.Array
    value_probs: chex.Array
    q: jnp.number
    terminated: jnp.number
    embedding: chex.ArrayTree

    @property
    def w(self) -> jnp.number:
        """cumulative value estimate"""
        return self.q * self.n


# an MCTSTree is a Tree containing MCTSNodes
MCTSTree = Tree[MCTSNode] 


@dataclass(frozen=True)
class TraversalState:
    """State used during traversal step of MCTS.
    - `parent`: parent node index
    - `action`: action taken from parent
    """
    parent: int
    action: int


@dataclass(frozen=True)
class BackpropState:
    """State used during backpropagation step of MCTS.
    - `node_idx`: current node
    - `value`: value to backpropagate
    - `tree`: search tree
    """
    node_idx: int
    value: float
    tree: MCTSTree


@dataclass(frozen=True)
class MCTSOutput(EvalOutput):
    """Output of an MCTS evaluation. See EvalOutput.
    - `eval_state`: The updated internal state of the Evaluator.
    - `policy_weights`: The policy weights assigned to each action.
    """
    eval_state: MCTSTree
    policy_weights: chex.Array


def tree_to_graph(tree, batch_id=0):
    """Converts a search tree to a graphviz graph."""
    graph = graphviz.Digraph()
    
    # Check if it's a StochasticMCTSTree to access stochastic flags
    is_stochastic_tree = hasattr(tree, 'node_is_stochastic')

    def get_child_visits_no_batch(current_tree, index):
        # Adjust access based on potential batch dimension if needed
        # Assuming tree structure might vary, check dimensions. 
        # For simplicity, sticking to the original logic assuming batch_id usage.
        
        # Determine shape based on potential batching
        if current_tree.edge_map.ndim == 3: # Batched tree
            mapping = current_tree.edge_map[batch_id, index]
            # Access data based on batching
            if current_tree.data.n.ndim == 2: # Batched data
                 child_data = current_tree.data.n[batch_id, mapping]
            else: # Unbatched data (shouldn't happen if tree is batched)
                 child_data = current_tree.data.n[mapping]
        else: # Unbatched tree
            mapping = current_tree.edge_map[index]
            child_data = current_tree.data.n[mapping] # Assumes data is unbatched too
        
        # Reshape condition correctly based on child_data's dimensions
        reshape_dims = (-1,) + (1,) * (child_data.ndim - 1)
        return jnp.where(
            (mapping == Tree.NULL_INDEX).reshape(reshape_dims),
            0,
            child_data,
        )

    # Determine the number of nodes based on dimension
    num_nodes = tree.parents.shape[1] if tree.parents.ndim == 2 else tree.parents.shape[0]

    for n_i in range(num_nodes):
        # Adjust data access based on batch dimension
        if tree.data.n.ndim == 2: # Batched data
             node_visits = tree.data.n[batch_id, n_i].item()
             node_q = tree.data.q[batch_id, n_i].item()
             node_terminated = tree.data.terminated[batch_id, n_i].item()
             node_p = tree.data.p[batch_id, n_i]
             node_is_stochastic_flag = tree.node_is_stochastic[batch_id, n_i].item() if is_stochastic_tree and tree.node_is_stochastic.ndim == 2 else False
        else: # Unbatched data
             node_visits = tree.data.n[n_i].item()
             node_q = tree.data.q[n_i].item()
             node_terminated = tree.data.terminated[n_i].item()
             node_p = tree.data.p[n_i]
             node_is_stochastic_flag = tree.node_is_stochastic[n_i].item() if is_stochastic_tree else False
        
        if node_visits > 0:
            # Node attributes and styling
            node_label = f"i: {n_i}\nn: {node_visits}\nq: {node_q:.2f}\nt: {node_terminated}"
            node_attrs = {
                'shape': 'box',
                'style': 'filled',
                'fillcolor': 'lightblue' # Default for deterministic
            }
            
            if is_stochastic_tree and node_is_stochastic_flag:
                node_attrs['fillcolor'] = 'lightcoral' # Color for stochastic
                node_attrs['shape'] = 'ellipse' # Shape for stochastic
            
            graph.node(str(n_i), label=node_label, **node_attrs)

            child_visits = get_child_visits_no_batch(tree, n_i)
            # Determine edge map access based on batching
            mapping = tree.edge_map[batch_id, n_i] if tree.edge_map.ndim == 3 else tree.edge_map[n_i]
            
            for a_i in range(tree.branching_factor):
                v_a = child_visits[a_i].item()
                if v_a > 0:
                    edge_label = f"{a_i}:{node_p[a_i]:.4f}"
                    graph.edge(str(n_i), str(mapping[a_i]), label=edge_label)
        else:
             # Only break if we are reasonably sure we've passed all visited nodes
             # This is tricky without knowing the exact layout, maybe process all nodes?
             # For now, keep original break logic
             break 
             
    return graph
