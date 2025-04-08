
from functools import partial
from typing import Dict, Optional, Tuple
import jax
import chex
import jax.numpy as jnp
from core.evaluators.evaluator import Evaluator
from core.evaluators.mcts.action_selection import MCTSActionSelector
from core.evaluators.mcts.state import BackpropState, MCTSNode, MCTSTree, TraversalState, MCTSOutput
from core.trees.tree import init_tree
from core.types import EnvStepFn, EvalFn, StepMetadata

class StochasticMCTS(MCTS):
    """Batched implementation of Monte Carlo Tree Search (MCTS).
    
    Not stateful. This class operates on 'MCTSTree' state objects.
    
    Compatible with `jax.vmap`, `jax.pmap`, `jax.jit`, etc."""
    def __init__(self,
        eval_fn: EvalFn,
        action_selector: MCTSActionSelector,
        stochastic_action_probs: List[float],
        branching_factor: int,
        max_nodes: int,
        num_iterations: int,
        discount: float = -1.0,
        temperature: float = 1.0,
        tiebreak_noise: float = 1e-8,
        persist_tree: bool = True
    ):
        """
        Args:
        - `eval_fn`: leaf node evaluation function (env_state -> (policy_logits, value))
        - `action_selector`: action selection function (eval_state -> action)
        - `branching_factor`: max number of actions (== children per node)
        - `max_nodes`: allocated size of MCTS tree, any additional nodes will not be created, 
                but values from out-of-bounds leaf nodes will still backpropagate
        - `num_iterations`: number of MCTS iterations to perform per evaluate call
        - `discount`: discount factor for MCTS (default: -1.0)
            - use a negative discount in two-player games (e.g. -1.0)
            - use a positive discount in single-player games (e.g. 1.0)
        - `temperature`: temperature for root action selection (default: 1.0)
        - `tiebreak_noise`: magnitude of noise to add to policy weights for breaking ties (default: 1e-8)
        - `persist_tree`: whether to persist search tree state between calls to `evaluate` (default: True)
        """
        super().__init__(eval_fn, action_selector, branching_factor, max_nodes, num_iterations, discount, temperature, tiebreak_noise, persist_tree)

        self.stochastic_action_probs = stochastic_action_probs

    def existing_stochastic_action_selector(self, key: chex.PRNGKey, tree, node_idx, discount):
        """Select an action from the node.
        
        Args:
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from
        """
        choice_key, key = jax.random.split(key)
        return jax.random.choice(self.stochastic_action_probs)
    
    def explore_stochastic_action_selector(self, key: chex.PRNGKey, tree, node_idx, discount):
        # we want a function that will return the biggest delta between the visits counts vs the stochastic action probs

        # get normalized visit count percetnages of all children
        visit_counts = tree.data_at(node_idx).n
        visit_counts = visit_counts / jnp.sum(visit_counts)

        # get the delta between the visit counts and the stochastic action probs
        delta = jnp.abs(visit_counts - self.stochastic_action_probs)

        # add some noise to the delta
        delta = delta + jax.random.normal(key, delta.shape) * self.noise_scale

        # return the action with the biggest delta
        return jnp.argmax(delta)
    
    
    def deterministic_action_selector(self, key: chex.PRNGKey tree, node_idx, discount):
        # wrapper function to call the action selector removing the key argument
        return self.action_selector(tree, node_idx, discount)


    def cond_action_selector(self, key: chex.PRNGKey, tree: MCTSTree, node_idx: int, discount: float) -> int:
        """Select an action from the node, picks the right action selector based on the node type.
        
        Args:
        - `tree`: MCTSTree to evaluate
        - `node_idx`: index of the node to select an action from
        """

        jax.lax.cond(tree.data_at(node_idx).embedding.is_stochastic, 
                     self.stochastic_action_selector, 
                     self.deterministic_action_selector, 
                     tree, node_idx, discount)
        
        

    def iterate(self, key: chex.PRNGKey, tree: MCTSTree, params: chex.ArrayTree, env_step_fn: EnvStepFn) -> MCTSTree:
        """ Performs one iteration of MCTS.
        1. Traverse to leaf node.
        2. Evaluate Leaf Node
        3. Expand Leaf Node (add to tree)
        4. Backpropagate

        Args:
        - `tree`: MCTSTree to evaluate
        - `params`: parameters to pass to the the leaf evaluation function
        - `env_step_fn`: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
        - (MCTSTree): updated MCTSTree
        """
        # traverse from root -> leaf
        traversal_state = self.traverse(key, tree)
        parent, action = traversal_state.parent, traversal_state.action
        # get env state (embedding) for leaf node
        embedding = tree.data_at(parent).embedding
        new_embedding, metadata = env_step_fn(embedding, action)
        player_reward = metadata.rewards[metadata.cur_player_id]
        # evaluate leaf node
        eval_key, key = jax.random.split(key)
        policy_logits, value = self.eval_fn(new_embedding, params, eval_key)
        policy_logits = jnp.where(metadata.action_mask, policy_logits, jnp.finfo(policy_logits).min)
        policy = jax.nn.softmax(policy_logits)
        value = jnp.where(metadata.terminated, player_reward, value)
        # add leaf node to tree
        node_exists = tree.is_edge(parent, action)
        node_idx = tree.edge_map[parent, action]

        node_data = jax.lax.cond(
            node_exists,
            lambda: self.visit_node(node=tree.data_at(node_idx), value=value, p=policy, terminated=metadata.terminated, embedding=new_embedding),
            lambda: self.new_node(policy=policy, value=value, embedding=new_embedding, terminated=metadata.terminated)
        )

        tree = jax.lax.cond(
            node_exists,
            lambda: tree.update_node(index=node_idx, data = node_data),
            lambda: tree.add_node(parent_index=parent, edge_index=action, data=node_data)
        )
        # backpropagate
        return self.backpropagate(key, tree, parent, value)


    def traverse(self, key: chex.PRNGKey, tree: MCTSTree) -> TraversalState:
        """ Traverse from the root node until an unvisited leaf node is reached.
        
        Args:
        - `tree`: MCTSTree to evaluate
        
        Returns:
        - (TraversalState): state of the traversal
            - `parent`: index of the parent node
            - `action`: action to take from the parent node
        """

        # continue while:
        # - there is an existing edge corresponding to the chosen action
        # - AND the child node connected to that edge is not terminal
        def cond_fn(state: TraversalState) -> bool:
            return jnp.logical_and(
                tree.is_edge(state.parent, state.action),
                ~(tree.data_at(tree.edge_map[state.parent, state.action]).terminated)
                # TODO: maximum depth
            )
        
        # each iterration:
        # - get the index of the child node connected to the chosen action
        # - choose the action to take from the child node
        def body_fn(state: TraversalState) -> TraversalState:
            node_idx = tree.edge_map[state.parent, state.action]
            action = self.cond_action_selector(key, tree, node_idx, self.discount)
            return TraversalState(parent=node_idx, action=action)
        
        # choose the action to take from the root
        root_action = self.cond_action_selector(key, tree, tree.ROOT_INDEX, self.discount)
        # traverse from root to leaf
        return jax.lax.while_loop(
            cond_fn, body_fn, 
            TraversalState(parent=tree.ROOT_INDEX, action=root_action)
        )


    def stochastic_evaluate(self, #pylint: disable=arguments-differ
        key: chex.PRNGKey,
        eval_state: MCTSTree, 
        env_state: chex.ArrayTree,
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
        env_step_fn: EnvStepFn,
        **kwargs
    ) -> MCTSOutput:
        """Performs `self.num_iterations` MCTS iterations on an `MCTSTree`.
        Samples an action to take from the root node after search is completed.
        
        Args:
        - `eval_state`: `MCTSTree` to evaluate, could be empty or partially complete

        """

        action = self.stochastic_action_selector(key,eval_state, eval_state.ROOT_INDEX, self.discount)

        return MCTSOutput(
            eval_state=eval_state,
            action=action,
            policy_weights=jnp.zeros((self.branching_factor,))
        )



    def evaluate(self, #pylint: disable=arguments-differ
        key: chex.PRNGKey,
        eval_state: MCTSTree, 
        env_state: chex.ArrayTree,
        root_metadata: StepMetadata,
        params: chex.ArrayTree,
        env_step_fn: EnvStepFn,
        **kwargs
    ) -> MCTSOutput:
        """Performs `self.num_iterations` MCTS iterations on an `MCTSTree`.
        Samples an action to take from the root node after search is completed.
        
        Args:
        - `eval_state`: `MCTSTree` to evaluate, could be empty or partially complete
        - `env_state`: current environment state
        - `root_metadata`: metadata for the root node of the tree
        - `params`: parameters to pass to the the leaf evaluation function
        - `env_step_fn`: env step fn: (env_state, action) -> (new_env_state, metadata)

        Returns:
        - (MCTSOutput): contains new tree state, selected action, root value, and policy weights
        """

       jax.lax.cond(tree.data_at(tree.ROOT_INDEX).is_stochastic, 
                    self.stochastic_evaluate, 
                    super().evaluate, 
                    key, eval_state, env_state, root_metadata, params, env_step_fn)

    def backpropagate(self, key: chex.PRNGKey, tree: MCTSTree, parent: int, value: float) -> MCTSTree: #pylint: disable=unused-argument
        """Backpropagate the value estimate from the leaf node to the root node and update visit counts.
        
        Args:
        - `key`: rng
        - `tree`: MCTSTree to evaluate
        - `parent`: index of the parent node (in most cases, this is the new node added to the tree this iteration)
        - `value`: value estimate of the leaf node

        Returns:
        - (MCTSTree): updated search tree
        """

        def body_fn(state: BackpropState) -> Tuple[int, MCTSTree]:
            node_idx, value, tree = state.node_idx, state.value, state.tree
            # apply discount to value estimate
            value *= self.discount
            node = tree.data_at(node_idx)
            # increment visit count and update value estimate
            new_node = self.visit_node(node, value)
            tree = tree.update_node(node_idx, new_node)
            # go to parent 
            return BackpropState(node_idx=tree.parents[node_idx], value=value, tree=tree)
        
        # backpropagate while the node is a valid node
        # the root has no parent, so the loop will terminate 
        # when the parent of the root is visited
        state = jax.lax.while_loop(
            lambda s: s.node_idx != s.tree.NULL_INDEX, body_fn, 
            BackpropState(node_idx=parent, value=value, tree=tree)
        )
        return state.tree

    def stochastic_new_q_value(self, node: MCTSNode, value: float) -> float:
        """Calculate the new Q value for a node.
        
        Args:
        - `node`: MCTSNode to calculate the new Q value for
        - `value`: value estimate of the leaf node
        """
        # do a normalized weighted sum of the q values of the children
        # get the q values of the children
        q_values = node.q
        # get the visit counts of the children
        visit_counts = node.n
        # normalize the visit counts
        visit_counts = visit_counts / jnp.sum(visit_counts)
        # do a weighted sum of the q values
        return jnp.sum(q_values * visit_counts)

    def deterministic_new_q_value(self, node: MCTSNode, value: float) -> float:
        """Calculate the new Q value for a node.

        Args:
        - `node`: MCTSNode to calculate the new Q value for
        - `value`: value estimate of the leaf node
        """
        return((node.q * node.n) + value) / (node.n + 1)



    @staticmethod
    def visit_node(
        node: MCTSNode,
        value: float,
        p: Optional[chex.Array] = None,
        terminated: Optional[bool] = None,
        embedding: Optional[chex.ArrayTree] = None
    ) -> MCTSNode:
        """ Update the visit counts and value estimate of a node.

        Args:
        - `node`: MCTSNode to update
        - `value`: value estimate to update the node with

        ( we could optionally overwrite the following: )
        - `p`: policy weights to update the node with
        - `terminated`: whether the node is terminal
        - `embedding`: embedding to update the node with

        Returns:
        - (MCTSNode): updated MCTSNode
        """
        # update running value estimate
        q_value = jax.lax.cond(node.is_stochastic, 
                               stochastic_new_q_value, 
                               deterministic_new_q_value, 
                               node, value)
        # update other node attributes
        if p is None:
            p = node.p
        if terminated is None:
            terminated = node.terminated
        if embedding is None:
            embedding = node.embedding
        return node.replace(
            n=node.n + 1, # increment visit count
            q=q_value,
            p=p,
            terminated=terminated,
            embedding=embedding
        )

