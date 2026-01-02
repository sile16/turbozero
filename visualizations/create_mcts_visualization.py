#!/usr/bin/env python
"""
Create visualizations of MCTS trees for PGX Backgammon.
This script generates sample MCTS tree visualizations using graphviz directly,
without relying on the actual MCTS implementation.
"""

import os
import shutil
from pathlib import Path
import json
import graphviz
import numpy as np
import glob
from typing import Any
Array = Any

import jax
import jax.numpy as jnp

import pgx.backgammon as bg
from core.evaluators.mcts.unified_mcts import UnifiedMCTS
from core.evaluators.mcts.action_selection import PUCTSelector
from core.types import StepMetadata
from core.bgcommon import (
    bg_pip_count_eval,
    make_bg_decision_step_fn,
    make_bg_stochastic_step_fn,
)

# Check if graphviz 'dot' command is available
GRAPHVIZ_AVAILABLE = shutil.which('dot') is not None
if not GRAPHVIZ_AVAILABLE:
    print("WARNING: Graphviz 'dot' command not found. Install with: sudo apt-get install graphviz")
    print("         SVG rendering will be skipped, but DOT source files will be saved.")

# Create a sample Backgammon environment for action labels and SVG generation
backgammon_env = bg.Backgammon(short_game=True)
print("Backgammon environment initialized.")

def _is_chance_node(tree, node_idx: int) -> bool:
    if hasattr(tree.data, "is_chance_node"):
        return bool(np.asarray(tree.data.is_chance_node[node_idx]))
    node_data = tree.data_at(node_idx)
    return bool(np.asarray(getattr(node_data, "is_chance_node", False)))


def tree_to_graph(eval_state, output_dir, policy_size, stochastic_size, max_nodes_to_render=50, verbose=False):
    """Convert a tree to a Graphviz graph.

    Args:
        eval_state: The MCTS tree state
        output_dir: Directory for output files
        max_nodes_to_render: Maximum number of nodes to render (for performance)
        verbose: Enable verbose debug output
    """
    tree = eval_state  # The eval_state is the tree itself

    # Debug print to show tree structure
    if verbose:
        print(f"DEBUG - Tree structure: {jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else x, tree)}")
        print(f"DEBUG - Number of nodes: {tree.next_free_idx}")  # For StochasticMCTSTree use next_free_idx
        print(f"DEBUG - Root index: {tree.ROOT_INDEX}")  # For StochasticMCTSTree use ROOT_INDEX

        # Enhanced debugging: Show edge map for first few nodes
        for i in range(min(tree.next_free_idx, 3)):  # Show up to first 3 nodes
            print(f"DEBUG - Node {i} edge map:")
            for j in range(min(tree.branching_factor, 10)):  # Show first 10 edges
                target = tree.edge_map[i, j]
                if target != -1:  # -1 is usually the null value
                    print(f"  Edge {i}--{j}-->{target}")
    
    graph = graphviz.Digraph('MCTS Tree', format='svg')
    graph.attr(rankdir='TD')
    graph.attr('node', shape='box', style='filled', fillcolor='lightblue')
    
    # Add root node
    root_idx = tree.ROOT_INDEX
    root_node_id = str(root_idx)

    # Limit number of nodes to render for performance
    num_nodes_to_render = min(tree.next_free_idx, max_nodes_to_render)
    if tree.next_free_idx > max_nodes_to_render:
        print(f"  [PERF] Rendering {max_nodes_to_render} of {tree.next_free_idx} nodes for performance")

    # Only add nodes that are actually in the tree
    for node_idx in range(num_nodes_to_render):
        node_id = str(node_idx)

        # Get node data
        node_data = tree.data_at(node_idx)

        # Format node label
        q_value = float(node_data.q)
        visit_count = int(node_data.n)

        # Debug for each node (only if verbose)
        if verbose and node_idx < 10:  # Limit to first 10 nodes to avoid spam
            print(f"DEBUG - Node {node_idx}: visits={visit_count}, q_value={q_value:.4f}")
            # Get children info
            children = []
            for a in range(tree.branching_factor):
                child = tree.edge_map[node_idx, a]
                if child != -1:  # Valid edge
                    children.append((a, child))  # Include action with child index
            if children:
                print(f"DEBUG - Children of node {node_idx}: {children}")
            else:
                print(f"DEBUG - Node {node_idx} has no children")
        
        # Create node label
        player_id = node_data.embedding.current_player if hasattr(node_data.embedding, 'current_player') else 'N/A'
        label = f"Node {node_idx}\\nPlayer: {player_id} (0=W, 1=B)\\nVisits: {visit_count}\\nQ: {q_value:.4f}"
        # Add policy visualization (top 8 actions for brevity)
        if hasattr(node_data, 'p') and node_data.p is not None and visit_count > 0: # Only show for visited nodes with policy
            try:
                # Ensure policy is numpy array for sorting
                policy_array = np.array(node_data.p)[:policy_size]
                # Get indices of top k probabilities
                top_k = 8
                top_indices = np.argsort(policy_array)[-top_k:][::-1]
                policy_str_parts = []
                no_op = True
                is_stochastic_node = _is_chance_node(tree, node_idx)
                if not is_stochastic_node:
                    for idx in top_indices:
                        if idx > 5: 
                            no_op = False
                        prob = policy_array[idx]
                        if prob > 0.001:
                            action_label = bg.action_to_str(idx)
                            policy_str_parts.append(f"{action_label}={prob:.2f}")
                else:
                    no_op = False
                if no_op:
                    label += "\\nNo-Op"
                elif policy_str_parts:
                    label += "\\nPolicy: " + ", ".join(policy_str_parts)
                
            except Exception as e:
                label += "\\nPolicy: (Error)" # Indicate if policy parsing failed
        
        # Set node style - include player color for decision nodes
        is_stochastic = _is_chance_node(tree, node_idx)
        if is_stochastic:
            fillcolor = 'lightcoral'  # Stochastic nodes
        elif node_data.terminated:
            fillcolor = 'grey'        # Terminal nodes
        else:
            player_id = getattr(node_data.embedding, 'current_player', None)
            if player_id is not None and int(player_id) == 1:
                fillcolor = 'lightgoldenrod1'  # Player 1
            else:
                fillcolor = 'lightblue'   # Player 0/unknown
        
        graph.node(node_id, label=label, fillcolor=fillcolor)
        
        # Add edges to children (only to nodes we're rendering)
        for action in range(tree.branching_factor):
            child_node_idx = int(tree.edge_map[node_idx, action])
            if child_node_idx != -1 and child_node_idx < num_nodes_to_render:  # Valid edge within render limit
                child_id = str(child_node_idx)
                
                # Format edge label - REFINED LOGIC
                try:
                    is_parent_stochastic = _is_chance_node(tree, node_idx)
                    if is_parent_stochastic:
                        # Parent is stochastic, action selects a stochastic outcome
                        try:
                            outcome = action - policy_size
                            if 0 <= outcome < stochastic_size:
                                action_str = bg.stochastic_action_to_str(outcome)
                            else:
                                action_str = f"StochAction: {action} (Invalid Idx)"
                        except IndexError:
                             action_str = f"StochAction: {action} (Invalid Idx)" # Stochastic index out of range?
                        except Exception as inner_e:
                             action_str = f"StochAction: {action} (Err: {inner_e})"
                    else:
                        # Parent is deterministic, action is a game move
                        try:
                            action_str = bg.action_to_str(action)
                        except IndexError:
                             action_str = f"Action: {action} (Invalid Idx)" # Deterministic index out of range?
                        except Exception as inner_e:
                             action_str = f"Action: {action} (Err: {inner_e})"

                except Exception as e:
                    # Fallback for unexpected errors accessing node_is_stochastic or other issues
                    print(f"WARN: Error determining edge label type for edge from node {node_idx} (action {action}) to {child_node_idx}: {e}")
                    action_str = f"Action: {action} (Type Err)"
                
                # Prepend action number to the string
                edge_label = f"{action}: {action_str}"
                graph.edge(node_id, child_id, label=edge_label)
    
    return graph

def render_graph_safely(graph, output_path_without_ext):
    """Render a graphviz graph to SVG, handling missing graphviz gracefully."""
    if GRAPHVIZ_AVAILABLE:
        graph.render(filename=output_path_without_ext, cleanup=True, view=False)
    else:
        # Save DOT source file instead
        dot_path = output_path_without_ext + '.dot'
        with open(dot_path, 'w') as f:
            f.write(graph.source)
        # Create a placeholder SVG
        svg_path = output_path_without_ext + '.svg'
        placeholder_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="400" height="100">
  <rect width="100%" height="100%" fill="#ffeeee"/>
  <text x="10" y="50" font-family="sans-serif" font-size="14">
    Graphviz not installed. DOT source saved to {os.path.basename(dot_path)}
  </text>
</svg>'''
        with open(svg_path, 'w') as f:
            f.write(placeholder_svg)

def create_real_mcts_visualization(output_dir):
    """Create visualizations of a real UnifiedMCTS tree over multiple iterations and turns."""
    print("Creating UnifiedMCTS visualization...")
    
    # Set up environment
    key = jax.random.PRNGKey(48)
    
    decision_step_fn = make_bg_decision_step_fn(backgammon_env)
    stochastic_step_fn = make_bg_stochastic_step_fn(backgammon_env)

    # Set up MCTS with minimal iterations for visualization
    mcts = UnifiedMCTS(
        eval_fn=bg_pip_count_eval,
        action_selector=PUCTSelector(),
        policy_size=backgammon_env.num_actions,
        max_nodes=200,  # Reduced for faster visualization
        num_iterations=1,  # Use 1 iteration per step for step-by-step visualization
        decision_step_fn=decision_step_fn,
        stochastic_step_fn=stochastic_step_fn,
        stochastic_action_probs=backgammon_env.stochastic_action_probs,
        temperature=0.2,
    )
    
    # Confirm the MCTS configuration
    print(f"MCTS config: max_nodes={mcts.max_nodes}, branching_factor={mcts.branching_factor}")
    
    # Initialize state
    key, init_key = jax.random.split(key)
    state = backgammon_env.init(init_key)

    board: Array = jnp.array([-1, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 12, -12], dtype=jnp.int32)  # type: ignore
    state = state.replace(_board=board)

    
    # Initialize tree and metadata
    eval_state = mcts.init(template_embedding=state)
    metadata = StepMetadata(
        rewards=state.rewards,
        action_mask=state.legal_action_mask,
        terminated=state.terminated,
        cur_player_id=state.current_player,
        step=state._step_count,
        is_stochastic=state._is_stochastic,
    )

    visualization_frames = []
    frame_idx = 0
    total_turns = 5  # Reduced number of turns
    iterations_per_deterministic_turn = 10  # Reduced iterations per turn
    render_every_n_iterations = 2  # Only render every Nth iteration for performance
    verbose = False  # Set to True for detailed debug output

    # Keep track of the latest board arrays for visualization
    last_saved_curr_board = np.array(state._board).tolist()
    last_saved_prev_board = None

    # Simulation loop
    for turn in range(total_turns):
        print(f"\n--- Processing Turn {turn + 1}/{total_turns} ---")
        is_state_stochastic = bool(getattr(state, "_is_stochastic", False))
        print(f"Initial state: Player {state.current_player}, Stochastic: {is_state_stochastic}")

        current_iterations = 0
        # If the state is stochastic, MCTS handles it in one evaluate call
        is_turn_stochastic = is_state_stochastic  # Ensure it's a Python bool

        if is_turn_stochastic:
            key, eval_key, step_key = jax.random.split(key, 3)

            # Call MCTS.evaluate on the stochastic state
            # For stochastic state, this will sample a stochastic action but not modify the tree
            mcts_output = mcts.evaluate(
                key=eval_key,
                eval_state=eval_state,
                env_state=state,
                root_metadata=metadata,
                params={},
            )
            # Important: For stochastic nodes, the tree stays the same
            eval_state = mcts_output.eval_state
            action = mcts_output.action
            print(f"  Turn {turn+1}: Stochastic action selected: {action} ({bg.stochastic_action_to_str(action)})")
            
            # Add selected stochastic action to visualization
            graph = tree_to_graph(
                eval_state,
                output_dir,
                policy_size=mcts.policy_size,
                stochastic_size=mcts.stochastic_size,
                verbose=verbose,
            )
            
            # Visualize MCTS tree for the stochastic root
            graph_filename = f'real_tree_turn_{turn}_stochastic_eval.svg'
            graph_path_abs = os.path.join(output_dir, graph_filename)
            graph_path_rel = graph_filename # Relative path for HTML
            render_graph_safely(graph, graph_path_abs.replace('.svg', ''))
            
            # Store frame for stochastic evaluation
            visualization_frames.append({
                'type': 'stochastic_eval',
                'turn': turn,
                'graph_path': graph_path_rel,
                'info': f'Turn {turn+1}: Evaluating stochastic state (Player {state.current_player})',
                'current_board': last_saved_curr_board,
                'previous_board': last_saved_prev_board,
                'action_str': '' # No action yet
            })
            frame_idx += 1
            
            # --- Action Step ---
            action_str = bg.stochastic_action_to_str(action)
            previous_state = state
            last_saved_prev_board = last_saved_curr_board  # Current becomes previous

            # Step environment with the chosen stochastic action
            state, metadata = stochastic_step_fn(state, action, step_key)
            # Step MCTS tree - always step after environment steps with stochastic actions
            tree_action = mcts.policy_size + int(action)
            eval_state = mcts.step(eval_state, tree_action)

            # Save current board state
            last_saved_curr_board = np.array(state._board).tolist()

            # Visualize MCTS tree *after* the stochastic step
            graph_after_step = tree_to_graph(
                eval_state,
                output_dir,
                policy_size=mcts.policy_size,
                stochastic_size=mcts.stochastic_size,
                verbose=verbose,
            )
            graph_after_step_filename = f'real_tree_turn_{turn}_stochastic_after.svg'
            graph_after_step_path_abs = os.path.join(output_dir, graph_after_step_filename)
            graph_after_step_path_rel = graph_after_step_filename
            render_graph_safely(graph_after_step, graph_after_step_path_abs.replace('.svg', ''))

            # Store frame for the action step
            visualization_frames.append({
                'type': 'action',
                'turn': turn,
                'action': int(action), # Ensure action is serializable
                'action_str': action_str,
                'graph_path': graph_after_step_path_rel,
                'previous_board': last_saved_prev_board,
                'current_board': last_saved_curr_board,
                'info': f'Turn {turn+1}: Applied stochastic action {action_str} (Player {previous_state.current_player}) -> Player {state.current_player}'
            })
            frame_idx += 1

        else:
            if verbose:
                print(f"DEBUG - Processing deterministic state at turn {turn}")
                print(f"DEBUG - Before MCTS evaluate - Turn {turn}, steps {current_iterations}")
                print(f"DEBUG - eval_state tree node count: {eval_state.next_free_idx}")

            # Call MCTS.evaluate on the current state to get the action for visualization
            key, eval_key = jax.random.split(key)
            final_action = -1  # Placeholder

            for iteration in range(iterations_per_deterministic_turn):
                key, eval_key = jax.random.split(key)

                mcts_output = mcts.evaluate(
                    key=eval_key,
                    eval_state=eval_state,  # Current deterministic state
                    env_state=state,
                    root_metadata=metadata,
                    params={},
                )
                eval_state = mcts_output.eval_state
                current_iter_action = mcts_output.action  # Action MCTS would choose *if stopped now*
                final_action = current_iter_action  # Keep track of the best action found

                # DEBUG output (only if verbose)
                if verbose:
                    print(f"    DEBUG: Iter {iteration+1}: Action = {current_iter_action} ({bg.action_to_str(current_iter_action)})", flush=True)
                    if current_iter_action == 0:
                        is_only_legal = (state.legal_action_mask[0] == 1) and (jnp.sum(state.legal_action_mask) == 1)
                        print(f"      DEBUG: No-op (0) considered. Only legal? {is_only_legal}")
                    if eval_state.next_free_idx > 1:
                        try:
                            node1_data = eval_state.data_at(1)
                            node1_policy = np.array(node1_data.p)
                            print(f"      Node 1 Policy (Top 5): {np.argsort(node1_policy)[-5:][::-1]}")
                        except Exception as e:
                            print(f"      DEBUG: Error inspecting Node 1: {e}")
                    print(f"    DEBUG: After iteration {iteration+1}, tree size: {eval_state.next_free_idx}")

                # Only render every Nth iteration for performance
                should_render = (iteration + 1) % render_every_n_iterations == 0 or iteration == iterations_per_deterministic_turn - 1
                if should_render:
                    # Visualize tree after this iteration
                    graph = tree_to_graph(
                        eval_state,
                        output_dir,
                        policy_size=mcts.policy_size,
                        stochastic_size=mcts.stochastic_size,
                        verbose=verbose,
                    )
                    graph_filename = f'real_tree_turn_{turn}_iter_{iteration:02d}.svg'
                    graph_path_abs = os.path.join(output_dir, graph_filename)
                    graph_path_rel = graph_filename
                    render_graph_safely(graph, graph_path_abs.replace('.svg', ''))

                    # Store iteration frame
                    visualization_frames.append({
                        'type': 'iteration',
                        'turn': turn,
                        'iteration': iteration,
                        'graph_path': graph_path_rel,
                        'info': f'Turn {turn+1}, Iteration {iteration+1}/{iterations_per_deterministic_turn} (Player {state.current_player})',
                        'current_board': last_saved_curr_board,
                        'previous_board': last_saved_prev_board,
                        'action_str': ''  # No action during iteration
                    })
                    frame_idx += 1

                current_iterations += 1

            # --- Action Step ---
            action = final_action  # Use action determined by MCTS after all iterations
            action_str = bg.action_to_str(action)
            print(f"  Turn {turn+1}: Applying deterministic action: {action} ({action_str})")
            previous_state = state
            last_saved_prev_board = last_saved_curr_board # Current becomes previous
            key, step_key = jax.random.split(key)

            # Step environment based on the chosen action 
            state, metadata = decision_step_fn(state, action, step_key)

            # Step MCTS tree - always step after a deterministic action
            eval_state = mcts.step(eval_state, action)

            # Save current board state
            last_saved_curr_board = np.array(state._board).tolist()

            # Visualize MCTS tree *after* the step
            graph_after_step = tree_to_graph(
                eval_state,
                output_dir,
                policy_size=mcts.policy_size,
                stochastic_size=mcts.stochastic_size,
                verbose=verbose,
            )
            graph_after_step_filename = f'real_tree_turn_{turn}_deterministic_after.svg'
            graph_after_step_path_abs = os.path.join(output_dir, graph_after_step_filename)
            graph_after_step_path_rel = graph_after_step_filename
            render_graph_safely(graph_after_step, graph_after_step_path_abs.replace('.svg', ''))

            # Store action frame
            visualization_frames.append({
                'type': 'action',
                'turn': turn,
                'action': int(action), # Ensure action is serializable
                'action_str': action_str,
                'graph_path': graph_after_step_path_rel,
                'previous_board': last_saved_prev_board,
                'current_board': last_saved_curr_board,
                'info': f'Turn {turn+1}: Applied move {action_str} (Player {previous_state.current_player}) -> Player {state.current_player}'
            })
            frame_idx += 1

        # Check for game termination
        if bool(state.terminated):
             print(f"--- Game terminated at Turn {turn + 1} ---")
             # Add a final frame indicating termination
             visualization_frames.append({
                'type': 'terminal',
                'turn': turn,
                'graph_path': graph_after_step_path_rel, # Show last tree state
                'previous_board': last_saved_prev_board, # Show board before final state
                'current_board': last_saved_curr_board, # Show final board state
                'info': f'Game Over at Turn {turn+1}. Final Reward: {state.rewards[0]:.2f}',
                'action_str': 'Game Over'
             })
             frame_idx += 1
             break # End simulation

    print(f"\nSimulation complete. Generated {frame_idx} visualization frames.")
    # Don't create GIF here, let the HTML handle SVG display
    # create_gif_from_svgs(output_dir, "real_tree") 
    return visualization_frames

def create_single_page_visualization(output_dir, visualization_frames):
    """Create a single HTML page for viewing all visualizations sequentially."""
    print("Creating single page HTML visualization...")
    
    html_path = os.path.join(output_dir, "mcts_visualization.html")
    
    # Prepare data for JavaScript
    frames_json = json.dumps(visualization_frames)
    
    with open(html_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UnifiedMCTS Visualization</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1800px; /* Wider layout */
            margin: 20px auto;
            padding: 20px;
            background-color: #f0f2f5;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        h1 {{
            color: #1a237e; /* Dark blue */
            text-align: center;
            margin-bottom: 20px;
        }}
        .board-display-container {{
            display: flex;
            justify-content: space-between; /* Space out boards and action */
            align-items: center; /* Vertically align items */
            width: 95%; /* Use most of the width */
            margin-bottom: 20px;
            padding: 10px;
            background: #e8eaf6; /* Light background for contrast */
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .board-container {{
            flex: 0 0 35%; /* Assign fixed width percentage */
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
            padding: 10px;
            text-align: center;
            min-height: 250px; /* Ensure minimum height */
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center; /* Center content vertically */
        }}
        .board-svg {{
            width: 100%;
            height: auto;
            max-height: 260px;
        }}
        .board-container img {{
            max-width: 100%;
            height: auto;
            max-height: 250px; /* Max height for board SVGs */
            display: block;
            margin-bottom: 5px;
            border: 1px solid #ddd;
        }}
         .board-container h3 {{
            margin-top: 0;
            margin-bottom: 8px;
            color: #3f51b5;
            font-size: 1.1em;
        }}
        .action-display {{
            flex: 0 0 20%; /* Width for the action display */
            text-align: center;
            font-size: 1.5em; /* Larger arrow */
            color: #555;
        }}
        .action-text {{
            font-size: 0.7em; /* Smaller text for action */
            font-weight: bold;
            display: block; /* Place text below arrow */
            margin-top: 5px;
            color: #1a237e;
        }}
        .mcts-graph-container {{
             width: 95%;
             margin-top: 20px; /* Space below controls */
             background: white;
             border-radius: 8px;
             box-shadow: 0 5px 15px rgba(0,0,0,0.1);
             padding: 15px;
             text-align: center;
        }}
        .mcts-graph-container h2 {{
             margin-top: 0;
             color: #1a237e;
        }}
        .mcts-graph-container img {{
            max-width: 100%;
            height: auto; /* Maintain aspect ratio */
            max-height: 65vh; /* Limit height */
            display: block;
            margin: 10px auto;
            border: 1px solid #eee;
        }}
        .controls-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            padding: 15px 20px;
            margin-top: 0; /* Remove top margin, comes after boards */
            width: 95%; /* Match other container widths */
            text-align: center;
        }}
        .controls {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
            flex-wrap: wrap; /* Allow controls to wrap */
        }}
        .btn {{
            background: #3f51b5; /* Indigo */
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1em;
            transition: background-color 0.3s ease;
        }}
        .btn:hover {{
            background: #303f9f; /* Darker Indigo */
        }}
        .btn:disabled {{
            background: #9fa8da; /* Lighter disabled */
            cursor: not-allowed;
        }}
        .slider-container {{
            flex-grow: 1;
            margin: 0 15px;
            min-width: 300px; /* Ensure slider is usable */
            max-width: 60%;
        }}
        .slider {{
            width: 100%;
            cursor: pointer;
            height: 8px;
            background: #ddd;
            border-radius: 4px;
            appearance: none; /* Override default look */
        }}
        .slider::-webkit-slider-thumb {{
            appearance: none;
            width: 18px;
            height: 18px;
            background: #3f51b5;
            border-radius: 50%;
            cursor: pointer;
        }}
        .slider::-moz-range-thumb {{
            width: 18px;
            height: 18px;
            background: #3f51b5;
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }}
        .info-panel {{
            margin-bottom: 15px; /* Space above controls */
            font-size: 1.1em;
            color: #444;
            min-height: 25px; /* Reserve space */
            background-color: #e8eaf6; /* Light indigo background */
            padding: 8px 15px;
            border-radius: 4px;
        }}
        .frame-counter {{
            font-weight: bold;
            min-width: 100px; /* Ensure counter width */
            text-align: right;
            color: #555;
        }}
        .mcts-legend {{
            width: 95%;
            margin-top: 25px;
            padding: 15px;
            background: #f5f5f5; /* Lighter background */
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            font-size: 0.9em;
        }}
        .mcts-legend h3 {{
            margin-top: 0;
            color: #1a237e;
            border-bottom: 1px solid #ccc;
            padding-bottom: 5px;
        }}
        .mcts-legend ul {{
            list-style: none;
            padding: 0;
        }}
        .mcts-legend li {{
            margin-bottom: 8px;
            display: flex;
            align-items: center;
        }}
        .legend-color-box {{
            display: inline-block;
            width: 18px;
            height: 18px;
            margin-right: 10px;
            border: 1px solid #999;
            border-radius: 3px;
        }}
        .legend-term {{
             font-weight: bold;
             min-width: 25px;
             display: inline-block;
             margin-right: 5px;
        }}

        /* Simple loader */
        .loader {{
            border: 4px solid #f3f3f3;
            border-radius: 50%;
            border-top: 4px solid #3f51b5;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
            margin: 10px auto;
            display: none; /* Initially hidden */
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <h1>UnifiedMCTS Visualization</h1>

    <div class="board-display-container">
        <div id="currBoardContainer" class="board-container">
            <h3>Current Board State</h3>
            <div id="currBoard" class="board-svg"></div>
        </div>
        <div class="action-display">
            <span id="actionArrow">&#11013;</span> <!-- Left arrow -->
            <span id="actionText" class="action-text"></span>
        </div>
        <div id="prevBoardContainer" class="board-container">
            <h3>Previous Board State</h3>
            <div id="prevBoard" class="board-svg"></div>
        </div>
    </div>

    <div class="controls-container">
        <div id="infoPanel" class="info-panel">Loading visualization...</div>
        <div class="controls">
            <button id="firstBtn" class="btn" title="Go to First Frame">First</button>
            <button id="prevBtn" class="btn" title="Previous Frame">Prev</button>
            <div class="slider-container">
                <input type="range" min="0" max="0" value="0" class="slider" id="frameSlider" title="Navigate Frames">
            </div>
            <button id="nextBtn" class="btn" title="Next Frame">Next</button>
            <button id="lastBtn" class="btn" title="Go to Last Frame">Last</button>
            <span id="frameCounter" class="frame-counter">0 / 0</span>
        </div>
    </div>

    <div class="mcts-graph-container">
        <h2>MCTS Tree</h2>
        <div id="mctsLoader" class="loader"></div>
        <img id="mctsImage" src="" alt="MCTS Tree Visualization">
    </div>

    <div class="mcts-legend">
        <h3>MCTS Graph Legend</h3>
        <ul>
            <li><span class="legend-color-box" style="background-color: lightgreen;"></span> Node: Root</li>
            <li><span class="legend-color-box" style="background-color: lightblue;"></span> Node: Standard Deterministic</li>
            <li><span class="legend-color-box" style="background-color: lightcoral;"></span> Node: Stochastic (Dice Roll)</li>
            <li><span class="legend-color-box" style="background-color: grey;"></span> Node: Terminal State</li>
        </ul>
        <ul>
            <li><span class="legend-term">i:</span> Node index</li>
            <li><span class="legend-term">n:</span> Visit count</li>
            <li><span class="legend-term">q:</span> Value (expected reward from this node)</li>
            <li><span class="legend-term">t:</span> Terminated flag (is this node a terminal game state?)</li>
            <li><span class="legend-term">p:</span> Policy prior (initial probability of selecting edge)</li>
        </ul>
    </div>
    
    <script>
        const frames = {frames_json};
        let currentIndex = 0;
        
        // DOM elements
        const mctsImage = document.getElementById('mctsImage');
        const prevBoardContainer = document.getElementById('prevBoard');
        const currBoardContainer = document.getElementById('currBoard');
        const actionTextElement = document.getElementById('actionText');
        const slider = document.getElementById('frameSlider');
        const counter = document.getElementById('frameCounter');
        const infoPanel = document.getElementById('infoPanel');
        const firstBtn = document.getElementById('firstBtn');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const lastBtn = document.getElementById('lastBtn');
        
        // Loaders
        const mctsLoader = document.getElementById('mctsLoader');
        function showLoader(loaderElement) {{
            loaderElement.style.display = 'block';
        }}

        function hideLoader(loaderElement) {{
            requestAnimationFrame(() => {{
                loaderElement.style.display = 'none';
            }});
        }}

        function renderBoard(container, board) {{
            container.innerHTML = '';
            if (!board || board.length < 28) {{
                container.textContent = 'No board data';
                return;
            }}

            const width = 520;
            const height = 320;
            const barWidth = 40;
            const margin = 16;
            const midY = height / 2;
            const pointWidth = (width - barWidth) / 12;
            const triangleHeight = (height - 2 * margin) / 2 - 6;
            const checkerRadius = Math.max(6, Math.floor(pointWidth * 0.35));
            const maxVisible = 5;

            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
            svg.setAttribute('width', '100%');
            svg.setAttribute('height', '100%');

            const addRect = (x, y, w, h, fill) => {{
                const rect = document.createElementNS(svg.namespaceURI, 'rect');
                rect.setAttribute('x', x);
                rect.setAttribute('y', y);
                rect.setAttribute('width', w);
                rect.setAttribute('height', h);
                rect.setAttribute('fill', fill);
                svg.appendChild(rect);
            }};

            const addText = (x, y, text, color, size = 12, anchor = 'middle') => {{
                const t = document.createElementNS(svg.namespaceURI, 'text');
                t.setAttribute('x', x);
                t.setAttribute('y', y);
                t.setAttribute('fill', color);
                t.setAttribute('font-size', size);
                t.setAttribute('font-family', 'sans-serif');
                t.setAttribute('text-anchor', anchor);
                t.textContent = text;
                svg.appendChild(t);
            }};

            addRect(0, 0, width, height, '#f2d6a2');
            addRect(0, 0, width, margin, '#d4b17f');
            addRect(0, height - margin, width, margin, '#d4b17f');
            addRect(6 * pointWidth, 0, barWidth, height, '#c08c5a');

            const pointStr = (x, y) => String(x) + ',' + String(y);
            for (let col = 0; col < 12; col++) {{
                const xBase = col < 6 ? col * pointWidth : barWidth + col * pointWidth;
                const color = col % 2 === 0 ? '#d18b47' : '#f4e1bf';

                const top = document.createElementNS(svg.namespaceURI, 'polygon');
                top.setAttribute('points', [
                    pointStr(xBase, margin),
                    pointStr(xBase + pointWidth, margin),
                    pointStr(xBase + pointWidth / 2, midY - 6)
                ].join(' '));
                top.setAttribute('fill', color);
                svg.appendChild(top);

                const bottom = document.createElementNS(svg.namespaceURI, 'polygon');
                bottom.setAttribute('points', [
                    pointStr(xBase, height - margin),
                    pointStr(xBase + pointWidth, height - margin),
                    pointStr(xBase + pointWidth / 2, midY + 6)
                ].join(' '));
                bottom.setAttribute('fill', color);
                svg.appendChild(bottom);
            }}

            const drawStack = (idx, isTop) => {{
                const count = board[idx];
                if (!count) return;
                const absCount = Math.abs(count);
                const color = count > 0 ? '#fdfdfd' : '#2c2c2c';
                const stroke = count > 0 ? '#666' : '#111';
                const visible = Math.min(absCount, maxVisible);
                const xCenter = (idx % 12) < 6
                    ? (idx % 12) * pointWidth + pointWidth / 2
                    : barWidth + (idx % 12) * pointWidth + pointWidth / 2;

                for (let i = 0; i < visible; i++) {{
                    const circle = document.createElementNS(svg.namespaceURI, 'circle');
                    const y = isTop
                        ? margin + checkerRadius + i * (checkerRadius * 2 + 2)
                        : height - margin - checkerRadius - i * (checkerRadius * 2 + 2);
                    circle.setAttribute('cx', xCenter);
                    circle.setAttribute('cy', y);
                    circle.setAttribute('r', checkerRadius);
                    circle.setAttribute('fill', color);
                    circle.setAttribute('stroke', stroke);
                    circle.setAttribute('stroke-width', 1);
                    svg.appendChild(circle);
                }}

                if (absCount > maxVisible) {{
                    const textY = isTop
                        ? margin + checkerRadius + (visible - 1) * (checkerRadius * 2 + 2) + 4
                        : height - margin - checkerRadius - (visible - 1) * (checkerRadius * 2 + 2) - 4;
                    addText(xCenter, textY, String(absCount), count > 0 ? '#111' : '#f5f5f5', 11);
                }}
            }};

            for (let i = 0; i < 12; i++) {{
                drawStack(i, true);
            }}
            for (let i = 12; i < 24; i++) {{
                drawStack(i, false);
            }}

            const barCur = Math.max(board[24], 0);
            const barOpp = Math.max(-board[25], 0);
            const offCur = board[26];
            const offOpp = Math.max(-board[27], 0);
            addText(width / 2, margin + 12, 'Bar ' + barCur + '/' + barOpp, '#3b2f1c', 12);
            addText(width / 2, height - margin - 6, 'Off ' + offCur + '/' + offOpp, '#3b2f1c', 12);

            container.appendChild(svg);
        }}

        // Update the display based on the current frame index
        function updateDisplay() {{
            if (frames.length === 0) {{
                infoPanel.textContent = "No visualization frames found.";
                return;
            }}
            
            const frame = frames[currentIndex];
            
            // --- Update Info Panel ---
            infoPanel.textContent = frame.info;

            // --- Update MCTS Graph ---
            showLoader(mctsLoader);
            mctsImage.onload = () => hideLoader(mctsLoader);
            mctsImage.onerror = () => {{
                 hideLoader(mctsLoader);
                 mctsImage.alt = "Error loading MCTS graph";
                 mctsImage.src = ""; // Clear src on error
                 console.error("Error loading MCTS image:", frame.graph_path);
             }};
            // Check if graph_path exists before setting src
            mctsImage.src = frame.graph_path ? frame.graph_path : "";
            mctsImage.alt = frame.graph_path ? `MCTS Tree - ${{frame.info}}` : "MCTS Graph Unavailable";

            // --- Update Board Displays ---
            renderBoard(currBoardContainer, frame.current_board);
            renderBoard(prevBoardContainer, frame.previous_board);

            // --- Update Action Display ---
            if (frame.type === 'action' || frame.type === 'terminal') {{
                 actionTextElement.textContent = frame.action_str || 'N/A';
            }} else if (frame.type === 'stochastic_eval') {{
                 actionTextElement.textContent = 'Evaluating...';
            }} else if (frame.type === 'iteration') {{
                 actionTextElement.textContent = `Iteration ${{frame.iteration + 1}}`;
            }} else {{
                 actionTextElement.textContent = ''; // Clear for unknown types or start
            }}

            // --- Update Slider and Counter ---
            slider.value = currentIndex;
            counter.textContent = `${{currentIndex + 1}} / ${{frames.length}}`;
            
            // --- Update Button States ---
            firstBtn.disabled = currentIndex === 0;
            prevBtn.disabled = currentIndex === 0;
            nextBtn.disabled = currentIndex === frames.length - 1;
            lastBtn.disabled = currentIndex === frames.length - 1;
        }}
        
        // --- Event Listeners ---
        slider.addEventListener('input', () => {{
            currentIndex = parseInt(slider.value);
            updateDisplay();
        }});
        
        firstBtn.addEventListener('click', () => {{
            currentIndex = 0;
            updateDisplay();
        }});
        
        prevBtn.addEventListener('click', () => {{
            if (currentIndex > 0) {{
                currentIndex--;
                updateDisplay();
            }}
        }});
        
        nextBtn.addEventListener('click', () => {{
            if (currentIndex < frames.length - 1) {{
                currentIndex++;
                updateDisplay();
            }}
        }});
        
        lastBtn.addEventListener('click', () => {{
            currentIndex = frames.length - 1;
            updateDisplay();
        }});
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.target === slider) return; // Don't interfere with slider input

            if (e.key === 'ArrowLeft') {{
                prevBtn.click(); // Trigger button click logic
            }} else if (e.key === 'ArrowRight') {{
                nextBtn.click(); // Trigger button click logic
            }} else if (e.key === 'Home') {{
                 e.preventDefault();
                 firstBtn.click();
            }} else if (e.key === 'End') {{
                 e.preventDefault();
                 lastBtn.click();
            }}
        }});
        
        // --- Initialization ---
        if (frames.length > 0) {{
            slider.max = frames.length - 1;
            updateDisplay(); // Initial display
        }} else {{
            infoPanel.textContent = "No visualization frames generated.";
            counter.textContent = "0 / 0";
            // Disable all controls if no frames
            [firstBtn, prevBtn, nextBtn, lastBtn, slider].forEach(el => el.disabled = true);
        }}
    </script>
</body>
</html>
""")

    print(f"Created HTML visualization: {html_path}")
    return html_path

def main():
    """Generate MCTS visualizations and create a single HTML page."""
    # No need for argparse for now, but keep structure if needed later
    # parser = argparse.ArgumentParser(description='Create visualizations of MCTS trees')
    # args = parser.parse_args()
    
    # Create output directory
    output_dir = "visualizations/mcts_trees"
    os.makedirs(output_dir, exist_ok=True)
    # Clean up old SVGs before generation to avoid confusion
    print(f"Cleaning old SVGs from {output_dir}...")
    for file_pattern in ["*.svg"]:
        files_to_remove = glob.glob(os.path.join(output_dir, file_pattern))
        for f_path in files_to_remove:
            try:
                os.remove(f_path)
                print(f"  Removed: {os.path.basename(f_path)}")
            except OSError as e:
                print(f"  Error removing {f_path}: {e}")

    # --- Generation Phase ---
    # Generate MCTS tree/board visualizations and get frame data
    visualization_frames = create_real_mcts_visualization(output_dir)
    
    # --- HTML Creation Phase ---
    if visualization_frames:
        # Create the single HTML page using the generated frame data
        html_path = create_single_page_visualization(output_dir, visualization_frames)
        print(f"\nVisualization complete. Open the following file in your browser:")
        # Try to provide a file URI for easier clicking
        try:
            file_uri = Path(os.path.abspath(html_path)).as_uri()
            print(file_uri)
        except ImportError:
             print(os.path.abspath(html_path)) # Fallback to absolute path
    else:
        print("\nNo visualization frames were generated. HTML file not created.")

if __name__ == "__main__":
    main() 
