#!/usr/bin/env python
"""
Create visualizations of MCTS trees for PGX Backgammon.
This script generates sample MCTS tree visualizations using graphviz directly,
without relying on the actual MCTS implementation.
"""

import os
import shutil
from pathlib import Path
from functools import partial
import json
import graphviz
import numpy as np
import glob
from typing import Any
Array = Any

import jax
import jax.numpy as jnp

import pgx.backgammon as bg
from core.evaluators.mcts.stochastic_mcts import StochasticMCTS
from core.evaluators.mcts.action_selection import PUCTSelector
from core.types import StepMetadata
from core.bgcommon import bg_pip_count_eval, bg_step_fn

# Check if graphviz 'dot' command is available
GRAPHVIZ_AVAILABLE = shutil.which('dot') is not None
if not GRAPHVIZ_AVAILABLE:
    print("WARNING: Graphviz 'dot' command not found. Install with: sudo apt-get install graphviz")
    print("         SVG rendering will be skipped, but DOT source files will be saved.")

# Create a sample Backgammon environment for action labels and SVG generation
backgammon_env = bg.Backgammon(short_game=True)
print("Backgammon environment initialized.")

def tree_to_graph(eval_state, output_dir, boards_dir, max_nodes_to_render=50, verbose=False):
    """Convert a tree to a Graphviz graph.

    Args:
        eval_state: The MCTS tree state
        output_dir: Directory for output files
        boards_dir: Directory for board SVG files
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
                policy_array = np.array(node_data.p) 
                # Get indices of top k probabilities
                top_k = 8
                top_indices = np.argsort(policy_array)[-top_k:][::-1]
                policy_str_parts = []
                no_op = True
                is_stochastic_node = StochasticMCTS.is_node_idx_stochastic(tree, node_idx)
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
        
        # Set node style - simplified to just deterministic vs stochastic
        is_stochastic = StochasticMCTS.is_node_idx_stochastic(tree, node_idx)
        if is_stochastic:
            fillcolor = 'lightcoral'  # Stochastic nodes
        elif node_data.terminated:
            fillcolor = 'grey'        # Terminal nodes
        else:
            fillcolor = 'lightblue'   # Deterministic nodes
        
        # === Add Board SVG Tooltip ===
        try:
            if hasattr(node_data, 'embedding') and node_data.embedding is not None:
                board_state = node_data.embedding
                # Check if it looks like a valid state object (simple check)
                if hasattr(board_state, 'to_svg'): 
                    board_svg_str = board_state.to_svg()
                    board_filename_rel = os.path.join("boards", f"board_node_{node_idx}.svg")
                    board_filename_abs = os.path.join(output_dir, board_filename_rel)
                    save_svg_string(board_svg_str, board_filename_abs)
                    # Update the node with the tooltip
                    graph.node(node_id, label=label, fillcolor=fillcolor, tooltip=board_filename_rel) 
                else:
                     # Add node without tooltip if embedding is not a state
                     graph.node(node_id, label=label, fillcolor=fillcolor)
            else:
                 # Add node without tooltip if no embedding
                 graph.node(node_id, label=label, fillcolor=fillcolor)
        except Exception as e:
             print(f"WARN: Failed to generate/save board SVG for node {node_idx}: {e}")
             # Add node without tooltip on error
             graph.node(node_id, label=label, fillcolor=fillcolor)
        # === END Add Board SVG Tooltip ===
        
        graph.node(node_id, label=label, fillcolor=fillcolor)

        # Add edges to children (only to nodes we're rendering)
        for action in range(tree.branching_factor):
            child_node_idx = int(tree.edge_map[node_idx, action])
            if child_node_idx != -1 and child_node_idx < num_nodes_to_render:  # Valid edge within render limit
                child_id = str(child_node_idx)
                
                # Format edge label - REFINED LOGIC
                try:
                    is_parent_stochastic = StochasticMCTS.is_node_idx_stochastic(tree, node_idx)
                    if is_parent_stochastic:
                        # Parent is stochastic, action selects a stochastic outcome
                        try:
                            action_str = bg.stochastic_action_to_str(action)
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

def save_svg_string(svg_string, filepath):
    """Saves an SVG string to a file."""
    try:
        with open(filepath, 'w') as f:
            f.write(svg_string)
    except Exception as e:
        print(f"Error saving SVG to {filepath}: {e}")


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
    """Create visualizations of a real StochasticMCTS tree over multiple iterations and turns."""
    print("Creating StochasticMCTS visualization...")
    
    # Ensure boards subdirectory exists
    boards_dir = os.path.join(output_dir, "boards")
    os.makedirs(boards_dir, exist_ok=True)
    
    # Set up environment
    key = jax.random.PRNGKey(48)
    
    # Set up MCTS with minimal iterations for visualization
    mcts = StochasticMCTS(
        eval_fn=bg_pip_count_eval,
        action_selector=PUCTSelector(),
        branching_factor=backgammon_env.num_actions,
        max_nodes=200,  # Reduced for faster visualization
        num_iterations=1,  # Use 1 iteration per step for step-by-step visualization
        stochastic_action_probs=backgammon_env.stochastic_action_probs,
        discount=-1.0,
        temperature=0.2,
        persist_tree=True,  # Critical to preserve tree between steps
    )
    
    # Confirm the MCTS configuration
    print(f"MCTS config: max_nodes={mcts.max_nodes}, persist_tree={mcts.persist_tree}")
    
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
        step=state._step_count
    )

    visualization_frames = []
    frame_idx = 0
    total_turns = 5  # Reduced number of turns
    iterations_per_deterministic_turn = 10  # Reduced iterations per turn
    render_every_n_iterations = 2  # Only render every Nth iteration for performance
    verbose = False  # Set to True for detailed debug output

    # Keep track of the latest board SVG paths relative to output_dir
    last_saved_curr_board_path = None
    last_saved_prev_board_path = None

    # Save initial board state
    initial_board_filename = 'board_turn_0_initial.svg'
    initial_board_path_abs = os.path.join(boards_dir, initial_board_filename)
    save_svg_string(state.to_svg(), initial_board_path_abs)
    last_saved_curr_board_path = os.path.join("boards", initial_board_filename) # Relative path

    # Simulation loop
    for turn in range(total_turns):
        print(f"\n--- Processing Turn {turn + 1}/{total_turns} ---")
        print(f"Initial state: Player {state.current_player}, Stochastic: {state.is_stochastic}")

        current_iterations = 0
        # If the state is stochastic, MCTS handles it in one evaluate call
        is_turn_stochastic = bool(state.is_stochastic) # Ensure it's a Python bool

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
                env_step_fn=partial(bg_step_fn, backgammon_env)
            )
            # Important: For stochastic nodes, the tree stays the same
            eval_state = mcts_output.eval_state
            action = mcts_output.action
            print(f"  Turn {turn+1}: Stochastic action selected: {action} ({bg.stochastic_action_to_str(action)})")
            
            # Add selected stochastic action to visualization
            graph = tree_to_graph(eval_state, output_dir, boards_dir, verbose=verbose)
            
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
                'current_board_path': last_saved_curr_board_path,
                'previous_board_path': last_saved_prev_board_path,
                'action_str': '' # No action yet
            })
            frame_idx += 1
            
            # --- Action Step ---
            action_str = bg.stochastic_action_to_str(action)
            previous_state = state
            last_saved_prev_board_path = last_saved_curr_board_path  # Current becomes previous

            # Step environment with the chosen stochastic action
            state, metadata = bg_step_fn(backgammon_env, state, action, step_key)
            # Step MCTS tree - always step after environment steps with stochastic actions
            eval_state = mcts.step(eval_state, action)

            # Save current board state SVG
            curr_board_filename = f'board_turn_{turn}_stochastic_curr.svg'
            curr_board_path_abs = os.path.join(boards_dir, curr_board_filename)
            save_svg_string(state.to_svg(), curr_board_path_abs)
            last_saved_curr_board_path = os.path.join("boards", curr_board_filename) # Update last saved current board

            # Visualize MCTS tree *after* the stochastic step
            graph_after_step = tree_to_graph(eval_state, output_dir, boards_dir, verbose=verbose)
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
                'previous_board_path': last_saved_prev_board_path,
                'current_board_path': last_saved_curr_board_path,
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
                    env_step_fn=partial(bg_step_fn, backgammon_env)
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
                    graph = tree_to_graph(eval_state, output_dir, boards_dir, verbose=verbose)
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
                        'current_board_path': last_saved_curr_board_path,
                        'previous_board_path': last_saved_prev_board_path,
                        'action_str': ''  # No action during iteration
                    })
                    frame_idx += 1

                current_iterations += 1

            # --- Action Step ---
            action = final_action  # Use action determined by MCTS after all iterations
            action_str = bg.action_to_str(action)
            print(f"  Turn {turn+1}: Applying deterministic action: {action} ({action_str})")
            previous_state = state
            last_saved_prev_board_path = last_saved_curr_board_path # Current becomes previous
            key, step_key = jax.random.split(key)

            # Step environment based on the chosen action 
            state, metadata = bg_step_fn(backgammon_env, state, action, step_key)

            # Step MCTS tree - always step after a deterministic action
            eval_state = mcts.step(eval_state, action)

            # Save current board state SVG
            curr_board_filename = f'board_turn_{turn}_deterministic_curr.svg'
            curr_board_path_abs = os.path.join(boards_dir, curr_board_filename)
            save_svg_string(state.to_svg(), curr_board_path_abs)
            last_saved_curr_board_path = os.path.join("boards", curr_board_filename)  # Update last saved current board

            # Visualize MCTS tree *after* the step
            graph_after_step = tree_to_graph(eval_state, output_dir, boards_dir, verbose=verbose)
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
                'previous_board_path': last_saved_prev_board_path,
                'current_board_path': last_saved_curr_board_path,
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
                'previous_board_path': last_saved_prev_board_path, # Show board before final state
                'current_board_path': last_saved_curr_board_path, # Show final board state
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
    <title>Stochastic MCTS Visualization</title>
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
    <h1>Stochastic MCTS Visualization</h1>

    <div class="board-display-container">
        <div id="currBoardContainer" class="board-container">
            <h3>Current Board State</h3>
            <div id="currBoardLoader" class="loader"></div>
            <img id="currBoardImage" src="" alt="Current Board State">
        </div>
        <div class="action-display">
            <span id="actionArrow">&#11013;</span> <!-- Left arrow -->
            <span id="actionText" class="action-text"></span>
        </div>
        <div id="prevBoardContainer" class="board-container">
            <h3>Previous Board State</h3>
            <div id="prevBoardLoader" class="loader"></div>
            <img id="prevBoardImage" src="" alt="Previous Board State">
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
        const prevBoardImage = document.getElementById('prevBoardImage');
        const currBoardImage = document.getElementById('currBoardImage');
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
        const prevBoardLoader = document.getElementById('prevBoardLoader');
        const currBoardLoader = document.getElementById('currBoardLoader');

        function showLoader(loaderElement, imageElement) {{
            loaderElement.style.display = 'block';
            imageElement.style.visibility = 'hidden'; // Hide image but keep space
        }}

        function hideLoader(loaderElement, imageElement) {{
            // Use requestAnimationFrame to ensure DOM update happens before showing image
            requestAnimationFrame(() => {{
                loaderElement.style.display = 'none';
                imageElement.style.visibility = 'visible';
            }});
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
            showLoader(mctsLoader, mctsImage);
            mctsImage.onload = () => hideLoader(mctsLoader, mctsImage);
            mctsImage.onerror = () => {{
                 hideLoader(mctsLoader, mctsImage);
                 mctsImage.alt = "Error loading MCTS graph";
                 mctsImage.src = ""; // Clear src on error
                 console.error("Error loading MCTS image:", frame.graph_path);
             }};
            // Check if graph_path exists before setting src
            mctsImage.src = frame.graph_path ? frame.graph_path : "";
            mctsImage.alt = frame.graph_path ? `MCTS Tree - ${{frame.info}}` : "MCTS Graph Unavailable";

            // --- Update Board Displays ---
            // Current Board
            showLoader(currBoardLoader, currBoardImage);
            currBoardImage.onload = () => hideLoader(currBoardLoader, currBoardImage);
            currBoardImage.onerror = () => {{
                 hideLoader(currBoardLoader, currBoardImage);
                 currBoardImage.alt = "Error loading current board";
                 currBoardImage.src = ""; // Clear src
                 console.error("Error loading current board image:", frame.current_board_path);
             }};
             currBoardImage.src = frame.current_board_path ? frame.current_board_path : "";
             currBoardImage.alt = frame.current_board_path ? `Current Board State - ${{frame.info}}` : "Current Board Unavailable";

            // Previous Board
            showLoader(prevBoardLoader, prevBoardImage);
            prevBoardImage.onload = () => hideLoader(prevBoardLoader, prevBoardImage);
             prevBoardImage.onerror = () => {{
                 hideLoader(prevBoardLoader, prevBoardImage);
                 prevBoardImage.alt = "Error loading previous board";
                 prevBoardImage.src = ""; // Clear src
                 console.error("Error loading previous board image:", frame.previous_board_path);
             }};
            prevBoardImage.src = frame.previous_board_path ? frame.previous_board_path : ""; // Use path or empty string
            prevBoardImage.alt = frame.previous_board_path ? `Previous Board State - ${{frame.info}}` : "Previous Board Unavailable";

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
    for file_pattern in ["*.svg", "boards/*.svg"]:
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