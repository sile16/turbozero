"""
Example of how to use curriculum learning with StochasticTrainer.

This example shows how to progressively train on different game positions:
1. Start with endgame positions (easier to learn)
2. Progress to midgame positions 
3. Finally train on full games from the beginning

Usage in your notebook:
```python
from curriculum_example import create_backgammon_curriculum

# Create curriculum function
curriculum_fn = create_backgammon_curriculum(
    your_endgame_init_fn,
    your_midgame_init_fn, 
    your_normal_init_fn
)

# Set curriculum on trainer
trainer.set_curriculum_fn(curriculum_fn)
```
"""

def create_backgammon_curriculum(endgame_init_fn, midgame_init_fn, normal_init_fn, 
                                endgame_steps=50000, midgame_steps=100000):
    """
    Creates a curriculum function for backgammon training.
    
    Args:
        endgame_init_fn: Function to initialize near-endgame positions
        midgame_init_fn: Function to initialize midgame positions  
        normal_init_fn: Function to initialize normal starting positions
        endgame_steps: Number of training steps to use endgame positions
        midgame_steps: Number of training steps to use midgame positions
        
    Returns:
        Curriculum function that takes training_step and returns appropriate init_fn
    """
    def curriculum_func(training_step):
        if training_step < endgame_steps:
            return endgame_init_fn
        elif training_step < midgame_steps:
            return midgame_init_fn
        else:
            return normal_init_fn
    
    return curriculum_func


def create_progressive_curriculum(init_functions, step_thresholds):
    """
    Creates a curriculum function from a list of init functions and step thresholds.
    
    Args:
        init_functions: List of initialization functions, from easiest to hardest
        step_thresholds: List of training step thresholds for each stage
        
    Returns:
        Curriculum function
    """
    def curriculum_func(training_step):
        for i, threshold in enumerate(step_thresholds):
            if training_step < threshold:
                return init_functions[i]
        # Return the last (most difficult) init function
        return init_functions[-1]
    
    return curriculum_func


# Example usage patterns:

def linear_transition_curriculum(easy_init_fn, hard_init_fn, transition_start=50000, transition_end=100000):
    """
    Creates a curriculum that linearly transitions between two init functions.
    
    During transition period, randomly chooses between easy and hard based on training progress.
    """
    import jax.random
    
    def curriculum_func(training_step):
        if training_step < transition_start:
            return easy_init_fn
        elif training_step > transition_end:
            return hard_init_fn
        else:
            # Linear transition: probability of hard increases linearly
            progress = (training_step - transition_start) / (transition_end - transition_start)
            
            # Create a wrapper that probabilistically chooses between init functions
            def mixed_init_fn(key):
                choice_key, init_key = jax.random.split(key)
                use_hard = jax.random.uniform(choice_key) < progress
                return jax.lax.cond(
                    use_hard,
                    lambda k: hard_init_fn(k),
                    lambda k: easy_init_fn(k), 
                    init_key
                )
            return mixed_init_fn
    
    return curriculum_func