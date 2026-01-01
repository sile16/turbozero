"""Compatibility module: StochasticMCTS has been migrated to UnifiedMCTS.

This module provides backwards compatibility for scripts that import StochasticMCTS.
All functionality is now in UnifiedMCTS.

Migration notes:
- Replace `from core.evaluators.mcts.stochastic_mcts import StochasticMCTS`
  with `from core.evaluators.mcts.unified_mcts import UnifiedMCTS`
- StochasticMCTS parameters map to UnifiedMCTS as follows:
  - progressive_threshold: no longer needed (removed)
  - All other parameters are the same
- New features in UnifiedMCTS:
  - gumbel_k: number of actions to sample at root (default: 16)
  - c_visit, c_scale: Gumbel scaling parameters
  - Temperature can be a callable for scheduling
"""

import warnings

from core.evaluators.mcts.unified_mcts import UnifiedMCTS as _UnifiedMCTS


class StochasticMCTS(_UnifiedMCTS):
    """Compatibility wrapper: use UnifiedMCTS instead.

    This class exists for backwards compatibility. New code should use UnifiedMCTS directly.
    """

    def __init__(
        self,
        eval_fn,
        action_selector,
        policy_size,
        max_nodes,
        num_iterations,
        decision_step_fn,
        stochastic_step_fn=None,
        stochastic_action_probs=None,
        temperature=1.0,
        progressive_threshold=1.0,  # Ignored - no longer used
        gumbel_k=None,
        c_visit=50.0,
        c_scale=1.0,
    ):
        # Warn about deprecation
        warnings.warn(
            "StochasticMCTS is deprecated. Use UnifiedMCTS instead. "
            "See core/evaluators/mcts/stochastic_mcts.py for migration notes.",
            DeprecationWarning,
            stacklevel=2
        )

        # Ignore progressive_threshold (not used in UnifiedMCTS)
        if progressive_threshold != 1.0:
            warnings.warn(
                "progressive_threshold parameter is no longer used in UnifiedMCTS",
                DeprecationWarning,
                stacklevel=2
            )

        # Default gumbel_k to min(16, policy_size) if not specified
        if gumbel_k is None:
            gumbel_k = min(16, policy_size)

        super().__init__(
            eval_fn=eval_fn,
            action_selector=action_selector,
            policy_size=policy_size,
            max_nodes=max_nodes,
            num_iterations=num_iterations,
            decision_step_fn=decision_step_fn,
            stochastic_step_fn=stochastic_step_fn,
            stochastic_action_probs=stochastic_action_probs,
            temperature=temperature,
            gumbel_k=gumbel_k,
            c_visit=c_visit,
            c_scale=c_scale,
        )
