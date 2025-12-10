"""Tests for 4-way conditional value head functions."""

import jax
import jax.numpy as jnp
import pytest
from core.evaluators.mcts.equity import (
    reward_to_value_targets,
    value_outputs_to_equity,
    terminal_value_probs_from_reward_4way,
    normalize_value_probs_4way,
    probs_to_equity_4way,
    equity_to_normalized,
    normalized_to_equity,
)
from core.training.loss_fns import four_way_value_loss


class TestRewardToValueTargets:
    """Test reward_to_value_targets function."""

    def test_single_win(self):
        """reward=1 -> single win"""
        targets, masks = reward_to_value_targets(jnp.array(1.0))
        assert jnp.allclose(targets, jnp.array([1, 0, 0, 0]))
        assert jnp.allclose(masks, jnp.array([1, 1, 0, 0]))

    def test_gammon_win(self):
        """reward=2 -> gammon win"""
        targets, masks = reward_to_value_targets(jnp.array(2.0))
        assert jnp.allclose(targets, jnp.array([1, 1, 0, 0]))
        assert jnp.allclose(masks, jnp.array([1, 1, 0, 1]))

    def test_backgammon_win(self):
        """reward=3 -> backgammon win"""
        targets, masks = reward_to_value_targets(jnp.array(3.0))
        assert jnp.allclose(targets, jnp.array([1, 1, 0, 1]))
        assert jnp.allclose(masks, jnp.array([1, 1, 0, 1]))

    def test_single_loss(self):
        """reward=-1 -> single loss"""
        targets, masks = reward_to_value_targets(jnp.array(-1.0))
        assert jnp.allclose(targets, jnp.array([0, 0, 0, 0]))
        assert jnp.allclose(masks, jnp.array([1, 0, 1, 0]))

    def test_gammon_loss(self):
        """reward=-2 -> gammon loss"""
        targets, masks = reward_to_value_targets(jnp.array(-2.0))
        assert jnp.allclose(targets, jnp.array([0, 0, 1, 0]))
        assert jnp.allclose(masks, jnp.array([1, 0, 1, 1]))

    def test_backgammon_loss(self):
        """reward=-3 -> backgammon loss"""
        targets, masks = reward_to_value_targets(jnp.array(-3.0))
        assert jnp.allclose(targets, jnp.array([0, 0, 1, 1]))
        assert jnp.allclose(masks, jnp.array([1, 0, 1, 1]))

    def test_batched(self):
        """Test batched operation via vmap."""
        rewards = jnp.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
        targets, masks = jax.vmap(reward_to_value_targets)(rewards)

        # Check shapes
        assert targets.shape == (6, 4)
        assert masks.shape == (6, 4)

        # Check first row (single win)
        assert jnp.allclose(targets[0], jnp.array([1, 0, 0, 0]))


class TestValueOutputsToEquity:
    """Test value_outputs_to_equity function."""

    def test_single_win(self):
        """[1, 0, 0, 0] -> equity = 1.0 (single win)"""
        probs = jnp.array([1.0, 0.0, 0.0, 0.0])
        equity = value_outputs_to_equity(probs)
        assert jnp.isclose(equity, 1.0)

    def test_gammon_win(self):
        """[1, 1, 0, 0] -> equity = 2.0 (gammon win, no backgammon)"""
        probs = jnp.array([1.0, 1.0, 0.0, 0.0])
        equity = value_outputs_to_equity(probs)
        assert jnp.isclose(equity, 2.0)

    def test_backgammon_win(self):
        """[1, 1, 0, 1] -> equity = 3.0 (backgammon win)"""
        probs = jnp.array([1.0, 1.0, 0.0, 1.0])
        equity = value_outputs_to_equity(probs)
        assert jnp.isclose(equity, 3.0)

    def test_single_loss(self):
        """[0, 0, 0, 0] -> equity = -1.0 (single loss)"""
        probs = jnp.array([0.0, 0.0, 0.0, 0.0])
        equity = value_outputs_to_equity(probs)
        assert jnp.isclose(equity, -1.0)

    def test_gammon_loss(self):
        """[0, 0, 1, 0] -> equity = -2.0 (gammon loss)"""
        probs = jnp.array([0.0, 0.0, 1.0, 0.0])
        equity = value_outputs_to_equity(probs)
        assert jnp.isclose(equity, -2.0)

    def test_backgammon_loss(self):
        """[0, 0, 1, 1] -> equity = -3.0 (backgammon loss)"""
        probs = jnp.array([0.0, 0.0, 1.0, 1.0])
        equity = value_outputs_to_equity(probs)
        assert jnp.isclose(equity, -3.0)

    def test_even_position(self):
        """[0.5, 0, 0, 0] -> equity = 0.0 (50-50 single game)"""
        probs = jnp.array([0.5, 0.0, 0.0, 0.0])
        equity = value_outputs_to_equity(probs)
        assert jnp.isclose(equity, 0.0)

    def test_intermediate_position(self):
        """Test an intermediate position with gammon chances."""
        # win=0.5, gam_win_cond=0.2, gam_loss_cond=0.3, bg_rate=0.1
        probs = jnp.array([0.5, 0.2, 0.3, 0.1])
        equity = value_outputs_to_equity(probs)

        # Manual calculation:
        # win = 0.5, loss = 0.5
        # gam_win = 0.5 * 0.2 = 0.1
        # gam_loss = 0.5 * 0.3 = 0.15
        # bg_win = 0.1 * 0.1 = 0.01
        # bg_loss = 0.15 * 0.1 = 0.015
        # single_win = 0.5 - 0.1 = 0.4
        # single_loss = 0.5 - 0.15 = 0.35
        # gammon_win_only = 0.1 - 0.01 = 0.09
        # gammon_loss_only = 0.15 - 0.015 = 0.135
        # equity = 0.4*1 - 0.35*1 + 0.09*2 - 0.135*2 + 0.01*3 - 0.015*3
        #        = 0.05 + 0.18 - 0.27 + 0.03 - 0.045
        #        = -0.055
        expected = 0.4 - 0.35 + 0.09*2 - 0.135*2 + 0.01*3 - 0.015*3
        assert jnp.isclose(equity, expected, rtol=1e-5)


class TestTerminalValueProbs4way:
    """Test terminal_value_probs_from_reward_4way function."""

    def test_single_win(self):
        probs = terminal_value_probs_from_reward_4way(jnp.array(1.0))
        assert jnp.allclose(probs, jnp.array([1, 0, 0, 0]))

    def test_gammon_win(self):
        probs = terminal_value_probs_from_reward_4way(jnp.array(2.0))
        assert jnp.allclose(probs, jnp.array([1, 1, 0, 0]))

    def test_backgammon_win(self):
        probs = terminal_value_probs_from_reward_4way(jnp.array(3.0))
        assert jnp.allclose(probs, jnp.array([1, 1, 0, 1]))

    def test_single_loss(self):
        probs = terminal_value_probs_from_reward_4way(jnp.array(-1.0))
        assert jnp.allclose(probs, jnp.array([0, 0, 0, 0]))

    def test_gammon_loss(self):
        probs = terminal_value_probs_from_reward_4way(jnp.array(-2.0))
        assert jnp.allclose(probs, jnp.array([0, 0, 1, 0]))

    def test_backgammon_loss(self):
        probs = terminal_value_probs_from_reward_4way(jnp.array(-3.0))
        assert jnp.allclose(probs, jnp.array([0, 0, 1, 1]))


class TestNormalizeValueProbs4way:
    """Test normalize_value_probs_4way function."""

    def test_zeros(self):
        """Zeros should map to 0.5 (sigmoid center)."""
        probs = normalize_value_probs_4way(jnp.zeros(4))
        assert jnp.allclose(probs, jnp.array([0.5, 0.5, 0.5, 0.5]))

    def test_large_positive(self):
        """Large positive logits should map to ~1."""
        probs = normalize_value_probs_4way(jnp.array([10.0, 10.0, 10.0, 10.0]))
        assert jnp.all(probs > 0.99)

    def test_large_negative(self):
        """Large negative logits should map to ~0."""
        probs = normalize_value_probs_4way(jnp.array([-10.0, -10.0, -10.0, -10.0]))
        assert jnp.all(probs < 0.01)


class TestEquityConversions:
    """Test equity normalization functions."""

    def test_equity_to_normalized(self):
        """Test equity -> normalized conversion."""
        assert jnp.isclose(equity_to_normalized(jnp.array(-3.0)), 0.0)
        assert jnp.isclose(equity_to_normalized(jnp.array(0.0)), 0.5)
        assert jnp.isclose(equity_to_normalized(jnp.array(3.0)), 1.0)

    def test_normalized_to_equity(self):
        """Test normalized -> equity conversion."""
        assert jnp.isclose(normalized_to_equity(jnp.array(0.0)), -3.0)
        assert jnp.isclose(normalized_to_equity(jnp.array(0.5)), 0.0)
        assert jnp.isclose(normalized_to_equity(jnp.array(1.0)), 3.0)

    def test_roundtrip(self):
        """Test roundtrip conversion."""
        for eq in [-3.0, -1.5, 0.0, 1.5, 3.0]:
            eq = jnp.array(eq)
            assert jnp.isclose(normalized_to_equity(equity_to_normalized(eq)), eq)


class TestProbsToEquity4way:
    """Test probs_to_equity_4way wrapper function."""

    def test_single_win_normalized(self):
        """Single win should map to normalized ~0.667 (equity 1.0)."""
        probs = jnp.array([1.0, 0.0, 0.0, 0.0])
        normalized = probs_to_equity_4way(probs)
        expected = equity_to_normalized(jnp.array(1.0))
        assert jnp.isclose(normalized, expected)

    def test_backgammon_win_normalized(self):
        """Backgammon win should map to normalized 1.0 (equity 3.0)."""
        probs = jnp.array([1.0, 1.0, 0.0, 1.0])
        normalized = probs_to_equity_4way(probs)
        assert jnp.isclose(normalized, 1.0)


class TestFourWayValueLoss:
    """Test four_way_value_loss function."""

    def test_perfect_prediction(self):
        """Loss should be near zero for perfect predictions."""
        # Large positive logits for targets=1, large negative for targets=0
        pred_logits = jnp.array([[10.0, -10.0, -10.0, -10.0]])  # predicts [1, 0, 0, 0]
        targets = jnp.array([[1.0, 0.0, 0.0, 0.0]])
        masks = jnp.array([[1.0, 1.0, 0.0, 0.0]])  # win mask

        loss = four_way_value_loss(pred_logits, targets, masks)
        assert loss < 0.01

    def test_bad_prediction(self):
        """Loss should be high for bad predictions."""
        pred_logits = jnp.array([[-10.0, 10.0, 10.0, 10.0]])  # predicts [0, 1, 1, 1]
        targets = jnp.array([[1.0, 0.0, 0.0, 0.0]])
        masks = jnp.array([[1.0, 1.0, 0.0, 0.0]])

        loss = four_way_value_loss(pred_logits, targets, masks)
        assert loss > 1.0

    def test_mask_ignores_outputs(self):
        """Masked outputs should not contribute to loss."""
        pred_logits = jnp.array([[0.0, -10.0, 10.0, 10.0]])  # bad predictions for indices 2,3
        targets = jnp.array([[0.5, 0.0, 0.0, 0.0]])
        masks = jnp.array([[1.0, 0.0, 0.0, 0.0]])  # only loss on index 0

        loss = four_way_value_loss(pred_logits, targets, masks)
        # Should only compute loss for index 0, which is moderate
        assert 0.3 < loss < 1.0

    def test_batched(self):
        """Test batched loss computation."""
        pred_logits = jnp.array([
            [10.0, -10.0, -10.0, -10.0],
            [-10.0, -10.0, 10.0, -10.0],
        ])
        targets = jnp.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        masks = jnp.array([
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0],
        ])

        loss = four_way_value_loss(pred_logits, targets, masks)
        assert loss < 0.1  # Both predictions are good


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
