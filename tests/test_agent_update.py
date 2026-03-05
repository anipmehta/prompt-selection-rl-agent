"""
Unit tests for RLAgent Q-learning update logic.
"""

import pytest

from src.agent import RLAgent
from src.errors import ValidationError

# Test prompts
SAMPLE_PROMPTS = ["Be concise", "Explain step-by-step", "Use examples"]

# Test state/action
TEST_STATE = "summarize this article"
TEST_ACTION = SAMPLE_PROMPTS[0]

# Hyperparameters
LEARNING_RATE = 0.1
ZERO_LEARNING_RATE = 0.0
FULL_LEARNING_RATE = 1.0

# Rewards
POSITIVE_REWARD = 0.8
NEGATIVE_REWARD = -0.5
ZERO_REWARD = 0.0
REWARD_AT_MIN = -1.0
REWARD_AT_MAX = 1.0
REWARD_BELOW_MIN = -1.1
REWARD_ABOVE_MAX = 1.1

# Q-values
INITIAL_Q = 0.5


class TestQLearningFormula:
    """Test Q-learning update formula: Q(s,a) ← Q(s,a) + α[r - Q(s,a)]."""

    def test_update_from_zero(self):
        """Unseen state-action pair starts at 0.0, updates correctly."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS, learning_rate=LEARNING_RATE)
        agent.update(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)

        # Q = 0.0 + 0.1 * (0.8 - 0.0) = 0.08
        expected = LEARNING_RATE * POSITIVE_REWARD
        assert agent.q_table.get(TEST_STATE, TEST_ACTION) == pytest.approx(expected)

    def test_update_from_existing_q(self):
        """Update with pre-existing Q-value applies formula correctly."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS, learning_rate=LEARNING_RATE)
        agent.q_table.set(TEST_STATE, TEST_ACTION, INITIAL_Q)

        agent.update(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)

        # Q = 0.5 + 0.1 * (0.8 - 0.5) = 0.53
        expected = INITIAL_Q + LEARNING_RATE * (POSITIVE_REWARD - INITIAL_Q)
        assert agent.q_table.get(TEST_STATE, TEST_ACTION) == pytest.approx(expected)

    def test_negative_reward_decreases_q(self):
        """Negative reward should decrease Q-value."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS, learning_rate=LEARNING_RATE)
        agent.q_table.set(TEST_STATE, TEST_ACTION, INITIAL_Q)

        agent.update(TEST_STATE, TEST_ACTION, NEGATIVE_REWARD)

        # Q = 0.5 + 0.1 * (-0.5 - 0.5) = 0.5 + 0.1 * (-1.0) = 0.4
        expected = INITIAL_Q + LEARNING_RATE * (NEGATIVE_REWARD - INITIAL_Q)
        assert agent.q_table.get(TEST_STATE, TEST_ACTION) == pytest.approx(expected)

    def test_zero_learning_rate_no_change(self):
        """With α=0.0, Q-value should never change."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS, learning_rate=ZERO_LEARNING_RATE)
        agent.q_table.set(TEST_STATE, TEST_ACTION, INITIAL_Q)

        agent.update(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)

        assert agent.q_table.get(TEST_STATE, TEST_ACTION) == pytest.approx(INITIAL_Q)

    def test_full_learning_rate_replaces_q(self):
        """With α=1.0, Q-value should become the reward."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS, learning_rate=FULL_LEARNING_RATE)
        agent.q_table.set(TEST_STATE, TEST_ACTION, INITIAL_Q)

        agent.update(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)

        # Q = 0.5 + 1.0 * (0.8 - 0.5) = 0.8
        assert agent.q_table.get(TEST_STATE, TEST_ACTION) == pytest.approx(POSITIVE_REWARD)

    def test_successive_updates_converge(self):
        """Multiple updates with same reward should converge toward that reward."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS, learning_rate=LEARNING_RATE)

        for _ in range(100):
            agent.update(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)

        assert agent.q_table.get(TEST_STATE, TEST_ACTION) == pytest.approx(
            POSITIVE_REWARD, abs=0.01
        )


class TestRewardValidation:
    """Test reward range validation."""

    def test_reward_at_min_boundary(self):
        """Reward at -1.0 should be accepted."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.update(TEST_STATE, TEST_ACTION, REWARD_AT_MIN)

    def test_reward_at_max_boundary(self):
        """Reward at 1.0 should be accepted."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.update(TEST_STATE, TEST_ACTION, REWARD_AT_MAX)

    def test_zero_reward_accepted(self):
        """Reward of 0.0 should be accepted."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.update(TEST_STATE, TEST_ACTION, ZERO_REWARD)

    def test_reward_below_min_raises(self):
        """Reward below -1.0 should raise ValidationError."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        with pytest.raises(ValidationError):
            agent.update(TEST_STATE, TEST_ACTION, REWARD_BELOW_MIN)

    def test_reward_above_max_raises(self):
        """Reward above 1.0 should raise ValidationError."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        with pytest.raises(ValidationError):
            agent.update(TEST_STATE, TEST_ACTION, REWARD_ABOVE_MAX)


class TestInferenceModeUpdate:
    """Test that updates are rejected in inference mode."""

    def test_inference_mode_no_update(self):
        """Q-value should not change when agent is in inference mode."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.q_table.set(TEST_STATE, TEST_ACTION, INITIAL_Q)
        agent.mode = RLAgent.MODE_INFERENCE

        agent.update(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)

        assert agent.q_table.get(TEST_STATE, TEST_ACTION) == pytest.approx(INITIAL_Q)

    def test_inference_mode_still_validates_reward(self):
        """Inference mode should still validate reward before returning."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.mode = RLAgent.MODE_INFERENCE

        with pytest.raises(ValidationError):
            agent.update(TEST_STATE, TEST_ACTION, REWARD_ABOVE_MAX)
