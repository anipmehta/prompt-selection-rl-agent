# pylint: disable=protected-access
"""
Integration tests for ExperienceBuffer integration with RLAgent.

Tests the batch training workflow: store experiences → train → verify Q-values.
"""

import pytest

from src.agent import RLAgent

# Test prompts
SAMPLE_PROMPTS = ["Be concise", "Explain step-by-step", "Use examples"]

# Test states
STATE_A = "summarize this article"
STATE_B = "translate this text"

# Test actions (must be from SAMPLE_PROMPTS)
ACTION_A = SAMPLE_PROMPTS[0]
ACTION_B = SAMPLE_PROMPTS[1]

# Rewards
POSITIVE_REWARD = 0.8
NEGATIVE_REWARD = -0.3

# Hyperparameters
LEARNING_RATE = 0.1


class TestStoreExperience:
    """Test that store_experience delegates to the internal buffer."""

    def test_store_single_experience(self) -> None:
        """Storing one experience should increase buffer size to 1."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.store_experience(STATE_A, ACTION_A, POSITIVE_REWARD)

        assert agent._buffer.size() == 1

    def test_store_multiple_experiences(self) -> None:
        """Storing multiple experiences should track all of them."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.store_experience(STATE_A, ACTION_A, POSITIVE_REWARD)
        agent.store_experience(STATE_B, ACTION_B, NEGATIVE_REWARD)

        assert agent._buffer.size() == 2


class TestTrainBatch:
    """Test batch training updates Q-values from buffered experiences."""

    def test_train_batch_updates_q_values(self) -> None:
        """After train_batch, Q-values should reflect stored experiences."""
        agent = RLAgent(
            prompts=SAMPLE_PROMPTS, learning_rate=LEARNING_RATE
        )

        agent.store_experience(STATE_A, ACTION_A, POSITIVE_REWARD)
        agent.store_experience(STATE_B, ACTION_B, NEGATIVE_REWARD)
        agent.train_batch()

        # Q = 0.0 + 0.1 * (0.8 - 0.0) = 0.08
        expected_a = LEARNING_RATE * POSITIVE_REWARD
        assert agent.q_table.get(STATE_A, ACTION_A) == pytest.approx(expected_a)

        # Q = 0.0 + 0.1 * (-0.3 - 0.0) = -0.03
        expected_b = LEARNING_RATE * NEGATIVE_REWARD
        assert agent.q_table.get(STATE_B, ACTION_B) == pytest.approx(expected_b)

    def test_train_batch_with_empty_buffer(self) -> None:
        """train_batch on an empty buffer should not error or change Q-values."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.train_batch()

        assert len(agent.q_table) == 0

    def test_train_batch_does_not_clear_buffer(self) -> None:
        """train_batch should leave the buffer intact for potential replay."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.store_experience(STATE_A, ACTION_A, POSITIVE_REWARD)
        agent.train_batch()

        assert agent._buffer.size() == 1


class TestClearBuffer:
    """Test that clear_buffer empties the experience buffer."""

    def test_clear_buffer_empties(self) -> None:
        """After clear_buffer, buffer size should be 0."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.store_experience(STATE_A, ACTION_A, POSITIVE_REWARD)
        agent.store_experience(STATE_B, ACTION_B, NEGATIVE_REWARD)

        agent.clear_buffer()

        assert agent._buffer.size() == 0

    def test_clear_buffer_on_empty(self) -> None:
        """Clearing an already-empty buffer should not error."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.clear_buffer()
        assert agent._buffer.size() == 0


class TestBatchTrainingWorkflow:  # pylint: disable=too-few-public-methods
    """End-to-end integration: collect → train → verify → clear."""

    def test_full_workflow(self) -> None:
        """Complete batch training cycle updates Q-values then clears."""
        agent = RLAgent(
            prompts=SAMPLE_PROMPTS, learning_rate=LEARNING_RATE
        )

        # 1. Collect experiences
        agent.store_experience(STATE_A, ACTION_A, POSITIVE_REWARD)
        agent.store_experience(STATE_A, ACTION_A, POSITIVE_REWARD)

        # 2. Train
        agent.train_batch()

        # First update: Q = 0.0 + 0.1*(0.8 - 0.0) = 0.08
        # Second update: Q = 0.08 + 0.1*(0.8 - 0.08) = 0.08 + 0.072 = 0.152
        q_val = agent.q_table.get(STATE_A, ACTION_A)
        first = LEARNING_RATE * POSITIVE_REWARD
        expected = first + LEARNING_RATE * (POSITIVE_REWARD - first)
        assert q_val == pytest.approx(expected)

        # 3. Clear buffer
        agent.clear_buffer()
        assert agent._buffer.size() == 0

        # 4. Q-values persist after clearing buffer
        assert agent.q_table.get(STATE_A, ACTION_A) == pytest.approx(expected)
