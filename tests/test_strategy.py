"""Unit tests for learning strategies and strategy swapping."""

import pytest

from src.agent import RLAgent
from src.strategy import QLearningStrategy

# Test prompts
SAMPLE_PROMPTS = ["Be concise", "Be detailed", "Be creative"]

# Test constants
TEST_STATE = "summarize this article"
TEST_ACTION = SAMPLE_PROMPTS[0]
POSITIVE_REWARD = 0.8
LEARNING_RATE = 0.1


class TestQLearningStrategy:
    """Tests for QLearningStrategy in isolation."""

    def test_update_applies_q_learning_formula(self) -> None:
        """Q-value should follow Q(s,a) + α[r - Q(s,a)]."""
        strategy = QLearningStrategy(learning_rate=LEARNING_RATE)
        strategy.update(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)

        expected = LEARNING_RATE * POSITIVE_REWARD
        assert strategy.q_table.get(TEST_STATE, TEST_ACTION) == pytest.approx(expected)

    def test_select_action_returns_valid_prompt(self) -> None:
        """select_action should return a prompt from the list."""
        strategy = QLearningStrategy()
        for _ in range(50):
            action = strategy.select_action(TEST_STATE, SAMPLE_PROMPTS, 1.0)
            assert action in SAMPLE_PROMPTS

    def test_get_q_values_empty_state(self) -> None:
        """Unseen state should return empty dict."""
        strategy = QLearningStrategy()
        assert not strategy.get_q_values("unseen")

    def test_get_table_returns_full_table(self) -> None:
        """get_table should return all stored Q-values."""
        strategy = QLearningStrategy(learning_rate=LEARNING_RATE)
        strategy.update(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)

        table = strategy.get_table()
        assert TEST_STATE in table
        assert TEST_ACTION in table[TEST_STATE]


class TestStrategySwapping:
    """Tests that the agent works with custom strategies."""

    def test_agent_uses_custom_strategy(self) -> None:
        """Agent should delegate to the provided strategy."""
        strategy = QLearningStrategy(learning_rate=0.5)
        agent = RLAgent(prompts=SAMPLE_PROMPTS, strategy=strategy)

        agent.update(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)

        # With lr=0.5: Q = 0.0 + 0.5 * (0.8 - 0.0) = 0.4
        expected = 0.5 * POSITIVE_REWARD
        assert agent.q_table.get(TEST_STATE, TEST_ACTION) == pytest.approx(expected)

    def test_default_strategy_is_q_learning(self) -> None:
        """Without explicit strategy, agent should use QLearningStrategy."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        assert isinstance(agent._strategy, QLearningStrategy)  # pylint: disable=protected-access

    def test_custom_strategy_overrides_learning_rate(self) -> None:
        """Custom strategy's learning rate should be used, not agent's."""
        strategy = QLearningStrategy(learning_rate=0.9)
        agent = RLAgent(
            prompts=SAMPLE_PROMPTS,
            learning_rate=0.1,
            strategy=strategy,
        )

        agent.update(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)

        # Strategy lr=0.9 should be used: Q = 0.0 + 0.9 * 0.8 = 0.72
        expected = 0.9 * POSITIVE_REWARD
        assert agent.q_table.get(TEST_STATE, TEST_ACTION) == pytest.approx(expected)
