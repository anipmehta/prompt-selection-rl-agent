"""Unit tests for extensibility interfaces (src/interfaces.py)."""

import pytest

from src.interfaces import ActionExecutor, RewardFunction

# Test constants
TEST_PROMPT = "Be concise"
TEST_TASK = "summarize this article"
TEST_RESULT = "This article is about RL agents."
TEST_REWARD = 0.8


class StubExecutor(ActionExecutor):  # pylint: disable=too-few-public-methods
    """Concrete implementation for testing."""

    def execute(self, prompt: str, task: str) -> str:
        """Return a fixed response."""
        return f"response for {task}"


class StubReward(RewardFunction):  # pylint: disable=too-few-public-methods
    """Concrete implementation for testing."""

    def compute(self, task: str, prompt: str, result: str) -> float:
        """Return a fixed reward."""
        return TEST_REWARD


class TestActionExecutor:
    """Tests for ActionExecutor interface."""

    def test_cannot_instantiate_directly(self) -> None:
        """ActionExecutor is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            ActionExecutor()  # pylint: disable=abstract-class-instantiated

    def test_subclass_can_execute(self) -> None:
        """Concrete subclass should implement execute."""
        executor = StubExecutor()
        result = executor.execute(TEST_PROMPT, TEST_TASK)
        assert result == f"response for {TEST_TASK}"

    def test_execute_returns_string(self) -> None:
        """execute should return a string."""
        executor = StubExecutor()
        result = executor.execute(TEST_PROMPT, TEST_TASK)
        assert isinstance(result, str)


class TestRewardFunction:
    """Tests for RewardFunction interface."""

    def test_cannot_instantiate_directly(self) -> None:
        """RewardFunction is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            RewardFunction()  # pylint: disable=abstract-class-instantiated

    def test_subclass_can_compute(self) -> None:
        """Concrete subclass should implement compute."""
        reward_fn = StubReward()
        reward = reward_fn.compute(TEST_TASK, TEST_PROMPT, TEST_RESULT)
        assert reward == TEST_REWARD

    def test_compute_returns_float(self) -> None:
        """compute should return a float."""
        reward_fn = StubReward()
        reward = reward_fn.compute(TEST_TASK, TEST_PROMPT, TEST_RESULT)
        assert isinstance(reward, float)
