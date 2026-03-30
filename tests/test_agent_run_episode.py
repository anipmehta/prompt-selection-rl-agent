"""Wiring tests for agent.run_episode (integrates executor + reward function)."""

from src.agent import RLAgent
from src.interfaces import ActionExecutor, RewardFunction

# Test constants
SAMPLE_PROMPTS = ["Be concise", "Be detailed"]
TEST_TASK = "summarize this article"
FIXED_RESPONSE = "This is a summary."
FIXED_REWARD = 0.7


class StubExecutor(ActionExecutor):  # pylint: disable=too-few-public-methods
    """Returns a fixed response."""

    def execute(self, prompt: str, task: str) -> str:
        """Return fixed response."""
        return FIXED_RESPONSE


class StubReward(RewardFunction):  # pylint: disable=too-few-public-methods
    """Returns a fixed reward."""

    def compute(self, task: str, prompt: str, result: str) -> float:
        """Return fixed reward."""
        return FIXED_REWARD


class TestRunEpisode:
    """Tests for agent.run_episode method."""

    def test_returns_executor_result(self) -> None:
        """run_episode should return the executor's response."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        result = agent.run_episode(TEST_TASK, StubExecutor(), StubReward())
        assert result == FIXED_RESPONSE

    def test_updates_q_table(self) -> None:
        """run_episode should update Q-values."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS, exploration_rate=0.0)
        agent.run_episode(TEST_TASK, StubExecutor(), StubReward())
        # Q-value should be non-zero after one episode
        q_values = agent.q_table.get_state_actions(TEST_TASK)
        assert any(v > 0.0 for v in q_values.values())

    def test_stores_experience(self) -> None:
        """run_episode should store the episode in the buffer."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.run_episode(TEST_TASK, StubExecutor(), StubReward())
        assert agent.episode_count == 1

    def test_tracks_metrics(self) -> None:
        """run_episode should update metrics."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.run_episode(TEST_TASK, StubExecutor(), StubReward())
        metrics = agent.get_metrics()
        assert metrics["episode_count"] == 1
        assert metrics["cumulative_reward"] == FIXED_REWARD
