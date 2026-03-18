"""Integration tests for agent metrics wiring (delegates to MetricsTracker)."""

from src.agent import RLAgent

# Test prompts
SAMPLE_PROMPTS = ["Be concise", "Be detailed", "Be creative"]

# Test constants
TEST_STATE = "summarize this article"
TEST_ACTION = SAMPLE_PROMPTS[0]
POSITIVE_REWARD = 0.8


class TestAgentMetricsWiring:
    """Tests that agent correctly delegates to MetricsTracker."""

    def test_store_experience_increments_episode_count(self) -> None:
        """store_experience should delegate to metrics.record_episode."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.store_experience(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)
        assert agent.episode_count == 1

    def test_select_action_tracks_selection(self) -> None:
        """select_action should delegate to metrics.record_selection."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        action = agent.select_action(TEST_STATE)
        assert agent.prompt_selection_counts[action] == 1

    def test_get_metrics_includes_exploration_rate(self) -> None:
        """get_metrics should pass agent's exploration_rate to tracker."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS, exploration_rate=0.5)
        metrics = agent.get_metrics()
        assert metrics["exploration_rate"] == 0.5
