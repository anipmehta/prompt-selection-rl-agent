"""Unit tests for MetricsTracker in isolation."""

import pytest

from src.metrics import MetricsTracker

# Test constants
POSITIVE_REWARD = 0.8
NEGATIVE_REWARD = -0.3
ZERO_REWARD = 0.0
TEST_ACTION_A = "Be concise"
TEST_ACTION_B = "Be detailed"
EXPLORATION_RATE = 0.5


class TestMetricsTrackerInit:
    """Tests for initial state."""

    def test_episode_count_starts_zero(self) -> None:
        """New tracker should have zero episodes."""
        tracker = MetricsTracker()
        assert tracker.episode_count == 0

    def test_cumulative_reward_starts_zero(self) -> None:
        """New tracker should have zero cumulative reward."""
        tracker = MetricsTracker()
        assert tracker.cumulative_reward == 0.0

    def test_selection_counts_starts_empty(self) -> None:
        """New tracker should have empty selection counts."""
        tracker = MetricsTracker()
        assert not tracker.prompt_selection_counts


class TestRecordEpisode:
    """Tests for record_episode method."""

    def test_increments_episode_count(self) -> None:
        """Each call should increment episode_count by 1."""
        tracker = MetricsTracker()
        tracker.record_episode(POSITIVE_REWARD)
        assert tracker.episode_count == 1

    def test_accumulates_reward(self) -> None:
        """Cumulative reward should sum all recorded rewards."""
        tracker = MetricsTracker()
        tracker.record_episode(POSITIVE_REWARD)
        tracker.record_episode(NEGATIVE_REWARD)
        expected = POSITIVE_REWARD + NEGATIVE_REWARD
        assert tracker.cumulative_reward == pytest.approx(expected)

    def test_handles_zero_reward(self) -> None:
        """Zero reward should still increment count."""
        tracker = MetricsTracker()
        tracker.record_episode(ZERO_REWARD)
        assert tracker.episode_count == 1
        assert tracker.cumulative_reward == 0.0


class TestRecordSelection:
    """Tests for record_selection method."""

    def test_tracks_single_selection(self) -> None:
        """First selection should set count to 1."""
        tracker = MetricsTracker()
        tracker.record_selection(TEST_ACTION_A)
        assert tracker.prompt_selection_counts[TEST_ACTION_A] == 1

    def test_increments_existing_count(self) -> None:
        """Repeated selections should increment the count."""
        tracker = MetricsTracker()
        tracker.record_selection(TEST_ACTION_A)
        tracker.record_selection(TEST_ACTION_A)
        assert tracker.prompt_selection_counts[TEST_ACTION_A] == 2

    def test_tracks_multiple_actions(self) -> None:
        """Different actions should be tracked independently."""
        tracker = MetricsTracker()
        tracker.record_selection(TEST_ACTION_A)
        tracker.record_selection(TEST_ACTION_B)
        assert tracker.prompt_selection_counts[TEST_ACTION_A] == 1
        assert tracker.prompt_selection_counts[TEST_ACTION_B] == 1


class TestGetMetrics:
    """Tests for get_metrics method."""

    def test_returns_all_keys(self) -> None:
        """get_metrics should return all expected keys."""
        tracker = MetricsTracker()
        metrics = tracker.get_metrics(EXPLORATION_RATE)
        expected_keys = {
            "episode_count",
            "cumulative_reward",
            "average_reward",
            "exploration_rate",
            "prompt_selection_counts",
        }
        assert set(metrics.keys()) == expected_keys

    def test_average_reward_zero_episodes(self) -> None:
        """Average reward should be 0.0 when no episodes recorded."""
        tracker = MetricsTracker()
        metrics = tracker.get_metrics(EXPLORATION_RATE)
        assert metrics["average_reward"] == 0.0

    def test_average_reward_calculated(self) -> None:
        """Average reward should be cumulative / episode_count."""
        tracker = MetricsTracker()
        tracker.record_episode(POSITIVE_REWARD)
        tracker.record_episode(NEGATIVE_REWARD)
        metrics = tracker.get_metrics(EXPLORATION_RATE)
        expected_avg = (POSITIVE_REWARD + NEGATIVE_REWARD) / 2
        assert metrics["average_reward"] == pytest.approx(expected_avg)

    def test_includes_exploration_rate(self) -> None:
        """get_metrics should include the passed exploration_rate."""
        tracker = MetricsTracker()
        metrics = tracker.get_metrics(EXPLORATION_RATE)
        assert metrics["exploration_rate"] == EXPLORATION_RATE

    def test_selection_counts_is_copy(self) -> None:
        """Returned counts should be a copy, not a reference."""
        tracker = MetricsTracker()
        tracker.record_selection(TEST_ACTION_A)
        metrics = tracker.get_metrics(EXPLORATION_RATE)
        metrics["prompt_selection_counts"]["tampered"] = 999
        assert "tampered" not in tracker.prompt_selection_counts
