"""Metrics tracking for the RL agent."""

from typing import Dict, Union


class MetricsTracker:
    """
    Tracks agent performance metrics.

    Records episode counts, cumulative rewards, and prompt
    selection distribution for observability.
    """

    def __init__(self) -> None:
        """Initialize all metrics to zero."""
        self.episode_count: int = 0
        self.cumulative_reward: float = 0.0
        self.prompt_selection_counts: Dict[str, int] = {}

    def record_episode(self, reward: float) -> None:
        """
        Record a completed episode.

        Args:
            reward: Reward received for the episode
        """
        self.episode_count += 1
        self.cumulative_reward += reward

    def record_selection(self, action: str) -> None:
        """
        Record a prompt selection.

        Args:
            action: The prompt that was selected
        """
        self.prompt_selection_counts[action] = (
            self.prompt_selection_counts.get(action, 0) + 1
        )

    def get_metrics(self, exploration_rate: float) -> Dict[str, Union[int, float, Dict[str, int]]]:
        """
        Return current performance metrics.

        Args:
            exploration_rate: Current exploration rate from the agent

        Returns:
            Dictionary with episode_count, cumulative_reward,
            average_reward, exploration_rate, and prompt_selection_counts.
        """
        average_reward = (
            self.cumulative_reward / self.episode_count
            if self.episode_count > 0
            else 0.0
        )
        return {
            "episode_count": self.episode_count,
            "cumulative_reward": self.cumulative_reward,
            "average_reward": average_reward,
            "exploration_rate": exploration_rate,
            "prompt_selection_counts": dict(self.prompt_selection_counts),
        }
