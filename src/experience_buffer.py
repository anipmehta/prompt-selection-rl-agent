"""Experience buffer for storing episodes used in offline batch training."""

from typing import List, Tuple


class ExperienceBuffer:
    """
    Storage for episodes used in offline batch training.

    Each episode is a (state, action, reward) tuple representing
    a single interaction cycle.
    """

    def __init__(self) -> None:
        """Initialize the buffer with an empty episode list."""
        self._episodes: List[Tuple[str, str, float]] = []

    def add(self, state: str, action: str, reward: float) -> None:
        """
        Add an episode to the buffer.

        Args:
            state: State where action was taken
            action: Action (prompt) selected
            reward: Reward received
        """
        self._episodes.append((state, action, reward))

    def get_all(self) -> List[Tuple[str, str, float]]:
        """
        Retrieve all stored episodes.

        Returns:
            List of (state, action, reward) tuples
        """
        return list(self._episodes)

    def clear(self) -> None:
        """Remove all episodes from the buffer."""
        self._episodes.clear()

    def size(self) -> int:
        """Return the number of episodes in the buffer."""
        return len(self._episodes)
