"""Experience buffer for storing episodes used in offline batch training."""

from typing import List, Tuple

from .errors import PersistenceError
from .persistence import load_json, save_json


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

    def save(self, filepath: str) -> None:
        """
        Save buffer contents to a JSON file.

        Args:
            filepath: Path to the output JSON file

        Raises:
            PersistenceError: If the file cannot be written
        """
        data = [list(episode) for episode in self._episodes]
        save_json(data, filepath)

    def load(self, filepath: str) -> None:
        """
        Load buffer contents from a JSON file.

        Replaces current buffer contents with loaded data.

        Args:
            filepath: Path to the input JSON file

        Raises:
            PersistenceError: If the file cannot be read or contains invalid data
        """
        data = load_json(filepath)

        try:
            self._episodes = [(str(e[0]), str(e[1]), float(e[2])) for e in data]
        except (IndexError, TypeError, ValueError, KeyError) as exc:
            raise PersistenceError(
                f"Invalid buffer file structure in {filepath}: {exc}"
            ) from exc
