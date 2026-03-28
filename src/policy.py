"""Policy serialization for saving and loading agent state."""

from typing import Any, Dict

from .errors import PersistenceError
from .persistence import load_json, save_json

# Serialization keys
KEY_Q_TABLE = "q_table"
KEY_CONFIG = "config"
KEY_METRICS = "metrics"
KEY_MODE = "mode"
KEY_PROMPTS = "prompts"

_REQUIRED_KEYS = {KEY_Q_TABLE, KEY_CONFIG, KEY_METRICS, KEY_MODE, KEY_PROMPTS}


def save_policy(state: Dict[str, Any], filepath: str) -> None:
    """
    Save agent state to a JSON file.

    Args:
        state: Dictionary with q_table, config, metrics, mode, prompts
        filepath: Path to the output JSON file

    Raises:
        PersistenceError: If the file cannot be written
    """
    save_json(state, filepath)


def load_policy(filepath: str) -> Dict[str, Any]:
    """
    Load agent state from a JSON file.

    Args:
        filepath: Path to the input JSON file

    Returns:
        Dictionary with q_table, config, metrics, mode, prompts

    Raises:
        PersistenceError: If the file cannot be read or contains invalid data
    """
    data = load_json(filepath)
    try:
        missing = _REQUIRED_KEYS - set(data.keys())
        if missing:
            raise PersistenceError(
                f"Invalid policy file structure in {filepath}: Missing keys: {missing}"
            )
    except (AttributeError, TypeError) as exc:
        raise PersistenceError(
            f"Invalid policy file structure in {filepath}: {exc}"
        ) from exc
    return data
