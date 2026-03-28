"""Shared JSON persistence utilities."""

import json
from typing import Any

from .errors import PersistenceError


def save_json(data: Any, filepath: str) -> None:
    """
    Serialize data to a JSON file.

    Args:
        data: Data to serialize (must be JSON-serializable)
        filepath: Path to the output JSON file

    Raises:
        PersistenceError: If the file cannot be written
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as exc:
        raise PersistenceError(f"Failed to save to {filepath}: {exc}") from exc


def load_json(filepath: str) -> Any:
    """
    Deserialize data from a JSON file.

    Args:
        filepath: Path to the input JSON file

    Returns:
        Parsed JSON data

    Raises:
        PersistenceError: If the file cannot be read or contains invalid JSON
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise PersistenceError(f"File not found: {filepath}") from exc
    except json.JSONDecodeError as exc:
        raise PersistenceError(f"Invalid JSON in {filepath}: {exc}") from exc
    except OSError as exc:
        raise PersistenceError(f"Failed to load from {filepath}: {exc}") from exc
