"""Unit tests for persistence utilities (save_json / load_json)."""

import json
import os

import pytest

from src.errors import PersistenceError
from src.persistence import load_json, save_json

# Test constants
TEST_FILE = "test_persistence.json"
SAMPLE_DICT = {"key": "value", "number": 42}
SAMPLE_LIST = [1, 2, 3]
NESTED_DATA = {"outer": {"inner": [1, 2, 3]}}


@pytest.fixture(autouse=True)
def cleanup_files():
    """Remove test files after each test."""
    yield
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)


class TestSaveJson:
    """Tests for save_json function."""

    def test_creates_file(self) -> None:
        """save_json should create a file at the given path."""
        save_json(SAMPLE_DICT, TEST_FILE)
        assert os.path.exists(TEST_FILE)

    def test_writes_valid_json(self) -> None:
        """Saved file should contain valid JSON."""
        save_json(SAMPLE_DICT, TEST_FILE)
        with open(TEST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data == SAMPLE_DICT

    def test_saves_list(self) -> None:
        """save_json should handle list data."""
        save_json(SAMPLE_LIST, TEST_FILE)
        with open(TEST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data == SAMPLE_LIST

    def test_saves_nested_data(self) -> None:
        """save_json should handle nested structures."""
        save_json(NESTED_DATA, TEST_FILE)
        with open(TEST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data == NESTED_DATA

    def test_invalid_path_raises_persistence_error(self) -> None:
        """Writing to an invalid path should raise PersistenceError."""
        with pytest.raises(PersistenceError, match="Failed to save"):
            save_json(SAMPLE_DICT, "/nonexistent/dir/file.json")


class TestLoadJson:
    """Tests for load_json function."""

    def test_loads_dict(self) -> None:
        """load_json should return parsed dict."""
        save_json(SAMPLE_DICT, TEST_FILE)
        data = load_json(TEST_FILE)
        assert data == SAMPLE_DICT

    def test_loads_list(self) -> None:
        """load_json should return parsed list."""
        save_json(SAMPLE_LIST, TEST_FILE)
        data = load_json(TEST_FILE)
        assert data == SAMPLE_LIST

    def test_file_not_found_raises_persistence_error(self) -> None:
        """Loading a nonexistent file should raise PersistenceError."""
        with pytest.raises(PersistenceError, match="not found"):
            load_json("nonexistent.json")

    def test_invalid_json_raises_persistence_error(self) -> None:
        """Loading invalid JSON should raise PersistenceError."""
        with open(TEST_FILE, "w", encoding="utf-8") as f:
            f.write("not valid json {{{")
        with pytest.raises(PersistenceError, match="Invalid JSON"):
            load_json(TEST_FILE)

    def test_round_trip(self) -> None:
        """save then load should return identical data."""
        save_json(NESTED_DATA, TEST_FILE)
        result = load_json(TEST_FILE)
        assert result == NESTED_DATA
