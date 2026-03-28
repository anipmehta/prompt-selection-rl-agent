"""Unit tests for policy serialization (src/policy.py)."""

import json
import os

import pytest

from src.errors import PersistenceError
from src.policy import load_policy, save_policy

# Test constants
POLICY_FILE = "test_policy.json"
VALID_STATE = {
    "q_table": {"state1": {"action1": 0.5}},
    "config": {
        "learning_rate": 0.1,
        "discount_factor": 0.0,
        "exploration_rate": 0.8,
        "decay_rate": 0.995,
        "min_exploration": 0.01,
    },
    "metrics": {
        "episode_count": 10,
        "cumulative_reward": 7.5,
        "prompt_selection_counts": {"action1": 10},
    },
    "mode": "training",
    "prompts": ["action1", "action2"],
}


@pytest.fixture(autouse=True)
def cleanup_files():
    """Remove test files after each test."""
    yield
    if os.path.exists(POLICY_FILE):
        os.remove(POLICY_FILE)


class TestSavePolicy:
    """Tests for save_policy function."""

    def test_saves_state_to_file(self) -> None:
        """save_policy should write state dict to JSON file."""
        save_policy(VALID_STATE, POLICY_FILE)
        with open(POLICY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data == VALID_STATE

    def test_saves_all_required_keys(self) -> None:
        """Saved file should contain all five top-level keys."""
        save_policy(VALID_STATE, POLICY_FILE)
        with open(POLICY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        expected_keys = {"q_table", "config", "metrics", "mode", "prompts"}
        assert set(data.keys()) == expected_keys


class TestLoadPolicy:
    """Tests for load_policy function."""

    def test_loads_valid_state(self) -> None:
        """load_policy should return the saved state dict."""
        save_policy(VALID_STATE, POLICY_FILE)
        data = load_policy(POLICY_FILE)
        assert data == VALID_STATE

    def test_missing_keys_raises_persistence_error(self) -> None:
        """Loading JSON with missing required keys should raise PersistenceError."""
        with open(POLICY_FILE, "w", encoding="utf-8") as f:
            json.dump({"incomplete": True}, f)
        with pytest.raises(PersistenceError, match="Missing keys"):
            load_policy(POLICY_FILE)

    def test_non_dict_raises_persistence_error(self) -> None:
        """Loading a JSON list instead of dict should raise PersistenceError."""
        with open(POLICY_FILE, "w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        with pytest.raises(PersistenceError):
            load_policy(POLICY_FILE)


class TestPolicyRoundTrip:
    """Tests for save → load round-trip."""

    def test_round_trip_preserves_state(self) -> None:
        """save then load should return identical state."""
        save_policy(VALID_STATE, POLICY_FILE)
        result = load_policy(POLICY_FILE)
        assert result == VALID_STATE

    def test_round_trip_inference_mode(self) -> None:
        """Round-trip should preserve inference mode state."""
        inference_state = {**VALID_STATE, "mode": "inference"}
        inference_state["config"] = {**VALID_STATE["config"], "exploration_rate": 0.0}
        save_policy(inference_state, POLICY_FILE)
        result = load_policy(POLICY_FILE)
        assert result["mode"] == "inference"
        assert result["config"]["exploration_rate"] == 0.0
