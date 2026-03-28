"""Unit tests for ExperienceBuffer save/load (Task 17)."""

import json
import os

import pytest

from src.errors import PersistenceError
from src.experience_buffer import ExperienceBuffer

# Test constants
BUFFER_FILE = "test_buffer.json"
TEST_STATE = "summarize this article"
TEST_ACTION = "Be concise"
TEST_REWARD = 0.8
SECOND_STATE = "translate this text"
SECOND_ACTION = "Be detailed"
SECOND_REWARD = -0.3


@pytest.fixture(autouse=True)
def cleanup_files():
    """Remove test files after each test."""
    yield
    if os.path.exists(BUFFER_FILE):
        os.remove(BUFFER_FILE)


class TestBufferSave:
    """Tests for ExperienceBuffer.save — buffer-specific serialization."""

    def test_saves_empty_buffer(self) -> None:
        """Empty buffer should save as empty list."""
        buf = ExperienceBuffer()
        buf.save(BUFFER_FILE)
        with open(BUFFER_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data == []

    def test_saves_episode_as_list(self) -> None:
        """Episodes (tuples) should be serialized as JSON arrays."""
        buf = ExperienceBuffer()
        buf.add(TEST_STATE, TEST_ACTION, TEST_REWARD)
        buf.save(BUFFER_FILE)
        with open(BUFFER_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data[0] == [TEST_STATE, TEST_ACTION, TEST_REWARD]


class TestBufferLoad:
    """Tests for ExperienceBuffer.load — buffer-specific deserialization."""

    def test_restores_episodes_as_tuples(self) -> None:
        """Loaded episodes should be (state, action, reward) tuples."""
        buf = ExperienceBuffer()
        buf.add(TEST_STATE, TEST_ACTION, TEST_REWARD)
        buf.save(BUFFER_FILE)

        new_buf = ExperienceBuffer()
        new_buf.load(BUFFER_FILE)
        episodes = new_buf.get_all()
        assert len(episodes) == 1
        assert episodes[0] == (TEST_STATE, TEST_ACTION, TEST_REWARD)

    def test_replaces_existing_contents(self) -> None:
        """load should replace current buffer, not append."""
        buf = ExperienceBuffer()
        buf.add(TEST_STATE, TEST_ACTION, TEST_REWARD)
        buf.save(BUFFER_FILE)

        new_buf = ExperienceBuffer()
        new_buf.add(SECOND_STATE, SECOND_ACTION, SECOND_REWARD)
        new_buf.load(BUFFER_FILE)
        assert new_buf.size() == 1
        assert new_buf.get_all()[0] == (TEST_STATE, TEST_ACTION, TEST_REWARD)

    def test_invalid_structure_raises_persistence_error(self) -> None:
        """Loading JSON with wrong episode structure should raise PersistenceError."""
        with open(BUFFER_FILE, "w", encoding="utf-8") as f:
            json.dump([{"bad": "structure"}], f)
        buf = ExperienceBuffer()
        with pytest.raises(PersistenceError, match="Invalid buffer file"):
            buf.load(BUFFER_FILE)


class TestBufferRoundTrip:
    """Tests for save → load round-trip consistency."""

    def test_round_trip_multiple_episodes(self) -> None:
        """Save then load should preserve all episodes in order."""
        buf = ExperienceBuffer()
        buf.add(TEST_STATE, TEST_ACTION, TEST_REWARD)
        buf.add(SECOND_STATE, SECOND_ACTION, SECOND_REWARD)
        buf.save(BUFFER_FILE)

        new_buf = ExperienceBuffer()
        new_buf.load(BUFFER_FILE)
        assert new_buf.get_all() == buf.get_all()

    def test_round_trip_empty_buffer(self) -> None:
        """Save then load of empty buffer should produce empty buffer."""
        buf = ExperienceBuffer()
        buf.save(BUFFER_FILE)

        new_buf = ExperienceBuffer()
        new_buf.load(BUFFER_FILE)
        assert new_buf.size() == 0
