"""Unit tests for ExperienceBuffer class."""

from src.experience_buffer import ExperienceBuffer

# Test data constants
STATE_A = "task_a"
STATE_B = "task_b"
ACTION_A = "prompt_1"
ACTION_B = "prompt_2"
REWARD_POSITIVE = 0.8
REWARD_NEGATIVE = -0.5


class TestExperienceBufferEmpty:
    """Tests for empty buffer behavior."""

    def test_new_buffer_has_size_zero(self) -> None:
        """A freshly created buffer should have size 0."""
        buf = ExperienceBuffer()
        assert buf.size() == 0

    def test_get_all_on_empty_buffer_returns_empty_list(self) -> None:
        """get_all on an empty buffer should return an empty list."""
        buf = ExperienceBuffer()
        assert not buf.get_all()


class TestExperienceBufferAdd:
    """Tests for adding episodes and retrieving them."""

    def test_add_single_episode(self) -> None:
        """Adding one episode should make it retrievable."""
        buf = ExperienceBuffer()
        buf.add(STATE_A, ACTION_A, REWARD_POSITIVE)

        episodes = buf.get_all()
        assert len(episodes) == 1
        assert episodes[0] == (STATE_A, ACTION_A, REWARD_POSITIVE)

    def test_add_multiple_episodes_preserves_order(self) -> None:
        """Episodes should be returned in insertion order."""
        buf = ExperienceBuffer()
        buf.add(STATE_A, ACTION_A, REWARD_POSITIVE)
        buf.add(STATE_B, ACTION_B, REWARD_NEGATIVE)

        episodes = buf.get_all()
        assert len(episodes) == 2
        assert episodes[0] == (STATE_A, ACTION_A, REWARD_POSITIVE)
        assert episodes[1] == (STATE_B, ACTION_B, REWARD_NEGATIVE)

    def test_get_all_returns_copy(self) -> None:
        """get_all should return a copy, not the internal list."""
        buf = ExperienceBuffer()
        buf.add(STATE_A, ACTION_A, REWARD_POSITIVE)

        episodes = buf.get_all()
        episodes.clear()

        assert buf.size() == 1


class TestExperienceBufferClear:
    """Tests for clearing the buffer."""

    def test_clear_empties_buffer(self) -> None:
        """Clearing a buffer with episodes should result in size 0."""
        buf = ExperienceBuffer()
        buf.add(STATE_A, ACTION_A, REWARD_POSITIVE)
        buf.add(STATE_B, ACTION_B, REWARD_NEGATIVE)

        buf.clear()

        assert buf.size() == 0
        assert not buf.get_all()

    def test_clear_on_empty_buffer(self) -> None:
        """Clearing an already-empty buffer should not raise."""
        buf = ExperienceBuffer()
        buf.clear()
        assert buf.size() == 0


class TestExperienceBufferSize:
    """Tests for size tracking."""

    def test_size_increments_on_add(self) -> None:
        """Size should increase by 1 for each added episode."""
        buf = ExperienceBuffer()
        assert buf.size() == 0

        buf.add(STATE_A, ACTION_A, REWARD_POSITIVE)
        assert buf.size() == 1

        buf.add(STATE_B, ACTION_B, REWARD_NEGATIVE)
        assert buf.size() == 2

    def test_size_resets_after_clear(self) -> None:
        """Size should be 0 after clearing."""
        buf = ExperienceBuffer()
        buf.add(STATE_A, ACTION_A, REWARD_POSITIVE)
        buf.clear()
        assert buf.size() == 0
