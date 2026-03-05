"""
Unit tests for Q-Table implementation.
"""

from src.q_table import QTable

# Test constants - defined at module level for reuse within this file
STATE_1 = "state1"
STATE_2 = "state2"
UNSEEN_STATE = "unseen_state"

ACTION_1 = "action1"
ACTION_2 = "action2"
ACTION_3 = "action3"

Q_VALUE_HIGH = 0.75
Q_VALUE_MID = 0.42
Q_VALUE_LOW = 0.33
Q_VALUE_UPDATED = 0.85
Q_VALUE_MODIFIED = 0.91
Q_VALUE_TAMPERED = 0.99

INITIAL_Q_VALUE = 0.0


class TestQTableBasics:
    """Test basic Q-table operations."""

    def test_initialization(self):
        """Test that Q-table initializes empty."""
        q_table = QTable()
        assert len(q_table) == 0

    def test_get_unseen_returns_zero(self):
        """Test that unseen state-action pairs return 0.0."""
        q_table = QTable()
        assert q_table.get(STATE_1, ACTION_1) == INITIAL_Q_VALUE

    def test_set_and_get(self):
        """Test setting and retrieving Q-values."""
        q_table = QTable()
        q_table.set(STATE_1, ACTION_1, Q_VALUE_HIGH)
        assert q_table.get(STATE_1, ACTION_1) == Q_VALUE_HIGH

    def test_multiple_actions_same_state(self):
        """Test multiple actions for the same state."""
        q_table = QTable()
        q_table.set(STATE_1, ACTION_1, Q_VALUE_HIGH)
        q_table.set(STATE_1, ACTION_2, Q_VALUE_MID)

        assert q_table.get(STATE_1, ACTION_1) == Q_VALUE_HIGH
        assert q_table.get(STATE_1, ACTION_2) == Q_VALUE_MID

    def test_multiple_states(self):
        """Test multiple states with different actions."""
        q_table = QTable()
        q_table.set(STATE_1, ACTION_1, Q_VALUE_HIGH)
        q_table.set(STATE_2, ACTION_1, Q_VALUE_MID)

        assert q_table.get(STATE_1, ACTION_1) == Q_VALUE_HIGH
        assert q_table.get(STATE_2, ACTION_1) == Q_VALUE_MID

    def test_update_existing_value(self):
        """Test updating an existing Q-value."""
        q_table = QTable()
        q_table.set(STATE_1, ACTION_1, Q_VALUE_HIGH)
        q_table.set(STATE_1, ACTION_1, Q_VALUE_UPDATED)

        assert q_table.get(STATE_1, ACTION_1) == Q_VALUE_UPDATED


class TestQTableStateActions:
    """Test retrieving all actions for a state."""

    def test_get_state_actions_unseen_state(self):
        """Test getting actions for unseen state returns empty dict."""
        q_table = QTable()
        actions = q_table.get_state_actions(UNSEEN_STATE)
        assert actions == {}

    def test_get_state_actions_with_values(self):
        """Test getting all actions for a state."""
        q_table = QTable()
        q_table.set(STATE_1, ACTION_1, Q_VALUE_HIGH)
        q_table.set(STATE_1, ACTION_2, Q_VALUE_MID)
        q_table.set(STATE_1, ACTION_3, Q_VALUE_MODIFIED)

        actions = q_table.get_state_actions(STATE_1)
        assert actions == {
            ACTION_1: Q_VALUE_HIGH,
            ACTION_2: Q_VALUE_MID,
            ACTION_3: Q_VALUE_MODIFIED
        }

    def test_get_state_actions_returns_copy(self):
        """Test that get_state_actions returns a copy, not reference."""
        q_table = QTable()
        q_table.set(STATE_1, ACTION_1, Q_VALUE_HIGH)

        actions = q_table.get_state_actions(STATE_1)
        actions[ACTION_1] = Q_VALUE_TAMPERED  # Modify the returned dict

        # Original should be unchanged
        assert q_table.get(STATE_1, ACTION_1) == Q_VALUE_HIGH


class TestQTableSerialization:
    """Test Q-table serialization and deserialization."""

    def test_to_dict_empty(self):
        """Test exporting empty Q-table."""
        q_table = QTable()
        assert q_table.to_dict() == {}

    def test_to_dict_with_values(self):
        """Test exporting Q-table with values."""
        q_table = QTable()
        q_table.set(STATE_1, ACTION_1, Q_VALUE_HIGH)
        q_table.set(STATE_1, ACTION_2, Q_VALUE_MID)
        q_table.set(STATE_2, ACTION_1, Q_VALUE_LOW)

        data = q_table.to_dict()
        assert data == {
            STATE_1: {
                ACTION_1: Q_VALUE_HIGH,
                ACTION_2: Q_VALUE_MID
            },
            STATE_2: {
                ACTION_1: Q_VALUE_LOW
            }
        }

    def test_from_dict(self):
        """Test importing Q-table from dict."""
        q_table = QTable()
        data = {
            STATE_1: {
                ACTION_1: Q_VALUE_HIGH,
                ACTION_2: Q_VALUE_MID
            },
            STATE_2: {
                ACTION_1: Q_VALUE_LOW
            }
        }
        q_table.from_dict(data)

        assert q_table.get(STATE_1, ACTION_1) == Q_VALUE_HIGH
        assert q_table.get(STATE_1, ACTION_2) == Q_VALUE_MID
        assert q_table.get(STATE_2, ACTION_1) == Q_VALUE_LOW

    def test_round_trip(self):
        """Test that export then import preserves Q-table."""
        q_table1 = QTable()
        q_table1.set(STATE_1, ACTION_1, Q_VALUE_HIGH)
        q_table1.set(STATE_2, ACTION_2, Q_VALUE_MID)

        data = q_table1.to_dict()

        q_table2 = QTable()
        q_table2.from_dict(data)

        assert q_table2.to_dict() == q_table1.to_dict()
