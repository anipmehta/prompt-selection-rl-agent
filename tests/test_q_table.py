"""
Unit tests for Q-Table implementation.
"""

import pytest
from src.q_table import QTable


class TestQTableBasics:
    """Test basic Q-table operations."""
    
    def test_initialization(self):
        """Test that Q-table initializes empty."""
        q_table = QTable()
        assert len(q_table) == 0
    
    def test_get_unseen_returns_zero(self):
        """Test that unseen state-action pairs return 0.0."""
        q_table = QTable()
        assert q_table.get("state1", "action1") == 0.0
    
    def test_set_and_get(self):
        """Test setting and retrieving Q-values."""
        q_table = QTable()
        q_table.set("state1", "action1", 0.75)
        assert q_table.get("state1", "action1") == 0.75
    
    def test_multiple_actions_same_state(self):
        """Test multiple actions for the same state."""
        q_table = QTable()
        q_table.set("state1", "action1", 0.75)
        q_table.set("state1", "action2", 0.42)
        
        assert q_table.get("state1", "action1") == 0.75
        assert q_table.get("state1", "action2") == 0.42
    
    def test_multiple_states(self):
        """Test multiple states with different actions."""
        q_table = QTable()
        q_table.set("state1", "action1", 0.75)
        q_table.set("state2", "action1", 0.42)
        
        assert q_table.get("state1", "action1") == 0.75
        assert q_table.get("state2", "action1") == 0.42
    
    def test_update_existing_value(self):
        """Test updating an existing Q-value."""
        q_table = QTable()
        q_table.set("state1", "action1", 0.75)
        q_table.set("state1", "action1", 0.85)
        
        assert q_table.get("state1", "action1") == 0.85


class TestQTableStateActions:
    """Test retrieving all actions for a state."""
    
    def test_get_state_actions_unseen_state(self):
        """Test getting actions for unseen state returns empty dict."""
        q_table = QTable()
        actions = q_table.get_state_actions("unseen_state")
        assert actions == {}
    
    def test_get_state_actions_with_values(self):
        """Test getting all actions for a state."""
        q_table = QTable()
        q_table.set("state1", "action1", 0.75)
        q_table.set("state1", "action2", 0.42)
        q_table.set("state1", "action3", 0.91)
        
        actions = q_table.get_state_actions("state1")
        assert actions == {
            "action1": 0.75,
            "action2": 0.42,
            "action3": 0.91
        }
    
    def test_get_state_actions_returns_copy(self):
        """Test that get_state_actions returns a copy, not reference."""
        q_table = QTable()
        q_table.set("state1", "action1", 0.75)
        
        actions = q_table.get_state_actions("state1")
        actions["action1"] = 0.99  # Modify the returned dict
        
        # Original should be unchanged
        assert q_table.get("state1", "action1") == 0.75


class TestQTableSerialization:
    """Test Q-table serialization and deserialization."""
    
    def test_to_dict_empty(self):
        """Test exporting empty Q-table."""
        q_table = QTable()
        assert q_table.to_dict() == {}
    
    def test_to_dict_with_values(self):
        """Test exporting Q-table with values."""
        q_table = QTable()
        q_table.set("state1", "action1", 0.75)
        q_table.set("state1", "action2", 0.42)
        q_table.set("state2", "action1", 0.33)
        
        data = q_table.to_dict()
        assert data == {
            "state1": {"action1": 0.75, "action2": 0.42},
            "state2": {"action1": 0.33}
        }
    
    def test_from_dict(self):
        """Test importing Q-table from dict."""
        q_table = QTable()
        data = {
            "state1": {"action1": 0.75, "action2": 0.42},
            "state2": {"action1": 0.33}
        }
        q_table.from_dict(data)
        
        assert q_table.get("state1", "action1") == 0.75
        assert q_table.get("state1", "action2") == 0.42
        assert q_table.get("state2", "action1") == 0.33
    
    def test_round_trip(self):
        """Test that export then import preserves Q-table."""
        q_table1 = QTable()
        q_table1.set("state1", "action1", 0.75)
        q_table1.set("state2", "action2", 0.42)
        
        data = q_table1.to_dict()
        
        q_table2 = QTable()
        q_table2.from_dict(data)
        
        assert q_table2.to_dict() == q_table1.to_dict()
