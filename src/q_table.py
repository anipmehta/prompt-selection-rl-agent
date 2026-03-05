"""
Q-Table implementation for storing state-action values.

The Q-table is a nested dictionary structure that maps (state, action) pairs
to Q-values (expected cumulative rewards). Unseen state-action pairs default to 0.0.
"""

from typing import Dict


class QTable:
    """
    Q-Table for storing and retrieving state-action values.
    
    Structure: Dict[str, Dict[str, float]]
    - Outer key: State representation (string)
    - Inner key: Action/prompt identifier (string)
    - Value: Q-value (float)
    """
    
    def __init__(self):
        """Initialize an empty Q-table."""
        self._table: Dict[str, Dict[str, float]] = {}
    
    def get(self, state: str, action: str) -> float:
        """
        Get Q-value for a state-action pair.
        
        Args:
            state: State representation
            action: Action/prompt identifier
            
        Returns:
            Q-value for the pair, or 0.0 if unseen
        """
        if state not in self._table:
            return 0.0
        return self._table[state].get(action, 0.0)
    
    def set(self, state: str, action: str, value: float) -> None:
        """
        Set Q-value for a state-action pair.
        
        Args:
            state: State representation
            action: Action/prompt identifier
            value: Q-value to store
        """
        if state not in self._table:
            self._table[state] = {}
        self._table[state][action] = value
    
    def get_state_actions(self, state: str) -> Dict[str, float]:
        """
        Get all action-value pairs for a given state.
        
        Args:
            state: State representation
            
        Returns:
            Dictionary mapping actions to Q-values for this state,
            or empty dict if state is unseen
        """
        return self._table.get(state, {}).copy()
    
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Export Q-table as a nested dictionary.
        
        Returns:
            Complete Q-table structure
        """
        return {state: actions.copy() for state, actions in self._table.items()}
    
    def from_dict(self, data: Dict[str, Dict[str, float]]) -> None:
        """
        Import Q-table from a nested dictionary.
        
        Args:
            data: Q-table structure to import
        """
        self._table = {state: actions.copy() for state, actions in data.items()}
    
    def __len__(self) -> int:
        """Return the number of states in the Q-table."""
        return len(self._table)
    
    def __repr__(self) -> str:
        """Return string representation of Q-table."""
        return f"QTable(states={len(self._table)})"
