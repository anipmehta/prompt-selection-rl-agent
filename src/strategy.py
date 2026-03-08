"""
Learning strategies for the RL agent.

Strategies encapsulate the learning algorithm (how to select actions
and update knowledge) while the Agent handles lifecycle concerns
(buffer, mode, metrics).
"""

import random
from abc import ABC, abstractmethod
from typing import Dict, List

from .q_table import QTable


class BaseLearningStrategy(ABC):
    """
    Abstract base class for learning strategies.

    A strategy defines how the agent selects actions and updates
    its knowledge from rewards. Implementations can use different
    RL algorithms (Q-learning, bandits, SARSA, etc.) while sharing
    the same agent lifecycle.
    """

    @abstractmethod
    def select_action(
        self, state: str, prompts: List[str], exploration_rate: float
    ) -> str:
        """
        Select an action for the given state.

        Args:
            state: Current task context
            prompts: Available prompt options
            exploration_rate: Current epsilon for exploration

        Returns:
            Selected prompt text
        """

    @abstractmethod
    def update(self, state: str, action: str, reward: float) -> None:
        """
        Update knowledge based on received reward.

        Args:
            state: State where action was taken
            action: Action (prompt) that was selected
            reward: Reward received
        """


    @abstractmethod
    def get_q_values(self, state: str) -> Dict[str, float]:
        """
        Get current value estimates for all actions in a state.

        Args:
            state: State to query

        Returns:
            Dictionary mapping actions to value estimates
        """

    @abstractmethod
    def get_table(self) -> Dict[str, Dict[str, float]]:
        """
        Export the full knowledge table as a nested dictionary.

        Returns:
            Complete state-action value structure
        """


class QLearningStrategy(BaseLearningStrategy):
    """
    Q-learning with ε-greedy exploration.

    Uses a tabular Q-table to store state-action values and applies
    the simplified Q-learning update rule (γ=0):
        Q(s,a) ← Q(s,a) + α[r - Q(s,a)]
    """

    # Default hyperparameters
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_DISCOUNT_FACTOR = 0.0

    def __init__(
        self,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
    ):
        """
        Initialize Q-learning strategy.

        Args:
            learning_rate: Q-learning alpha (0.0-1.0)
            discount_factor: Q-learning gamma (0.0-1.0)
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = QTable()

    def select_action(
        self, state: str, prompts: List[str], exploration_rate: float
    ) -> str:
        """
        Select action using ε-greedy strategy.

        With probability ε, explore (random). Otherwise, exploit
        (pick the action with the highest Q-value).
        """
        if random.random() > exploration_rate:
            return self._exploit(state, prompts)
        return self._explore(prompts)

    def _exploit(self, state: str, prompts: List[str]) -> str:
        """Select the prompt with the highest Q-value for the given state."""
        state_actions = self.q_table.get_state_actions(state)
        if not state_actions:
            return random.choice(prompts)
        return max(state_actions, key=state_actions.get)

    @staticmethod
    def _explore(prompts: List[str]) -> str:
        """Select a random prompt."""
        return random.choice(prompts)

    def update(self, state: str, action: str, reward: float) -> None:
        """
        Apply Q-learning update rule.

        Q(s,a) ← Q(s,a) + α[r - Q(s,a)]
        """
        current_q = self.q_table.get(state, action)
        new_q = current_q + self.learning_rate * (reward - current_q)
        self.q_table.set(state, action, new_q)

    def get_q_values(self, state: str) -> Dict[str, float]:
        """Get Q-values for all actions in a state."""
        return self.q_table.get_state_actions(state)

    def get_table(self) -> Dict[str, Dict[str, float]]:
        """Export the full Q-table."""
        return self.q_table.to_dict()
