"""
RL Agent implementing Q-learning for prompt selection.
"""

import random
from typing import List

from .errors import ConfigurationError, ValidationError
from .q_table import QTable


class RLAgent:
    """
    Reinforcement learning agent that learns to select optimal prompts
    using Q-learning with ε-greedy exploration.
    """

    # pylint: disable=too-many-instance-attributes

    # Parameter range constants
    PARAM_MIN = 0.0
    PARAM_MAX = 1.0

    # Default hyperparameters
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_DISCOUNT_FACTOR = 0.0
    DEFAULT_EXPLORATION_RATE = 1.0
    DEFAULT_DECAY_RATE = 0.995
    DEFAULT_MIN_EXPLORATION = 0.01

    # Mode constants
    MODE_TRAINING = "training"
    MODE_INFERENCE = "inference"

    # Reward range constants
    REWARD_MIN = -1.0
    REWARD_MAX = 1.0

    def __init__(
        self,
        prompts: List[str],
        learning_rate: float = DEFAULT_LEARNING_RATE,
        discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
        exploration_rate: float = DEFAULT_EXPLORATION_RATE,
        decay_rate: float = DEFAULT_DECAY_RATE,
        min_exploration: float = DEFAULT_MIN_EXPLORATION,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """
        Initialize the RL agent with configuration parameters.

        Args:
            prompts: List of available prompt templates
            learning_rate: Q-learning alpha (0.0-1.0)
            discount_factor: Q-learning gamma (0.0-1.0)
            exploration_rate: Initial epsilon for ε-greedy (0.0-1.0)
            decay_rate: Multiplicative decay per episode (0.0-1.0)
            min_exploration: Minimum epsilon threshold
        """
        self._validate_prompts(prompts)
        self._validate_param("learning_rate", learning_rate)
        self._validate_param("discount_factor", discount_factor)
        self._validate_param("exploration_rate", exploration_rate)
        self._validate_param("decay_rate", decay_rate)
        self._validate_param("min_exploration", min_exploration)

        self.prompts = list(prompts)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.decay_rate = decay_rate
        self.min_exploration = min_exploration
        self.q_table = QTable()
        self.mode = self.MODE_TRAINING

    def _validate_param(self, name: str, value: float) -> None:
        """Validate that a parameter is within [0.0, 1.0]."""
        if not isinstance(value, (int, float)) or value < self.PARAM_MIN or value > self.PARAM_MAX:
            raise ConfigurationError(
                f"Invalid {name}: {value}. Must be in range "
                f"[{self.PARAM_MIN}, {self.PARAM_MAX}]"
            )

    @staticmethod
    def _validate_prompts(prompts: List[str]) -> None:
        """Validate that prompts list is not empty."""
        if not prompts:
            raise ConfigurationError("Prompts list must not be empty")

    def select_action(self, state: str) -> str:
        """
        Select a prompt for the given state using ε-greedy strategy.

        Training mode: explore (random) with probability ε, exploit (best Q) otherwise.
        Inference mode: always exploit (best Q-value).

        Args:
            state: Current task context

        Returns:
            Selected prompt text
        """
        if self.mode == self.MODE_INFERENCE or random.random() > self.exploration_rate:
            return self._exploit(state)
        return self._explore()

    def _exploit(self, state: str) -> str:
        """Select the prompt with the highest Q-value for the given state."""
        state_actions = self.q_table.get_state_actions(state)
        if not state_actions:
            return random.choice(self.prompts)

        best_action = max(state_actions, key=state_actions.get)
        return best_action

    def _explore(self) -> str:
        """Select a random prompt."""
        return random.choice(self.prompts)

    def update(self, state: str, action: str, reward: float) -> None:
        """
        Update Q-value based on received reward (training mode only).

        Applies simplified Q-learning (γ=0): Q(s,a) ← Q(s,a) + α[r - Q(s,a)]

        Args:
            state: State where action was taken
            action: Action (prompt) that was selected
            reward: Reward received (-1.0 to 1.0)
        """
        if reward < self.REWARD_MIN or reward > self.REWARD_MAX:
            raise ValidationError(
                f"Invalid reward: {reward}. Must be in range "
                f"[{self.REWARD_MIN}, {self.REWARD_MAX}]"
            )

        if self.mode == self.MODE_INFERENCE:
            return

        current_q = self.q_table.get(state, action)
        new_q = current_q + self.learning_rate * (reward - current_q)
        self.q_table.set(state, action, new_q)
