"""
RL Agent for prompt selection with pluggable learning strategies.
"""

from typing import Dict, List, Optional

from .errors import ConfigurationError, ModeError, PersistenceError, ValidationError
from .experience_buffer import ExperienceBuffer
from .metrics import MetricsTracker
from .policy import KEY_CONFIG, KEY_METRICS, KEY_MODE, KEY_PROMPTS, KEY_Q_TABLE
from .policy import load_policy as _load_policy
from .policy import save_policy as _save_policy
from .strategy import BaseLearningStrategy, QLearningStrategy


class RLAgent:
    """
    Reinforcement learning agent that learns to select optimal prompts.

    The agent handles lifecycle concerns (mode, buffer, validation)
    and delegates learning decisions to a pluggable LearningStrategy.
    Default strategy is Q-learning with ε-greedy exploration.
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
        strategy: Optional[BaseLearningStrategy] = None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        """
        Initialize the RL agent with configuration parameters.

        Args:
            prompts: List of available prompt templates
            learning_rate: Alpha for default Q-learning strategy (0.0-1.0)
            discount_factor: Gamma for default Q-learning strategy (0.0-1.0)
            exploration_rate: Initial epsilon for ε-greedy (0.0-1.0)
            decay_rate: Multiplicative decay per episode (0.0-1.0)
            min_exploration: Minimum epsilon threshold
            strategy: Learning strategy (defaults to QLearningStrategy)
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
        self.mode = self.MODE_TRAINING
        self._buffer = ExperienceBuffer()

        # Use provided strategy or default to Q-learning
        if strategy is not None:
            self._strategy = strategy
        else:
            self._strategy = QLearningStrategy(
                learning_rate=learning_rate,
                discount_factor=discount_factor,
            )

        # Metrics
        self._metrics = MetricsTracker()

    @property
    def q_table(self):
        """Access the strategy's Q-table (for backward compatibility)."""
        return self._strategy.q_table

    @property
    def episode_count(self) -> int:
        """Access the metrics tracker's episode count."""
        return self._metrics.episode_count

    @property
    def cumulative_reward(self) -> float:
        """Access the metrics tracker's cumulative reward."""
        return self._metrics.cumulative_reward

    @property
    def prompt_selection_counts(self) -> Dict[str, int]:
        """Access the metrics tracker's prompt selection counts."""
        return self._metrics.prompt_selection_counts

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

    # Valid modes
    VALID_MODES = {MODE_TRAINING, MODE_INFERENCE}

    def set_mode(self, mode: str) -> None:
        """
        Switch agent between training and inference modes.

        When switching to inference, exploration_rate is set to 0.0
        so the agent always exploits learned knowledge.

        Args:
            mode: Target mode ("training" or "inference")

        Raises:
            ModeError: If mode is not a valid mode string
        """
        if mode not in self.VALID_MODES:
            raise ModeError(
                f"Invalid mode: '{mode}'. Must be one of {self.VALID_MODES}"
            )
        self.mode = mode
        if mode == self.MODE_INFERENCE:
            self.exploration_rate = self.PARAM_MIN

    def decay_exploration(self) -> None:
        """
        Apply multiplicative decay to exploration rate.

        exploration_rate *= decay_rate, floored at min_exploration.
        Only decays in training mode.
        """
        if self.mode != self.MODE_TRAINING:
            return
        self.exploration_rate = max(
            self.exploration_rate * self.decay_rate,
            self.min_exploration,
        )

    def select_action(self, state: str) -> str:
        """
        Select a prompt for the given state.

        Training mode: delegates to strategy (ε-greedy by default).
        Inference mode: always exploit (exploration_rate=0.0).

        Args:
            state: Current task context

        Returns:
            Selected prompt text
        """
        if self.mode == self.MODE_INFERENCE:
            action = self._strategy.select_action(state, self.prompts, 0.0)
        else:
            action = self._strategy.select_action(
                state, self.prompts, self.exploration_rate
            )
        self._metrics.record_selection(action)
        return action

    def update(self, state: str, action: str, reward: float) -> None:
        """
        Update knowledge based on received reward (training mode only).

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

        self._strategy.update(state, action, reward)

    def store_experience(self, state: str, action: str, reward: float) -> None:
        """
        Add episode to experience buffer, update metrics, and decay exploration.

        Args:
            state: State where action was taken
            action: Action (prompt) that was selected
            reward: Reward received
        """
        self._buffer.add(state, action, reward)
        self._metrics.record_episode(reward)
        self.decay_exploration()

    def train_batch(self) -> None:
        """
        Train on all experiences in the buffer (offline batch training).

        Iterates through all stored episodes and updates via strategy.
        Does not clear the buffer automatically.
        """
        for state, action, reward in self._buffer.get_all():
            self.update(state, action, reward)

    def clear_buffer(self) -> None:
        """Clear all experiences from the buffer."""
        self._buffer.clear()

    def get_metrics(self) -> dict:
        """
        Return current agent performance metrics.

        Returns:
            Dictionary with episode_count, cumulative_reward,
            average_reward, exploration_rate, and prompt_selection_counts.
        """
        return self._metrics.get_metrics(self.exploration_rate)

    def save_policy(self, filepath: str) -> None:
        """
        Save agent state to a JSON file.

        Persists Q-table, hyperparameters, metrics, mode, and prompts
        so the agent can be restored later for inference or continued training.

        Args:
            filepath: Path to the output JSON file

        Raises:
            PersistenceError: If the file cannot be written
        """
        state = {
            KEY_Q_TABLE: self._strategy.get_table(),
            KEY_CONFIG: {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "exploration_rate": self.exploration_rate,
                "decay_rate": self.decay_rate,
                "min_exploration": self.min_exploration,
            },
            KEY_METRICS: {
                "episode_count": self._metrics.episode_count,
                "cumulative_reward": self._metrics.cumulative_reward,
                "prompt_selection_counts": dict(self._metrics.prompt_selection_counts),
            },
            KEY_MODE: self.mode,
            KEY_PROMPTS: list(self.prompts),
        }
        _save_policy(state, filepath)

    def load_policy(self, filepath: str) -> None:
        """
        Load agent state from a JSON file.

        Restores Q-table, hyperparameters, metrics, mode, and prompts
        from a previously saved policy file.

        Args:
            filepath: Path to the input JSON file

        Raises:
            PersistenceError: If the file cannot be read or contains invalid data
        """
        data = _load_policy(filepath)

        try:
            # Restore Q-table
            self._strategy.q_table.from_dict(data[KEY_Q_TABLE])

            # Restore config
            config = data[KEY_CONFIG]
            self.learning_rate = config["learning_rate"]
            self.discount_factor = config["discount_factor"]
            self.exploration_rate = config["exploration_rate"]
            self.decay_rate = config["decay_rate"]
            self.min_exploration = config["min_exploration"]

            # Restore metrics
            metrics = data[KEY_METRICS]
            self._metrics.episode_count = metrics["episode_count"]
            self._metrics.cumulative_reward = metrics["cumulative_reward"]
            self._metrics.prompt_selection_counts = dict(metrics["prompt_selection_counts"])

            # Restore mode and prompts
            self.mode = data[KEY_MODE]
            self.prompts = list(data[KEY_PROMPTS])
        except (KeyError, TypeError) as exc:
            raise PersistenceError(
                f"Invalid policy file structure in {filepath}: {exc}"
            ) from exc
