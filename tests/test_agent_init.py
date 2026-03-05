"""
Unit tests for RLAgent initialization and parameter validation.
"""

import pytest

from src.agent import RLAgent
from src.errors import ConfigurationError

# Test prompts
SAMPLE_PROMPTS = ["Be concise", "Explain step-by-step", "Use examples"]

# Valid parameter values
VALID_RATE = 0.5
VALID_BOUNDARY_MIN = 0.0
VALID_BOUNDARY_MAX = 1.0

# Invalid parameter values
INVALID_RATE_NEGATIVE = -0.1
INVALID_RATE_OVER = 1.1


class TestAgentInitialization:
    """Test agent initialization with valid parameters."""

    def test_default_parameters(self):
        """Test agent initializes with correct defaults."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)

        assert agent.learning_rate == RLAgent.DEFAULT_LEARNING_RATE
        assert agent.discount_factor == RLAgent.DEFAULT_DISCOUNT_FACTOR
        assert agent.exploration_rate == RLAgent.DEFAULT_EXPLORATION_RATE
        assert agent.decay_rate == RLAgent.DEFAULT_DECAY_RATE
        assert agent.min_exploration == RLAgent.DEFAULT_MIN_EXPLORATION

    def test_custom_parameters(self):
        """Test agent initializes with custom parameters."""
        agent = RLAgent(
            prompts=SAMPLE_PROMPTS,
            learning_rate=VALID_RATE,
            discount_factor=VALID_RATE,
            exploration_rate=VALID_RATE,
            decay_rate=VALID_RATE,
            min_exploration=VALID_RATE,
        )

        assert agent.learning_rate == VALID_RATE
        assert agent.discount_factor == VALID_RATE

    def test_prompts_stored(self):
        """Test that prompts list is stored correctly."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        assert agent.prompts == SAMPLE_PROMPTS

    def test_prompts_stored_as_copy(self):
        """Test that prompts are stored as a copy, not reference."""
        original = list(SAMPLE_PROMPTS)
        agent = RLAgent(prompts=original)
        original.append("new prompt")
        assert len(agent.prompts) == len(SAMPLE_PROMPTS)

    def test_empty_q_table(self):
        """Test that Q-table initializes empty."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        assert len(agent.q_table) == 0

    def test_default_mode_is_training(self):
        """Test that default mode is training."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        assert agent.mode == RLAgent.MODE_TRAINING

    def test_boundary_values_accepted(self):
        """Test that boundary values 0.0 and 1.0 are accepted."""
        agent = RLAgent(
            prompts=SAMPLE_PROMPTS,
            learning_rate=VALID_BOUNDARY_MIN,
            discount_factor=VALID_BOUNDARY_MAX,
        )
        assert agent.learning_rate == VALID_BOUNDARY_MIN
        assert agent.discount_factor == VALID_BOUNDARY_MAX


class TestAgentParameterValidation:
    """Test agent parameter validation raises errors for invalid inputs."""

    def test_invalid_learning_rate_negative(self):
        """Test that negative learning rate raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="learning_rate"):
            RLAgent(prompts=SAMPLE_PROMPTS, learning_rate=INVALID_RATE_NEGATIVE)

    def test_invalid_learning_rate_over(self):
        """Test that learning rate > 1.0 raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="learning_rate"):
            RLAgent(prompts=SAMPLE_PROMPTS, learning_rate=INVALID_RATE_OVER)

    def test_invalid_discount_factor(self):
        """Test that invalid discount factor raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="discount_factor"):
            RLAgent(prompts=SAMPLE_PROMPTS, discount_factor=INVALID_RATE_NEGATIVE)

    def test_invalid_exploration_rate(self):
        """Test that invalid exploration rate raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="exploration_rate"):
            RLAgent(prompts=SAMPLE_PROMPTS, exploration_rate=INVALID_RATE_OVER)

    def test_invalid_decay_rate(self):
        """Test that invalid decay rate raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="decay_rate"):
            RLAgent(prompts=SAMPLE_PROMPTS, decay_rate=INVALID_RATE_NEGATIVE)

    def test_empty_prompts_raises_error(self):
        """Test that empty prompts list raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="empty"):
            RLAgent(prompts=[])
