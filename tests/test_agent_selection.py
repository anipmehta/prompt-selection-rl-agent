"""
Unit tests for RLAgent action selection (ε-greedy).
"""

from unittest.mock import patch

from src.agent import RLAgent

# Test prompts
SAMPLE_PROMPTS = ["Be concise", "Explain step-by-step", "Use examples"]

# Exploration rates
FULL_EXPLORATION = 1.0
NO_EXPLORATION = 0.0

# Q-values for testing exploitation
HIGH_Q_VALUE = 0.9
LOW_Q_VALUE = 0.1

# Test state
TEST_STATE = "summarize this article"
UNSEEN_STATE = "never seen before"


class TestActionSelectionValidity:
    """Test that select_action always returns a valid prompt."""

    def test_returns_valid_prompt_training(self):
        """Action selection in training mode returns a prompt from the list."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        for _ in range(50):
            action = agent.select_action(TEST_STATE)
            assert action in SAMPLE_PROMPTS

    def test_returns_valid_prompt_inference(self):
        """Action selection in inference mode returns a prompt from the list."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.mode = RLAgent.MODE_INFERENCE
        for _ in range(50):
            action = agent.select_action(TEST_STATE)
            assert action in SAMPLE_PROMPTS


class TestExploration:
    """Test exploration behavior (random selection)."""

    def test_full_exploration_selects_randomly(self):
        """With ε=1.0, agent should explore (random selection)."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS, exploration_rate=FULL_EXPLORATION)

        # Set up Q-table so one prompt is clearly best
        agent.q_table.set(TEST_STATE, SAMPLE_PROMPTS[0], HIGH_Q_VALUE)
        agent.q_table.set(TEST_STATE, SAMPLE_PROMPTS[1], LOW_Q_VALUE)
        agent.q_table.set(TEST_STATE, SAMPLE_PROMPTS[2], LOW_Q_VALUE)

        # With full exploration, we should see multiple prompts selected
        selected = {agent.select_action(TEST_STATE) for _ in range(100)}
        assert len(selected) > 1

    @patch("src.agent.random.random", return_value=0.5)
    @patch("src.agent.random.choice", return_value="Be concise")
    def test_explores_when_random_below_epsilon(self, mock_choice, _mock_random):
        """Agent explores when random value <= exploration_rate."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS, exploration_rate=0.8)
        action = agent.select_action(TEST_STATE)
        assert action == "Be concise"
        mock_choice.assert_called_once()


class TestExploitation:
    """Test exploitation behavior (best Q-value selection)."""

    def test_no_exploration_selects_best(self):
        """With ε=0.0, agent should always select best Q-value prompt."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS, exploration_rate=NO_EXPLORATION)

        best_prompt = SAMPLE_PROMPTS[1]
        agent.q_table.set(TEST_STATE, SAMPLE_PROMPTS[0], LOW_Q_VALUE)
        agent.q_table.set(TEST_STATE, best_prompt, HIGH_Q_VALUE)
        agent.q_table.set(TEST_STATE, SAMPLE_PROMPTS[2], LOW_Q_VALUE)

        for _ in range(20):
            action = agent.select_action(TEST_STATE)
            assert action == best_prompt

    def test_inference_mode_always_exploits(self):
        """Inference mode should always exploit regardless of exploration rate."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS, exploration_rate=FULL_EXPLORATION)
        agent.mode = RLAgent.MODE_INFERENCE

        best_prompt = SAMPLE_PROMPTS[2]
        agent.q_table.set(TEST_STATE, SAMPLE_PROMPTS[0], LOW_Q_VALUE)
        agent.q_table.set(TEST_STATE, SAMPLE_PROMPTS[1], LOW_Q_VALUE)
        agent.q_table.set(TEST_STATE, best_prompt, HIGH_Q_VALUE)

        for _ in range(20):
            action = agent.select_action(TEST_STATE)
            assert action == best_prompt

    def test_unseen_state_returns_valid_prompt(self):
        """Unseen state with no Q-values should still return a valid prompt."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS, exploration_rate=NO_EXPLORATION)
        action = agent.select_action(UNSEEN_STATE)
        assert action in SAMPLE_PROMPTS
