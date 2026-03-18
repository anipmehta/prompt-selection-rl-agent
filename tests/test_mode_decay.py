"""Tests for mode switching and exploration decay (Phase 3)."""

import pytest

from src.agent import RLAgent
from src.errors import ModeError

# Test prompts
SAMPLE_PROMPTS = ["Be concise", "Be detailed", "Be creative"]

# Test constants
TEST_STATE = "summarize this article"
TEST_ACTION = SAMPLE_PROMPTS[0]
POSITIVE_REWARD = 0.8
CUSTOM_DECAY_RATE = 0.9
CUSTOM_MIN_EXPLORATION = 0.05
INITIAL_EXPLORATION = 1.0
ZERO_EXPLORATION = 0.0


class TestSetMode:
    """Tests for set_mode() method."""

    def test_default_mode_is_training(self) -> None:
        """Agent should start in training mode."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        assert agent.mode == RLAgent.MODE_TRAINING

    def test_switch_to_inference(self) -> None:
        """set_mode('inference') should update mode."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.set_mode(RLAgent.MODE_INFERENCE)
        assert agent.mode == RLAgent.MODE_INFERENCE

    def test_switch_to_training(self) -> None:
        """set_mode('training') should update mode."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.set_mode(RLAgent.MODE_INFERENCE)
        agent.set_mode(RLAgent.MODE_TRAINING)
        assert agent.mode == RLAgent.MODE_TRAINING

    def test_inference_zeroes_exploration(self) -> None:
        """Switching to inference should set exploration_rate to 0.0."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        assert agent.exploration_rate == INITIAL_EXPLORATION
        agent.set_mode(RLAgent.MODE_INFERENCE)
        assert agent.exploration_rate == ZERO_EXPLORATION

    def test_invalid_mode_raises_mode_error(self) -> None:
        """Invalid mode string should raise ModeError."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        with pytest.raises(ModeError):
            agent.set_mode("banana")

    def test_invalid_mode_empty_string(self) -> None:
        """Empty string should raise ModeError."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        with pytest.raises(ModeError):
            agent.set_mode("")

    def test_inference_blocks_q_updates(self) -> None:
        """Q-table should not change during inference mode."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.update(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)
        q_before = agent.q_table.get(TEST_STATE, TEST_ACTION)

        agent.set_mode(RLAgent.MODE_INFERENCE)
        agent.update(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)
        q_after = agent.q_table.get(TEST_STATE, TEST_ACTION)

        assert q_before == q_after

    def test_inference_always_exploits(self) -> None:
        """In inference mode, agent should always pick the best action."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS, exploration_rate=INITIAL_EXPLORATION)
        # Train a clear winner
        for _ in range(20):
            agent.update(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)

        agent.set_mode(RLAgent.MODE_INFERENCE)

        # Should always pick the trained action
        for _ in range(50):
            assert agent.select_action(TEST_STATE) == TEST_ACTION


class TestDecayExploration:
    """Tests for decay_exploration() method."""

    def test_decay_reduces_exploration(self) -> None:
        """decay_exploration should reduce exploration_rate."""
        agent = RLAgent(
            prompts=SAMPLE_PROMPTS,
            decay_rate=CUSTOM_DECAY_RATE,
        )
        initial = agent.exploration_rate
        agent.decay_exploration()
        assert agent.exploration_rate < initial

    def test_decay_applies_multiplicative_formula(self) -> None:
        """exploration_rate should equal initial * decay_rate after one decay."""
        agent = RLAgent(
            prompts=SAMPLE_PROMPTS,
            decay_rate=CUSTOM_DECAY_RATE,
        )
        expected = INITIAL_EXPLORATION * CUSTOM_DECAY_RATE
        agent.decay_exploration()
        assert agent.exploration_rate == pytest.approx(expected)

    def test_decay_respects_min_exploration(self) -> None:
        """exploration_rate should never go below min_exploration."""
        agent = RLAgent(
            prompts=SAMPLE_PROMPTS,
            exploration_rate=CUSTOM_MIN_EXPLORATION,
            decay_rate=CUSTOM_DECAY_RATE,
            min_exploration=CUSTOM_MIN_EXPLORATION,
        )
        agent.decay_exploration()
        assert agent.exploration_rate == CUSTOM_MIN_EXPLORATION

    def test_decay_skipped_in_inference_mode(self) -> None:
        """decay_exploration should do nothing in inference mode."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.set_mode(RLAgent.MODE_INFERENCE)
        agent.decay_exploration()
        assert agent.exploration_rate == ZERO_EXPLORATION

    def test_store_experience_triggers_decay(self) -> None:
        """store_experience should call decay_exploration."""
        agent = RLAgent(
            prompts=SAMPLE_PROMPTS,
            decay_rate=CUSTOM_DECAY_RATE,
        )
        initial = agent.exploration_rate
        agent.store_experience(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)
        assert agent.exploration_rate < initial

    def test_multiple_decays_converge_to_min(self) -> None:
        """Repeated decay should converge to min_exploration."""
        agent = RLAgent(
            prompts=SAMPLE_PROMPTS,
            decay_rate=CUSTOM_DECAY_RATE,
            min_exploration=CUSTOM_MIN_EXPLORATION,
        )
        for _ in range(1000):
            agent.decay_exploration()
        assert agent.exploration_rate == CUSTOM_MIN_EXPLORATION

    def test_decay_after_mode_roundtrip(self) -> None:
        """Decay should work after switching back to training from inference."""
        agent = RLAgent(
            prompts=SAMPLE_PROMPTS,
            exploration_rate=0.5,
            decay_rate=CUSTOM_DECAY_RATE,
            min_exploration=CUSTOM_MIN_EXPLORATION,
        )
        agent.set_mode(RLAgent.MODE_INFERENCE)
        agent.set_mode(RLAgent.MODE_TRAINING)
        # exploration_rate was zeroed by inference switch, now back in training
        # decay floors at min_exploration since 0.0 * decay < min
        agent.decay_exploration()
        assert agent.exploration_rate == CUSTOM_MIN_EXPLORATION
