"""Wiring tests for agent state_encoder (delegates to src/state_encoder.py)."""

from src.agent import RLAgent
from src.state_encoder import lowercase_encoder

# Test constants
SAMPLE_PROMPTS = ["Be concise", "Be detailed"]
RAW_STATE = "Summarize This"
NORMALIZED_STATE = "summarize this"
POSITIVE_REWARD = 0.8
NO_EXPLORATION = 0.0


class TestAgentEncoderWiring:
    """Tests that agent applies state_encoder before Q-table operations."""

    def test_no_encoder_uses_raw_state(self) -> None:
        """Without encoder, state should be used as-is."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS, exploration_rate=NO_EXPLORATION)
        agent.update(RAW_STATE, SAMPLE_PROMPTS[0], POSITIVE_REWARD)
        assert agent.q_table.get(RAW_STATE, SAMPLE_PROMPTS[0]) > 0.0

    def test_lowercase_encoder_normalizes_state(self) -> None:
        """With lowercase encoder, mixed-case state should map to lowercase key."""
        agent = RLAgent(
            prompts=SAMPLE_PROMPTS,
            exploration_rate=NO_EXPLORATION,
            state_encoder=lowercase_encoder,
        )
        agent.update(RAW_STATE, SAMPLE_PROMPTS[0], POSITIVE_REWARD)
        assert agent.q_table.get(NORMALIZED_STATE, SAMPLE_PROMPTS[0]) > 0.0
        assert agent.q_table.get(RAW_STATE, SAMPLE_PROMPTS[0]) == 0.0

    def test_encoder_applied_in_select_action(self) -> None:
        """select_action should encode state before Q-table lookup."""
        agent = RLAgent(
            prompts=SAMPLE_PROMPTS,
            exploration_rate=NO_EXPLORATION,
            state_encoder=lowercase_encoder,
        )
        agent.update(NORMALIZED_STATE, SAMPLE_PROMPTS[0], POSITIVE_REWARD)
        action = agent.select_action(RAW_STATE)
        assert action == SAMPLE_PROMPTS[0]
