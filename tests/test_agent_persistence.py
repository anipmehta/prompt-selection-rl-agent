"""Wiring tests for agent save_policy/load_policy (delegates to src/policy.py)."""

import os

import pytest

from src.agent import RLAgent

# Test constants
SAMPLE_PROMPTS = ["Be concise", "Be detailed", "Be creative"]
TEST_STATE = "summarize this article"
TEST_ACTION = SAMPLE_PROMPTS[0]
POSITIVE_REWARD = 0.8
POLICY_FILE = "test_agent_policy.json"


@pytest.fixture(autouse=True)
def cleanup_files():
    """Remove test files after each test."""
    yield
    if os.path.exists(POLICY_FILE):
        os.remove(POLICY_FILE)


class TestAgentPersistenceWiring:
    """Tests that agent correctly delegates to policy module."""

    def test_save_load_restores_q_table(self) -> None:
        """Round-trip through agent should restore Q-table."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.update(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)
        agent.save_policy(POLICY_FILE)

        new_agent = RLAgent(prompts=SAMPLE_PROMPTS)
        new_agent.load_policy(POLICY_FILE)
        assert new_agent.q_table.get(TEST_STATE, TEST_ACTION) > 0.0

    def test_save_load_restores_mode(self) -> None:
        """Round-trip should restore inference mode."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.set_mode("inference")
        agent.save_policy(POLICY_FILE)

        new_agent = RLAgent(prompts=SAMPLE_PROMPTS)
        new_agent.load_policy(POLICY_FILE)
        assert new_agent.mode == "inference"

    def test_save_load_restores_metrics(self) -> None:
        """Round-trip should restore metrics."""
        agent = RLAgent(prompts=SAMPLE_PROMPTS)
        agent.store_experience(TEST_STATE, TEST_ACTION, POSITIVE_REWARD)
        agent.save_policy(POLICY_FILE)

        new_agent = RLAgent(prompts=SAMPLE_PROMPTS)
        new_agent.load_policy(POLICY_FILE)
        assert new_agent.episode_count == 1
