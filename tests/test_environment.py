"""Tests for the Environment class."""

import pytest

from src.environment import Environment
from src.errors import ValidationError


class TestEnvironmentInit:
    """Tests for Environment initialization."""

    def test_defaults_to_manual_mode(self):
        env = Environment()
        assert env._manual_mode is True


class TestSetManualMode:
    """Tests for set_manual_mode."""

    def test_disable_manual_mode(self):
        env = Environment()
        env.set_manual_mode(False)
        assert env._manual_mode is False

    def test_enable_manual_mode(self):
        env = Environment()
        env.set_manual_mode(False)
        env.set_manual_mode(True)
        assert env._manual_mode is True


class TestExecuteManualMode:
    """Tests for execute in manual mode."""

    def test_returns_valid_reward(self, monkeypatch):
        env = Environment()
        monkeypatch.setattr("builtins.input", lambda _: "0.5")
        reward = env.execute("prompt1", "task1")
        assert reward == 0.5

    def test_prints_prompt_and_task(self, monkeypatch, capsys):
        env = Environment()
        monkeypatch.setattr("builtins.input", lambda _: "0.0")
        env.execute("my prompt", "my task")
        captured = capsys.readouterr()
        assert "Prompt: my prompt" in captured.out
        assert "Task: my task" in captured.out

    def test_accepts_negative_reward(self, monkeypatch):
        env = Environment()
        monkeypatch.setattr("builtins.input", lambda _: "-0.75")
        assert env.execute("p", "t") == -0.75

    def test_accepts_boundary_min(self, monkeypatch):
        env = Environment()
        monkeypatch.setattr("builtins.input", lambda _: "-1.0")
        assert env.execute("p", "t") == -1.0

    def test_accepts_boundary_max(self, monkeypatch):
        env = Environment()
        monkeypatch.setattr("builtins.input", lambda _: "1.0")
        assert env.execute("p", "t") == 1.0

    def test_rejects_reward_above_range(self, monkeypatch):
        env = Environment()
        monkeypatch.setattr("builtins.input", lambda _: "1.5")
        with pytest.raises(ValidationError, match="Invalid reward"):
            env.execute("p", "t")

    def test_rejects_reward_below_range(self, monkeypatch):
        env = Environment()
        monkeypatch.setattr("builtins.input", lambda _: "-1.5")
        with pytest.raises(ValidationError, match="Invalid reward"):
            env.execute("p", "t")

    def test_rejects_non_numeric_input(self, monkeypatch):
        env = Environment()
        monkeypatch.setattr("builtins.input", lambda _: "abc")
        with pytest.raises(ValidationError, match="Invalid reward"):
            env.execute("p", "t")

    def test_rejects_empty_input(self, monkeypatch):
        env = Environment()
        monkeypatch.setattr("builtins.input", lambda _: "")
        with pytest.raises(ValidationError, match="Invalid reward"):
            env.execute("p", "t")


class TestExecuteNonManualMode:
    """Tests for execute when manual mode is disabled."""

    def test_raises_not_implemented(self):
        env = Environment()
        env.set_manual_mode(False)
        with pytest.raises(NotImplementedError):
            env.execute("p", "t")


class TestParseReward:
    """Tests for the static _parse_reward helper."""

    def test_valid_zero(self):
        assert Environment._parse_reward("0.0") == 0.0

    def test_valid_positive(self):
        assert Environment._parse_reward("0.42") == 0.42

    def test_valid_negative(self):
        assert Environment._parse_reward("-0.99") == -0.99

    def test_integer_string(self):
        assert Environment._parse_reward("1") == 1.0

    def test_invalid_string(self):
        with pytest.raises(ValidationError):
            Environment._parse_reward("not_a_number")

    def test_out_of_range_high(self):
        with pytest.raises(ValidationError):
            Environment._parse_reward("2.0")

    def test_out_of_range_low(self):
        with pytest.raises(ValidationError):
            Environment._parse_reward("-2.0")
