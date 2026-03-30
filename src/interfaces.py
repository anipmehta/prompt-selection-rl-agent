"""Extensibility interfaces for integrating the RL agent with external systems.

ActionExecutor: sends a prompt to an LLM and returns the response.
RewardFunction: scores the quality of an LLM response.

Implement these to connect the agent to your annotation platform.
"""

from abc import ABC, abstractmethod  # ABC = Abstract Base Class (defines interface contracts)


class ActionExecutor(ABC):  # pylint: disable=too-few-public-methods
    """
    Interface for executing a prompt against an external system (e.g. LLM).

    Implementations handle the actual API call to the language model,
    passing the selected prompt and task context, and returning the
    raw response text.

    Example implementation::

        class OpenAIExecutor(ActionExecutor):
            def execute(self, prompt: str, task: str) -> str:
                response = openai.chat(prompt + task)
                return response.text
    """

    @abstractmethod
    def execute(self, prompt: str, task: str) -> str:
        """
        Execute a prompt for the given task.

        Args:
            prompt: The selected prompt template
            task: The task context (e.g. text to annotate)

        Returns:
            The LLM response text
        """


class RewardFunction(ABC):  # pylint: disable=too-few-public-methods
    """
    Interface for computing reward from an LLM response.

    Implementations score how well the LLM response matches
    the expected quality for the given task. The reward must
    be in [-1.0, 1.0].

    Example implementation::

        class JudgeReward(RewardFunction):
            def compute(self, task: str, prompt: str, result: str) -> float:
                score = judge_llm.evaluate(task, result)
                return score  # normalized to [-1.0, 1.0]
    """

    @abstractmethod
    def compute(self, task: str, prompt: str, result: str) -> float:
        """
        Compute reward for an LLM response.

        Args:
            task: The original task context
            prompt: The prompt that was used
            result: The LLM response text

        Returns:
            Reward value in [-1.0, 1.0]
        """
