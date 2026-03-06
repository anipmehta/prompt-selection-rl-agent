"""Environment for executing prompts and computing rewards."""

from .errors import ValidationError


class Environment:
    """Environment for executing prompts and computing rewards.

    Supports manual mode where a human provides reward values via console input.
    Defaults to manual mode for the PoC.
    """

    def __init__(self) -> None:
        self._manual_mode: bool = True

    def execute(self, prompt: str, task: str) -> float:
        """Execute a prompt against a task and return a reward.

        In manual mode, prints the prompt and task, then asks the user
        to enter a reward value in [-1.0, 1.0].

        Args:
            prompt: The selected prompt template.
            task: The task description.

        Returns:
            Reward value between -1.0 and 1.0.

        Raises:
            ValidationError: If the user-provided reward is not a valid
                float in [-1.0, 1.0].
        """
        if self._manual_mode:
            print(f"Prompt: {prompt}")
            print(f"Task: {task}")
            raw = input("Enter reward [-1.0, 1.0]: ")
            return self._parse_reward(raw)
        # Non-manual execution placeholder for future automated environments
        raise NotImplementedError("Automated environment execution is not yet supported.")

    def set_manual_mode(self, enabled: bool) -> None:
        """Enable or disable manual reward entry.

        Args:
            enabled: If True, prompts user for reward input via console.
        """
        self._manual_mode = enabled

    @staticmethod
    def _parse_reward(raw: str) -> float:
        """Parse and validate a raw reward string.

        Args:
            raw: The raw string entered by the user.

        Returns:
            Validated reward float.

        Raises:
            ValidationError: If the value is not a float in [-1.0, 1.0].
        """
        try:
            value = float(raw)
        except (ValueError, TypeError) as exc:
            raise ValidationError(
                f"Invalid reward: {raw!r}. Must be a float in range [-1.0, 1.0]"
            ) from exc

        if value < -1.0 or value > 1.0:
            raise ValidationError(
                f"Invalid reward: {value}. Must be in range [-1.0, 1.0]"
            )
        return value
