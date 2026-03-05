"""Custom error types for the RL agent."""


class ConfigurationError(ValueError):
    """Raised when agent is initialized with invalid parameters."""


class ValidationError(ValueError):
    """Raised when invalid values are provided during operation."""


class ModeError(ValueError):
    """Raised when an invalid mode is specified."""


class PersistenceError(IOError):
    """Raised when save/load operations fail."""
