"""Custom error types for the RL agent."""


class ConfigurationError(ValueError):
    """Raised when agent is initialized with invalid parameters."""
    pass


class ValidationError(ValueError):
    """Raised when invalid values are provided during operation."""
    pass


class ModeError(ValueError):
    """Raised when an invalid mode is specified."""
    pass


class PersistenceError(IOError):
    """Raised when save/load operations fail."""
    pass
