"""State encoders for normalizing task descriptions before Q-table lookup."""


def lowercase_encoder(state: str) -> str:
    """
    Normalize state by lowercasing and stripping whitespace.

    Ensures "Summarize This" and "  summarize this  " map to
    the same Q-table key.

    Args:
        state: Raw state string

    Returns:
        Lowercased, stripped state string
    """
    return state.strip().lower()
