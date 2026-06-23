# ABOUTME: TUI package for interactive simulation browsing
# ABOUTME: Provides a Textual-based interface for filtering and exploring simulations
"""TUI package for interactive simulation browsing."""


def _check_textual_available():
    """Check if textual is installed, raise helpful error if not."""
    try:
        import textual  # noqa: F401
    except ImportError:
        raise ImportError(
            "The TUI requires the 'textual' package. Install it with: pip install mdfactory[tui]"
        )
