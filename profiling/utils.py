import logging
import sys

logger = logging.getLogger(__name__)

# Store original matplotlib modules if they exist, used by restore_matplotlib
_original_matplotlib = sys.modules.get("matplotlib", None)
_original_pyplot = sys.modules.get("matplotlib.pyplot", None)
_matplotlib_disabled = False


class MockMatplotlibModule:
    """A mock object to replace matplotlib modules."""

    def __getattr__(self, name):
        # Return a function that does nothing for any attribute access.
        def _noop(*args, **kwargs):
            return None

        return _noop


def disable_matplotlib():
    """
    Disables matplotlib to prevent plotting overhead during profiling or benchmarking.

    Replaces matplotlib and matplotlib.pyplot in sys.modules with mock objects
    that swallow all calls. Call restore_matplotlib() to undo.
    """
    global _matplotlib_disabled
    if _matplotlib_disabled:
        logger.warning("Matplotlib already disabled.")
        return

    logger.info("Disabling matplotlib and matplotlib.pyplot...")
    sys.modules["matplotlib"] = MockMatplotlibModule()
    sys.modules["matplotlib.pyplot"] = MockMatplotlibModule()
    _matplotlib_disabled = True


def restore_matplotlib():
    """Restores the original matplotlib modules if they were disabled."""
    global _matplotlib_disabled
    if not _matplotlib_disabled:
        logger.warning("Matplotlib was not disabled.")
        return

    logger.info("Restoring original matplotlib modules...")
    if _original_matplotlib:
        sys.modules["matplotlib"] = _original_matplotlib
    else:
        # If matplotlib wasn't imported before disable, remove the mock.
        if "matplotlib" in sys.modules:
            del sys.modules["matplotlib"]

    if _original_pyplot:
        sys.modules["matplotlib.pyplot"] = _original_pyplot
    else:
        if "matplotlib.pyplot" in sys.modules:
            del sys.modules["matplotlib.pyplot"]

    _matplotlib_disabled = False


# Removed __main__ block as it was only for demonstration.
