# ABOUTME: Tests for the TUI browse command and Textual app
# ABOUTME: Validates app composition, filter parsing, and browse CLI integration
"""Tests for TUI browse command and Textual app."""

import pytest

from mdfactory import cli
from mdfactory.analysis.store import SimulationStore

# Check if textual is available for TUI tests
try:
    import textual  # noqa: F401

    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False


def test_browse_without_textual(tmp_path, monkeypatch):
    """Test browse command gives clear error when textual not installed."""
    import mdfactory.tui as tui_module

    def mock_check():
        raise ImportError(
            "The TUI requires the 'textual' package. Install it with: pip install mdfactory[tui]"
        )

    monkeypatch.setattr(tui_module, "_check_textual_available", mock_check)

    with pytest.raises(SystemExit):
        cli.browse_simulations(tmp_path)


def test_tui_check_textual_available():
    """Test _check_textual_available function."""
    from mdfactory.tui import _check_textual_available

    if HAS_TEXTUAL:
        _check_textual_available()
    else:
        with pytest.raises(ImportError, match="textual"):
            _check_textual_available()


@pytest.mark.skipif(not HAS_TEXTUAL, reason="textual not installed")
class TestSimulationBrowser:
    """Tests for the Textual SimulationBrowser app."""

    def test_app_creation(self, tmp_path):
        """Test SimulationBrowser can be instantiated."""
        from mdfactory.tui.app import SimulationBrowser

        store = SimulationStore(tmp_path, min_status="build")
        app = SimulationBrowser(store=store)
        assert app.store is store

    @pytest.mark.asyncio
    async def test_app_compose(self, tmp_path):
        """Test app composes correctly with all widgets."""
        from textual.widgets import DataTable, Input, Select

        from mdfactory.tui.app import SimulationBrowser, SimulationDetail

        store = SimulationStore(tmp_path, min_status="build")
        app = SimulationBrowser(store=store)

        async with app.run_test():
            assert app.query_one("#results-table", DataTable)
            assert app.query_one("#detail-panel", SimulationDetail)
            assert app.query_one("#filter-type", Select)
            assert app.query_one("#filter-status", Select)
            assert len(app.query(Input)) == 3  # hash, tags, smiles

    @pytest.mark.asyncio
    async def test_app_clear_filters(self, tmp_path):
        """Test clear action resets all filters."""
        from textual.widgets import Input, Select

        from mdfactory.tui.app import SimulationBrowser

        store = SimulationStore(tmp_path, min_status="build")
        app = SimulationBrowser(store=store)

        async with app.run_test() as pilot:
            # Set some filters
            app.query_one("#filter-hash", Input).value = "ABC"
            await pilot.pause()

            # Clear
            app.action_clear_filters()
            await pilot.pause()

            for inp in app.query(Input):
                assert inp.value == ""
            assert app.query_one("#filter-type", Select).value is None
            assert app.query_one("#filter-status", Select).value is None
