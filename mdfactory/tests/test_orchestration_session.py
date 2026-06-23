# ABOUTME: Tests for the reusable Parsl session context manager
# ABOUTME: Validates DFK guard, load, shutdown, and detach semantics
"""Tests for orchestration Parsl session management."""

from unittest.mock import MagicMock

import pytest

parsl = pytest.importorskip("parsl", reason="parsl not installed")

from mdfactory.orchestration.config import ExecutorConfig  # noqa: E402
from mdfactory.orchestration.session import (  # noqa: E402
    ParslSession,
    parsl_session,
)


def _patch_parsl(monkeypatch, *, active_dfk=False):
    """Patch parsl.load/clear/dfk for session tests."""
    monkeypatch.setattr(parsl, "load", MagicMock())
    monkeypatch.setattr(parsl, "clear", MagicMock())
    if active_dfk:
        monkeypatch.setattr(parsl, "dfk", MagicMock(return_value=MagicMock(executors={})))
    else:
        monkeypatch.setattr(parsl, "dfk", MagicMock(side_effect=RuntimeError("No DFK")))
    monkeypatch.setattr(ExecutorConfig, "to_parsl_config", lambda self: MagicMock())


def test_parsl_session_loads_and_shuts_down(monkeypatch):
    """parsl_session loads the config and clears the DFK on exit."""
    _patch_parsl(monkeypatch)

    with parsl_session(ExecutorConfig()) as session:
        assert isinstance(session, ParslSession)
        assert session.detached is False

    parsl.load.assert_called_once()
    parsl.clear.assert_called_once()


def test_parsl_session_detach_skips_shutdown(monkeypatch):
    """A detached session does not shut down the DFK on exit."""
    _patch_parsl(monkeypatch)

    with parsl_session(ExecutorConfig()) as session:
        session.detach()

    parsl.load.assert_called_once()
    parsl.clear.assert_not_called()


def test_parsl_session_guards_active_dfk(monkeypatch):
    """parsl_session raises if a DFK is already active and does not load."""
    _patch_parsl(monkeypatch, active_dfk=True)

    with pytest.raises(RuntimeError, match="already active"):
        with parsl_session(ExecutorConfig()):
            pass

    parsl.load.assert_not_called()


def test_parsl_session_shuts_down_on_exception(monkeypatch):
    """parsl_session clears the DFK even when the body raises."""
    _patch_parsl(monkeypatch)

    with pytest.raises(ValueError, match="boom"):
        with parsl_session(ExecutorConfig()):
            raise ValueError("boom")

    parsl.clear.assert_called_once()
