# ABOUTME: Tests for per-stage simulation progress tracking and shared progress loop
# ABOUTME: Covers StageProgressTracker state transitions, thread safety, and run_progress_loop
"""Tests for the progress tracker and shared progress loop."""

import threading
from unittest.mock import MagicMock, patch

from mdfactory.orchestration.progress import StageProgressTracker, run_progress_loop


class TestStageProgressTracker:
    """Unit tests for StageProgressTracker."""

    def _make_tracker(self):
        return StageProgressTracker(
            stages=["EM", "NVT", "NPT", "Production"],
            sim_hashes=["aaa", "bbb", "ccc"],
        )

    def test_initial_state_all_pending(self):
        tracker = self._make_tracker()
        snap = tracker.snapshot()
        for stage in ["EM", "NVT", "NPT", "Production"]:
            assert snap[stage]["pending"] == 3
            assert snap[stage]["running"] == 0

    def test_mark_running(self):
        tracker = self._make_tracker()
        tracker.mark_running("EM", "aaa")
        snap = tracker.snapshot()
        assert snap["EM"]["running"] == 1
        assert snap["EM"]["pending"] == 2

    def test_mark_succeeded(self):
        tracker = self._make_tracker()
        tracker.mark_running("EM", "aaa")
        tracker.mark_succeeded("EM", "aaa")
        snap = tracker.snapshot()
        assert snap["EM"]["succeeded"] == 1
        assert snap["EM"]["running"] == 0

    def test_mark_failed(self):
        tracker = self._make_tracker()
        tracker.mark_running("NVT", "bbb")
        tracker.mark_failed("NVT", "bbb")
        snap = tracker.snapshot()
        assert snap["NVT"]["failed"] == 1

    def test_mark_skipped(self):
        tracker = self._make_tracker()
        tracker.mark_skipped("NPT", "ccc")
        snap = tracker.snapshot()
        assert snap["NPT"]["skipped"] == 1

    def test_all_done_false_initially(self):
        tracker = self._make_tracker()
        assert not tracker.all_done()

    def test_all_done_true_when_last_stage_terminal(self):
        tracker = self._make_tracker()
        tracker.mark_succeeded("Production", "aaa")
        tracker.mark_failed("Production", "bbb")
        tracker.mark_skipped("Production", "ccc")
        assert tracker.all_done()

    def test_all_done_false_with_running_last_stage(self):
        tracker = self._make_tracker()
        tracker.mark_succeeded("Production", "aaa")
        tracker.mark_succeeded("Production", "bbb")
        tracker.mark_running("Production", "ccc")
        assert not tracker.all_done()

    def test_store_and_collect_results(self):
        tracker = self._make_tracker()
        tracker.store_result("bbb", {"hash": "bbb", "status": "success"})
        tracker.store_result("aaa", {"hash": "aaa", "status": "failed"})
        tracker.store_result("ccc", {"hash": "ccc", "status": "success"})

        results = tracker.collect_results()
        assert len(results) == 3
        assert results[0]["hash"] == "aaa"
        assert results[1]["hash"] == "bbb"
        assert results[2]["hash"] == "ccc"

    def test_collect_results_missing_returns_unknown(self):
        tracker = self._make_tracker()
        tracker.store_result("aaa", {"hash": "aaa", "status": "success"})
        results = tracker.collect_results()
        assert results[1] == {"hash": "bbb", "status": "unknown"}

    def test_thread_safety(self):
        """Concurrent mark calls from multiple threads don't crash."""
        tracker = StageProgressTracker(
            stages=["EM", "Production"],
            sim_hashes=[f"sim{i}" for i in range(20)],
        )
        errors = []

        def _worker(i):
            try:
                h = f"sim{i}"
                tracker.mark_running("EM", h)
                tracker.mark_succeeded("EM", h)
                tracker.mark_running("Production", h)
                tracker.mark_succeeded("Production", h)
                tracker.store_result(h, {"hash": h, "status": "success"})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert tracker.all_done()
        snap = tracker.snapshot()
        assert snap["EM"]["succeeded"] == 20
        assert snap["Production"]["succeeded"] == 20

    def test_failure_cascade_pattern(self):
        """Simulates a pipeline failing at NVT — remaining stages skipped."""
        tracker = self._make_tracker()
        tracker.mark_running("EM", "aaa")
        tracker.mark_succeeded("EM", "aaa")
        tracker.mark_running("NVT", "aaa")
        tracker.mark_failed("NVT", "aaa")
        tracker.mark_skipped("NPT", "aaa")
        tracker.mark_skipped("Production", "aaa")

        # Other sims succeed
        for h in ["bbb", "ccc"]:
            for s in ["EM", "NVT", "NPT", "Production"]:
                tracker.mark_succeeded(s, h)

        assert tracker.all_done()
        snap = tracker.snapshot()
        assert snap["NVT"]["failed"] == 1
        assert snap["NPT"]["skipped"] == 1
        assert snap["Production"]["skipped"] == 1


class TestRunProgressLoop:
    """Unit tests for the shared run_progress_loop driver."""

    def _progress(self):
        from mdfactory.orchestration.progress import _make_progress

        return _make_progress()

    @patch("mdfactory.orchestration.progress._get_block_status", return_value="")
    @patch("rich.live.Live")
    def test_update_polled_until_done(self, _mock_live, _mock_block):
        """update is called repeatedly until it returns True."""
        update = MagicMock(side_effect=[False, False, True])

        run_progress_loop(self._progress(), update=update, poll_interval=0.0)

        assert update.call_count == 3

    @patch("mdfactory.orchestration.progress._get_block_status", return_value="")
    @patch("rich.live.Live")
    def test_render_extras_called_each_tick(self, _mock_live, _mock_block):
        """render_extras is called on every render."""
        update = MagicMock(side_effect=[False, True])
        render_extras = MagicMock(return_value=[])

        run_progress_loop(
            self._progress(), update=update, render_extras=render_extras, poll_interval=0.0
        )

        assert render_extras.call_count >= 2

    @patch("mdfactory.orchestration.progress._get_block_status", return_value="")
    @patch("rich.live.Live")
    def test_keyboard_interrupt_reraised(self, _mock_live, _mock_block):
        """KeyboardInterrupt during the loop is caught, printed, and re-raised."""
        import pytest

        with pytest.raises(KeyboardInterrupt):
            run_progress_loop(self._progress(), update=MagicMock(side_effect=KeyboardInterrupt))
