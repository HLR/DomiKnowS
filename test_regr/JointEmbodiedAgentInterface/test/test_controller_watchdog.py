import json
import sys

import pytest

from ..utils.controller_watchdog import run


def test_watchdog_terminates_a_stalled_controller_child(tmp_path):
    log = tmp_path / "stalled.log"
    code = "import time; print('VLABench controller PPO 1/4 transition 1/291 backward', flush=True); time.sleep(30)"
    result = run([sys.executable, "-u", "-c", code], log,
                 stall_seconds=0.2, terminate_grace=0.1)
    assert result == 124
    status = json.loads(log.with_suffix(".log.watchdog.json").read_text())
    assert status["controller_stall"] is True
    assert status["child_exit_code"] is not None
    assert "terminating training child" in log.read_text()


def test_watchdog_does_not_time_out_rollouts_or_completed_controller_updates(tmp_path):
    log = tmp_path / "healthy.log"
    code = ("import time; time.sleep(.4); "
            "print('VLABench controller PPO 1/4 transition 1/1 backward',flush=True); "
            "print('VLABench controller PPO update complete',flush=True); time.sleep(.4)")
    assert run([sys.executable, "-u", "-c", code], log, stall_seconds=0.2) == 0
    status = json.loads(log.with_suffix(".log.watchdog.json").read_text())
    assert status["controller_stall"] is False
    with pytest.raises(FileExistsError):
        run([sys.executable, "-c", "pass"], log)
