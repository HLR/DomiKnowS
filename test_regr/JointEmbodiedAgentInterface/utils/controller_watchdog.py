"""Launch a training child with a deadline only while controller PPO is active.

A hung CUDA call cannot reliably be interrupted by a Python exception in the
training process. This independent parent terminates its own child process group
and exits 124; it never edits checkpoints or automatically retries a round.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
import threading
import time


def _stop_child(process, grace):
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass  # The child can exit between poll() and the signal.
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def run(command, log_path, *, stall_seconds=300.0, terminate_grace=5.0):
    if not command or not math.isfinite(stall_seconds) or stall_seconds <= 0:
        raise ValueError("a command and finite positive stall deadline are required")
    if not math.isfinite(terminate_grace) or terminate_grace < 0:
        raise ValueError("terminate grace must be finite and nonnegative")
    log_path = Path(log_path)
    status_path = log_path.with_suffix(log_path.suffix + ".watchdog.json")
    if log_path.exists() or status_path.exists():
        raise FileExistsError("use a new watchdog log; existing run evidence is preserved")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status = {"command": list(command), "stall_seconds": stall_seconds,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", env=env,
                                   start_new_session=os.name == "posix", bufsize=1)
        status["child_pid"] = process.pid
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        lines = Queue()

        def read_output():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        controller_active = False
        last_progress = time.monotonic()
        reader_done = False
        timed_out = False
        try:
            while process.poll() is None or not reader_done:
                try:
                    line = lines.get(timeout=0.1)
                except Empty:
                    line = ""
                if line is None:
                    reader_done = True
                elif line:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                    if "VLABench controller PPO" in line:
                        controller_active = "VLABench controller PPO update complete" not in line
                        last_progress = time.monotonic()
                if (not timed_out and controller_active and process.poll() is None
                        and time.monotonic() - last_progress > stall_seconds):
                    timed_out = True
                    message = (f"[controller-watchdog] no controller PPO progress for {stall_seconds:g}s; "
                               "terminating training child; resume from the last completed-round checkpoint\n")
                    log.write(message)
                    log.flush()
                    print(message, end="", flush=True)
                    _stop_child(process, terminate_grace)
            process.wait()
        finally:
            _stop_child(process, terminate_grace)
            reader.join(timeout=2)
            process.stdout.close()
            status.update(child_exit_code=process.returncode, controller_stall=timed_out,
                          exit_code=124 if timed_out else process.returncode,
                          ended_utc=datetime.now(timezone.utc).isoformat())
            status_path.write_text(json.dumps(status, indent=2) + "\n")
    return status["exit_code"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--stall-seconds", type=float, default=300.0)
    parser.add_argument("--terminate-grace", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run(command, args.log, stall_seconds=args.stall_seconds,
               terminate_grace=args.terminate_grace)


if __name__ == "__main__":
    raise SystemExit(main())
