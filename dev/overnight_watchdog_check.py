"""Linux-only CPU check that the deadline watchdog stops detached workers."""

import json
import os
import signal
import sys
import tempfile
from pathlib import Path

from dev.overnight_worker import command_run


def run():
    with tempfile.TemporaryDirectory(prefix="rc3-watchdog-check-") as directory:
        root = Path(directory)
        child = "import time; time.sleep(60)"
        parent = (
            "import json,subprocess,sys,time; "
            f"child=subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True); "
            "print(json.dumps({'child_pid':child.pid}),flush=True); time.sleep(60)"
        )
        outcome = command_run([sys.executable, "-c", parent], root / "timeout.log",
                              os.environ.copy(), 1.0)
        rows = [json.loads(line) for line in (root / "timeout.log").read_text().splitlines()]
        assert len(rows) == 1, rows
        child_pid = rows[0]["child_pid"]
        status = Path(f"/proc/{child_pid}/status")
        state = "absent"
        if status.exists():
            state = next(line.split()[1] for line in status.read_text().splitlines()
                         if line.startswith("State:"))
        assert outcome["timeout"] and outcome["returncode"] == -signal.SIGKILL, outcome
        assert state in ("absent", "Z"), (child_pid, state)
        normal = command_run([sys.executable, "-c", "print('normal completion')"],
                             root / "normal.log", os.environ.copy(), 2.0)
        assert normal["returncode"] == 0 and not normal["timeout"], normal
        return dict(timeout=outcome, detached_child_state=state,
                    detached_child_stopped=True, normal_completion=normal, gpu_used=False)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2), flush=True)
