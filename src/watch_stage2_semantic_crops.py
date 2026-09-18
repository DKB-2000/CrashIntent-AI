"""Single-start, bounded local watcher for the Stage2 semantic crop probe."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WATCH = ROOT / "artifacts/stage2-semantic-crops-watch-20260916"
RESULT = ROOT / "artifacts/stage2-semantic-crops-20260916/report.json"
SCRIPT = ROOT / "src/compare_stage2_semantic_crops.py"
TIMEOUT_SECONDS = 1800


def write(status: str, **extra) -> None:
    data = {"status": status, "watcher_pid": os.getpid(), "updated_unix": time.time(),
            "timeout_seconds": TIMEOUT_SECONDS, "retries": 0, "result_report": str(RESULT.relative_to(ROOT)), **extra}
    tmp = WATCH / "status.tmp"
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(WATCH / "status.json")


def main() -> None:
    WATCH.mkdir(parents=True, exist_ok=True)
    try:
        with (WATCH / "start.lock").open("x") as f:
            f.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError("Stage2 semantic crop watcher has already started") from exc
    write("STARTING")
    with (WATCH / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(SCRIPT)], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        write("RUNNING", child_pid=process.pid)
        try:
            code = process.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
            write("FAILED_TIMEOUT", child_pid=process.pid, exit_code=process.returncode)
            return
    try:
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        result_status = result.get("status")
    except Exception as exc:
        result_status = f"UNREADABLE: {exc!r}"
    write("COMPLETE_VALIDATED" if code == 0 and result_status == "COMPLETE_VALIDATED" else "FAILED",
          child_pid=process.pid, exit_code=code, result_status=result_status)


if __name__ == "__main__":
    main()
