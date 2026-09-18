"""Finite local watcher: validate acquisition completion, then run evaluation once."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import time

ACQ = Path("artifacts/stage1-dlc-range-pilot-20260917/status.json")
EVAL = Path("artifacts/stage1-dlc-range-pilot-evaluation-20260917/status.json")
STATE = Path("artifacts/stage1-dlc-range-pipeline-20260917/status.json")
TIMEOUT = 4 * 60 * 60


def save(**values):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(STATE.read_text()) if STATE.exists() else {}
    old.update(values, heartbeat_unix=time.time())
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(old, indent=2) + "\n")
    os.replace(tmp, STATE)


def main():
    start = time.time()
    save(status="WAITING_FOR_DATA", pid=os.getpid(), timeout_seconds=TIMEOUT)
    while time.time() - start < TIMEOUT:
        if ACQ.exists():
            data = json.loads(ACQ.read_text())
            if data.get("status") == "FAILED":
                save(status="FAILED", error="Acquisition failed", acquisition_error=data.get("error")); return 1
            if data.get("status") == "DATA_VALIDATED":
                save(status="EVALUATING", acquired_clips=data.get("completed_clips"))
                result = subprocess.run([str(Path(".venv/Scripts/python.exe")),
                                         "src/evaluate_stage1_dlc_range_pilot.py"])
                evaluated = json.loads(EVAL.read_text()) if EVAL.exists() else {}
                if result.returncode or evaluated.get("status") != "COMPLETE_VALIDATED":
                    save(status="FAILED", error="Evaluation failed", returncode=result.returncode,
                         evaluation_status=evaluated); return 1
                save(status="COMPLETE_VALIDATED", acquired_clips=data.get("completed_clips"),
                     completed_models=evaluated.get("completed_models")); return 0
        time.sleep(30)
        save(status="WAITING_FOR_DATA")
    save(status="FAILED", error="Four-hour pipeline timeout")
    return 1


if __name__ == "__main__": raise SystemExit(main())
