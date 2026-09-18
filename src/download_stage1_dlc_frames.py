"""Resumable, bounded acquisition of the public DLC-2021 frame archive."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage1-dlc-frames-acquisition-20260916"
DATA = ROOT / "data_raw/dlc-2021"
TARGET = DATA / "clips.tar"
STATUS = OUT / "status.json"
URL = "ftp://smartengines.com/dlc-2021/clips.tar"
EXPECTED_BYTES = 17_768_312_320
EXPECTED_MD5 = "0758a65d3ddccdd24eba25a98e4ba3c6"


def write(**values):
    state = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    state.update(values, updated_unix=time.time(), pid=os.getpid())
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temp, STATUS)


def digest(path):
    h = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    lock = OUT / "download.lock"
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("DLC frame acquisition already started; inspect status and process")
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    started = time.monotonic()
    try:
        if TARGET.exists() and TARGET.stat().st_size > EXPECTED_BYTES:
            raise ValueError("Existing archive exceeds official size")
        write(status="DOWNLOADING", completed_bytes=TARGET.stat().st_size if TARGET.exists() else 0,
              expected_bytes=EXPECTED_BYTES, attempts=0, source=URL, expected_md5=EXPECTED_MD5, last_error=None)
        errors = []
        for attempt in range(1, 4):
            if time.monotonic() - started > 8 * 3600:
                raise TimeoutError("Eight-hour acquisition deadline")
            write(attempts=attempt, completed_bytes=TARGET.stat().st_size if TARGET.exists() else 0)
            with (OUT / f"curl-{attempt}.log").open("wb") as log:
                process = subprocess.Popen(["curl.exe", "--fail", "--location", "--continue-at", "-",
                                            "--connect-timeout", "30", "--max-time", "21600", URL,
                                            "--output", str(TARGET)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    write(status="DOWNLOADING", attempts=attempt,
                          completed_bytes=TARGET.stat().st_size if TARGET.exists() else 0)
                    time.sleep(30)
            if process.returncode == 0 and TARGET.stat().st_size == EXPECTED_BYTES:
                break
            errors.append({"attempt": attempt, "returncode": process.returncode,
                           "bytes": TARGET.stat().st_size if TARGET.exists() else 0})
            write(status="RETRYING", errors=errors)
        else:
            raise RuntimeError("Download did not reach official byte size after three attempts")
        write(status="VERIFYING_MD5", completed_bytes=TARGET.stat().st_size)
        actual = digest(TARGET)
        if actual != EXPECTED_MD5:
            raise ValueError(f"MD5 mismatch: {actual}")
        probe = subprocess.run(["tar", "-tf", str(TARGET)], cwd=ROOT, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, timeout=1800)
        if probe.returncode:
            raise RuntimeError("tar archive listing failed: " + probe.stderr.decode(errors="replace")[-1000:])
        write(status="ACQUIRED_VALIDATED", completed_bytes=EXPECTED_BYTES, md5=actual,
              validation="official size + MD5 + complete tar listing", errors=errors, last_error=None)
    except Exception as error:
        write(status="FAILED", last_error=repr(error),
              completed_bytes=TARGET.stat().st_size if TARGET.exists() else 0)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
