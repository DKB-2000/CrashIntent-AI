"""Bounded single-start watcher for Stage2 semantic crop seed stability."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WATCH=ROOT/"artifacts/stage2-semantic-crop-seeds-watch-20260916"
RESULT=ROOT/"artifacts/stage2-semantic-crop-seeds-20260916/report.json"
SCRIPT=ROOT/"src/run_stage2_semantic_crop_seeds.py"
TIMEOUT=3600

def write(status,**extra):
    data={"status":status,"watcher_pid":os.getpid(),"updated_unix":time.time(),"timeout_seconds":TIMEOUT,"retries":0,"result_report":str(RESULT.relative_to(ROOT)),**extra}
    tmp=WATCH/"status.tmp";tmp.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8");tmp.replace(WATCH/"status.json")

def main():
    WATCH.mkdir(parents=True,exist_ok=True)
    try:
        with (WATCH/"start.lock").open("x") as f:f.write(str(os.getpid()))
    except FileExistsError as exc:raise RuntimeError("Seed stability watcher already started") from exc
    with (WATCH/"run.log").open("w",encoding="utf-8") as log:
        p=subprocess.Popen([sys.executable,str(SCRIPT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0));write("RUNNING",child_pid=p.pid)
        try: code=p.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            p.kill();p.wait(timeout=30);write("FAILED_TIMEOUT",child_pid=p.pid,exit_code=p.returncode);return
    try: result_status=json.loads(RESULT.read_text(encoding="utf-8")).get("status")
    except Exception as exc:result_status=f"UNREADABLE: {exc!r}"
    write("COMPLETE_VALIDATED" if code==0 and result_status=="COMPLETE_VALIDATED" else "FAILED",child_pid=p.pid,exit_code=code,result_status=result_status)

if __name__=="__main__":main()
