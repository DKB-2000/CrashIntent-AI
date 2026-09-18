"""Finite monitor for the already-started 5-35m/s sensor scout."""
import json,os,time
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'artifacts/stage3-civic-5-35-class-scout-20260916'
execution=json.loads((P/'execution.json').read_text());pid=int(execution['pid'])
(P/'watch-status.json').write_text(json.dumps({'status':'RUNNING','pid':os.getpid(),'child_pid':pid,'timeout_seconds':1800},indent=2))
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
    try:os.kill(pid,0);alive=True
    except OSError:alive=False
    if not alive:break
    time.sleep(5)
report=P/'report.json'
if report.exists() and json.loads(report.read_text())['status']=='COMPLETE_VALIDATED':
    (P/'watch-status.json').write_text(json.dumps({'status':'COMPLETE'},indent=2))
else:
    state={'status':'FAILED','error':'Worker ended or timed out without COMPLETE_VALIDATED report'}
    (P/'watch-status.json').write_text(json.dumps(state,indent=2));(P/'status.json').write_text(json.dumps(state,indent=2))
