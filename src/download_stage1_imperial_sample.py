"""Bounded resumable acquisition and validation of Imperial recapture sample."""
import hashlib, json, os, pathlib, subprocess, time, zipfile

URL="https://www.commsp.ee.ic.ac.uk/~tt1410/experiments/recapturedetection/resources/SubjectiveTestImages.zip"
ZIP=pathlib.Path("data_raw/imperial-recapture/SubjectiveTestImages.clean.zip")
OUT=pathlib.Path("artifacts/stage1-imperial-recapture-20260917")
EXPECTED=70145784

def save(**kw):
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/"status.json"
    d=json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}; d.update(kw,heartbeat_unix=time.time())
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8');os.replace(q,p)

def main():
    ZIP.parent.mkdir(parents=True,exist_ok=True)
    save(status='DOWNLOADING',pid=os.getpid(),url=URL,expected_bytes=EXPECTED,downloaded_bytes=ZIP.stat().st_size if ZIP.exists() else 0,error=None)
    try:
        # This clean target has a single recorded owner, so an interrupted transfer
        # may be resumed. The older concurrently-written file has a different name.
        if ZIP.exists() and ZIP.stat().st_size > EXPECTED:
            raise ValueError('Clean partial exceeds official Content-Length')
        r=subprocess.run(['curl.exe','--fail','--location','--retry','3','--retry-all-errors','--connect-timeout','30','--max-time','7200','-C','-',URL,'-o',str(ZIP)],timeout=7500)
        if r.returncode: raise RuntimeError(f'curl exit {r.returncode}')
        if ZIP.stat().st_size!=EXPECTED: raise ValueError(f'Unexpected size {ZIP.stat().st_size} != {EXPECTED}')
        with zipfile.ZipFile(ZIP) as z:
            bad=z.testzip(); names=z.namelist()
            if bad: raise ValueError('Bad ZIP member: '+bad)
        sha=hashlib.sha256(ZIP.read_bytes()).hexdigest()
        (OUT/'inventory.json').write_text(json.dumps({'members':len(names),'names':names},indent=2)+'\n')
        save(status='DATA_VALIDATED',downloaded_bytes=ZIP.stat().st_size,members=len(names),sha256=sha)
    except Exception as e:
        save(status='FAILED',downloaded_bytes=ZIP.stat().st_size if ZIP.exists() else 0,error=repr(e));raise
if __name__=='__main__':main()
