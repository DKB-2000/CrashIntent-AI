"""Download and validate the small official MM-AU detection-label archive only."""
from __future__ import annotations
import hashlib,json,os,tarfile,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data_raw/mm-au-detection-labels-20260917';ART=ROOT/'artifacts/stage2-mmau-label-acquisition-20260917'
TARGET=OUT/'labels.tar.gz';EXPECTED=113457793
URL='https://huggingface.co/datasets/JeffreyChou/MM-AU/resolve/main/MMAU_Det_paper/labels.tar.gz?download=true'
def write(status,**extra):
 ART.mkdir(parents=True,exist_ok=True);p=ART/'status.tmp';p.write_text(json.dumps({'status':status,'updated_unix':time.time(),'target':str(TARGET.relative_to(ROOT)),**extra},indent=2)+'\n',encoding='utf-8');p.replace(ART/'status.json')
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if TARGET.exists() and TARGET.stat().st_size==EXPECTED:write('VERIFYING',bytes=EXPECTED)
 else:
  for attempt in range(1,4):
   try:
    write('DOWNLOADING',attempt=attempt,bytes=TARGET.stat().st_size if TARGET.exists() else 0,expected=EXPECTED)
    request=urllib.request.Request(URL,headers={'User-Agent':'crashvideo-metadata-audit/1.0'})
    with urllib.request.urlopen(request,timeout=60) as src,TARGET.open('wb') as dst:
     while True:
      block=src.read(1024*1024)
      if not block:break
      dst.write(block)
    if TARGET.stat().st_size!=EXPECTED:raise RuntimeError(f'expected {EXPECTED}, got {TARGET.stat().st_size}')
    break
   except Exception as exc:
    write('RETRYING' if attempt<3 else 'FAILED',attempt=attempt,error=repr(exc),bytes=TARGET.stat().st_size if TARGET.exists() else 0)
    if attempt==3:raise
    time.sleep(2**attempt)
 digest=hashlib.sha256(TARGET.read_bytes()).hexdigest()
 with tarfile.open(TARGET,'r:gz') as tf:
  members=tf.getmembers();files=[m for m in members if m.isfile()]
  if not files:raise RuntimeError('Archive has no files')
  unsafe=[m.name for m in members if Path(m.name).is_absolute() or '..' in Path(m.name).parts]
  if unsafe:raise RuntimeError(f'Unsafe archive members: {unsafe[:3]}')
 write('COMPLETE_VALIDATED',bytes=TARGET.stat().st_size,sha256=digest,members=len(members),files=len(files),uncompressed_file_bytes=sum(m.size for m in files))
if __name__=='__main__':main()
