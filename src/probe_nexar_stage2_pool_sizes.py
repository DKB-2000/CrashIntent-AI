"""Resolve exact remote sizes and local free space for the Nexar Stage2 pool."""
from __future__ import annotations
import csv,json,shutil,time,urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];POOL=ROOT/'artifacts/stage2-nexar-temporal-pool-20260917';MANIFEST=POOL/'manifest.csv'
def head(row):
 for attempt in range(3):
  try:
   req=urllib.request.Request(row['url'],method='HEAD',headers={'User-Agent':'crashvideo-stage2-pool/1.0'})
   with urllib.request.urlopen(req,timeout=60) as r:return row['file_name'],int(r.headers['Content-Length'])
  except Exception:
   if attempt==2:raise
   time.sleep(2**attempt)
def main():
 rows=list(csv.DictReader(MANIFEST.open(encoding='utf-8-sig',newline='')));sizes={}
 with ThreadPoolExecutor(max_workers=8) as ex:
  jobs={ex.submit(head,r):r['file_name'] for r in rows}
  for i,f in enumerate(as_completed(jobs),1):name,size=f.result();sizes[name]=size
 fields=list(rows[0])+['remote_bytes']
 with (POOL/'manifest-sized.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();[w.writerow({**r,'remote_bytes':sizes[r['file_name']]}) for r in rows]
 usage=shutil.disk_usage(ROOT);total=sum(sizes.values());report=json.loads((POOL/'report.json').read_text(encoding='utf-8'));report.update({'status':'POOL_READY_SIZE_VALIDATED','remote_files':len(sizes),'remote_bytes':total,'remote_gib':total/2**30,'min_file_bytes':min(sizes.values()),'max_file_bytes':max(sizes.values()),'local_free_bytes':usage.free,'local_free_gib':usage.free/2**30,'required_free_bytes_policy':total+10*2**30,'space_gate':'PASS' if usage.free>=total+10*2**30 else 'FAIL'})
 (POOL/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
