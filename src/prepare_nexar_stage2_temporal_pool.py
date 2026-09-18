"""Select a deterministic, metadata-balanced Nexar temporal training pool."""
from __future__ import annotations
import csv,hashlib,json,random
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];META=ROOT/'data_raw/stage2-validation-nexar-20260914/metadata.csv';HOLD=ROOT/'data_raw/stage2-validation-nexar-20260914';OUT=ROOT/'artifacts/stage2-nexar-temporal-pool-20260917';N=200;SEED=20260917;REV='aa97deda5a59f00bb7187739053b7c72e14374df'
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir();rows=list(csv.DictReader(META.open(encoding='utf-8-sig',newline='')));hold={p.name for p in HOLD.glob('*.mp4')};groups=defaultdict(list)
 for r in rows:
  if r['file_name'] in hold:continue
  gap=float(r['time_of_event'])-float(r['time_of_alert']);bucket='short' if gap<1 else 'medium' if gap<2 else 'long'
  groups[(r['scene'],r['weather'],r['light_conditions'],bucket)].append(r)
 rng=random.Random(SEED)
 for v in groups.values():rng.shuffle(v)
 selected=[];keys=sorted(groups)
 while len(selected)<N:
  moved=False
  for k in keys:
   if groups[k] and len(selected)<N:selected.append(groups[k].pop());moved=True
  if not moved:break
 if len(selected)!=N:raise RuntimeError('Insufficient rows')
 fields=['file_name','time_of_event','time_of_alert','light_conditions','weather','scene','gap_seconds','url','purpose']
 with (OUT/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in selected:w.writerow({**{k:r[k] for k in fields[:6]},'gap_seconds':float(r['time_of_event'])-float(r['time_of_alert']),'url':f"https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/resolve/{REV}/train/positive/{r['file_name']}?download=true",'purpose':'TRAIN_ONLY_EXCLUDES_FROZEN_50'})
 report={'status':'POOL_READY_SIZE_UNCHECKED','selected':N,'seed':SEED,'available_after_holdout':len(rows)-len(hold),'frozen_holdout_videos':len(hold),'strata':len({(r['scene'],r['weather'],r['light_conditions'],('short' if float(r['time_of_event'])-float(r['time_of_alert'])<1 else 'medium' if float(r['time_of_event'])-float(r['time_of_alert'])<2 else 'long')) for r in selected}),'manifest_sha256':hashlib.sha256((OUT/'manifest.csv').read_bytes()).hexdigest(),'download_started':False}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
