"""Score frozen Stage2 temporal predictions against official Nexar alert/event times."""
from __future__ import annotations
import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/stage2-validation-nexar-20260914';META=ROOT/'data_raw/stage2-validation-nexar-20260914/review-inventory-resolved.csv';PRED=ROOT/'artifacts/stage2-unlabeled-comparison-20260914';OUT=ROOT/'artifacts/stage2-nexar-temporal-score-20260917'
MODELS=('incumbent','ccd_pretrained')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir();meta={r['ID']:r for r in read(META)};manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'));videos={v['ID']:v for v in manifest['videos']}
 report={'status':'RUNNING','scope':'Frozen 50-video Nexar positive validation subset; official alert/event labels; no training or selection','models':{},'inputs':{'metadata_sha256':sha(META),'manifest_sha256':sha(DATA/'manifest.json')}}
 all_rows=[]
 for model in MODELS:
  pred={r['ID']:r for r in read(PRED/f'{model}-predictions.csv')};rows=[]
  if set(pred)!=set(videos) or not set(videos)<=set(meta):raise RuntimeError('ID mismatch')
  for id_ in sorted(videos):
   fmap=read(DATA/videos[id_]['frame_map']);times=[float(r['timestamp_seconds']) for r in fmap];q=pred[id_];m=meta[id_];ci=int(q['collision_frame']);ei=int(q['entry_frame'])
   if not(0<=ci<len(times) and 0<=ei<len(times)):raise RuntimeError(f'Index out of range {id_}')
   event=float(m['reference_event_seconds']);alert=float(m['reference_alert_seconds']);ce=times[ci]-event;ee=times[ei]-alert
   rows.append({'model':model,'ID':id_,'scene':m['scene'],'event_seconds':event,'alert_seconds':alert,'pred_collision_seconds':times[ci],'pred_entry_seconds':times[ei],'collision_signed_error':ce,'entry_signed_error':ee,'collision_abs_error':abs(ce),'entry_abs_error':abs(ee)})
  def metrics(key):
   vals=[r[key] for r in rows];signed=[r[key.replace('_abs_','_signed_')] for r in rows]
   return {'mean_abs_seconds':sum(vals)/len(vals),'median_abs_seconds':statistics.median(vals),'mean_signed_seconds':sum(signed)/len(signed),**{f'within_{x}s':sum(v<=x+1e-12 for v in vals)/len(vals) for x in (.3,.5,1,2,5)}}
  report['models'][model]={'collision_vs_event':metrics('collision_abs_error'),'entry_vs_alert_proxy':metrics('entry_abs_error')};all_rows+=rows
 with (OUT/'per_video.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
 report['status']='COMPLETE_VALIDATED';report['limitations']=['Nexar positives include near-misses as well as collisions.','time_of_alert is an action-needed timestamp and is only a proxy for competition entry_frame.','The 50 videos were previously exposed to model predictions and are validation-only, never training labels.'];report['per_video_sha256']=sha(OUT/'per_video.csv')
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
