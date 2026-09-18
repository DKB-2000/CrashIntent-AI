"""Extract frozen official ResNet18 features at approximately 10 fps for Nexar training videos."""
from __future__ import annotations
import csv,hashlib,json,time
from pathlib import Path
import gc
import cv2,numpy as np,torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights,resnet18
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data_raw/nexar-stage2-temporal-train-20260917';MANIFEST=ROOT/'artifacts/stage2-nexar-temporal-pool-20260917/manifest-sized.csv';WEIGHTS=ROOT/'artifacts/stage2-final-validation-20260914/models/resnet.pt';OUT=ROOT/'artifacts/stage2-nexar-temporal-features-20260917';EXPECTED_SHA='b55eb2f9f3f559e2101e507d9acc49a5e0768f9d710eeb6a0912080e57595860'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(status,**extra):
 OUT.mkdir(parents=True,exist_ok=True);p=OUT/'status.tmp';p.write_text(json.dumps({'status':status,'updated_unix':time.time(),**extra},indent=2)+'\n',encoding='utf-8');p.replace(OUT/'status.json')
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'start.lock').write_text(str(time.time()),encoding='utf-8');(OUT/'features').mkdir(exist_ok=True)
 if sha(WEIGHTS)!=EXPECTED_SHA:raise RuntimeError('Backbone hash mismatch')
 cv2.setNumThreads(1);torch.set_num_threads(1);model=resnet18(weights=None);model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True);model.fc=nn.Identity();model.eval();transform=ResNet18_Weights.IMAGENET1K_V1.transforms();rows=list(csv.DictReader(MANIFEST.open(encoding='utf-8-sig',newline='')));summary=[];started=time.monotonic()
 with torch.inference_mode():
  for n,r in enumerate(rows,1):
   target=OUT/'features'/(Path(r['file_name']).stem+'.pt')
   if target.exists():
    item=torch.load(target,map_location='cpu',weights_only=True);feat=item['features'];idx=item['source_frame_indices'];fps=float(item['fps']);step=max(1,round(fps/10))
    if feat.ndim!=2 or feat.shape!=(len(idx),512) or not torch.isfinite(feat).all():raise RuntimeError(f'Existing feature validation failed: {target}')
    summary.append({'file_name':r['file_name'],'feature_rows':len(idx),'fps':fps,'step':step,'event_index':int(item['event_index']),'alert_index':int(item['alert_index']),'feature_sha256':sha(target)})
    write('EXTRACTING',completed=n,total=len(rows),current=r['file_name'],feature_rows=sum(x['feature_rows'] for x in summary),seconds=time.monotonic()-started,resumed=True)
    continue
   path=DATA/r['file_name'];cap=cv2.VideoCapture(str(path));fps=float(cap.get(cv2.CAP_PROP_FPS));total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));step=max(1,round(fps/10));frames=[];indices=[];chunks=[];i=0
   while True:
    ok,bgr=cap.read()
    if not ok:break
    if i%step==0:
     frames.append(transform(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))));indices.append(i)
     if len(frames)==8:chunks.append(model(torch.stack(frames)).float().half());frames=[]
    i+=1
   cap.release()
   if frames:chunks.append(model(torch.stack(frames)).float().half())
   feat=torch.cat(chunks);idx=torch.tensor(indices,dtype=torch.int32);times=idx.float()/fps;event=float(r['time_of_event']);alert=float(r['time_of_alert']);ei=int(torch.argmin(abs(times-event)));ai=int(torch.argmin(abs(times-alert)))
   if feat.shape!=(len(indices),512) or not torch.isfinite(feat).all() or i!=total:raise RuntimeError(f'Feature validation failed {r["file_name"]}: decoded {i}, declared {total}')
   item={'features':feat,'source_frame_indices':idx,'fps':fps,'decoded_frames':i,'event_seconds':event,'alert_seconds':alert,'event_index':ei,'alert_index':ai};tmp=target.with_suffix('.tmp');torch.save(item,tmp);tmp.replace(target);summary.append({'file_name':r['file_name'],'feature_rows':len(indices),'fps':fps,'step':step,'event_index':ei,'alert_index':ai,'feature_sha256':sha(target)});del feat,idx,chunks,item;gc.collect()
   write('EXTRACTING',completed=n,total=len(rows),current=r['file_name'],feature_rows=sum(x['feature_rows'] for x in summary),seconds=time.monotonic()-started)
 with (OUT/'index.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 write('COMPLETE_VALIDATED',completed=len(rows),total=len(rows),feature_rows=sum(x['feature_rows'] for x in summary),index_sha256=sha(OUT/'index.csv'),seconds=time.monotonic()-started)
if __name__=='__main__':main()
