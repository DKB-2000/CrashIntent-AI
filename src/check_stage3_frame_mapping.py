import json
from pathlib import Path
import cv2,numpy as np,pandas as pd
out=Path('artifacts/stage3-label-audit-20260910');a=json.loads((out/'audit.json').read_text());timing=pd.DataFrame(a['timing']);print(timing.sort_values('nominal_10hz_max_drift_s',ascending=False).head(5).to_string(index=False))
rows=[]
ids=list(dict.fromkeys([timing.sort_values('nominal_10hz_max_drift_s').iloc[-1].ID,'COMMA2K19_C1_0134','COMMA2K19_C1_0080']))
base=Path('artifacts/stage3-comma-chunk1')
for sid in ids:
 def decode(path):
  cap=cv2.VideoCapture(str(path));frames=[]
  while True:
   ok,f=cap.read()
   if not ok:break
   frames.append(cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(160,90)))
  cap.release();return frames
 raw=decode(base/'segments'/sid/'video.hevc');converted=decode(base/'converted'/sid/'videos'/f'{sid}.mp4');ft=np.load(base/'segments'/sid/'global_pose/frame_times').reshape(-1)
 assert len(raw)==len(ft)
 for j in np.linspace(5,len(converted)-6,20,dtype=int):
  candidates=range(max(0,2*j-2),min(len(raw),2*j+3));errors={i:float(np.mean((raw[i].astype(float)-converted[j])**2)) for i in candidates};best=min(errors,key=errors.get)
  rows.append(dict(ID=sid,output_index=int(j),best_raw_index=int(best),raw_offset=int(best-2*j),time_offset_s=float(ft[best]-ft[2*j]),mse=errors[best]))
pd.DataFrame(rows).to_csv(out/'frame-mapping.csv',index=False)
print(pd.DataFrame(rows).groupby(['ID','raw_offset']).size().to_string());print('MAX_ALIGNMENT_S',max(abs(r['time_offset_s']) for r in rows))
# Check every decoded review clip length.
for r in a['selected']:
 cap=cv2.VideoCapture(str(out/f'{r["index"]:02d}.mp4'));n=0
 while cap.read()[0]:n+=1
 cap.release();assert n==60
print('Review clips: 7 x 60 frames decoded')
