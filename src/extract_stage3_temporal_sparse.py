"""Execution-only optimization: same predeclared features, sparse decode, two workers."""
import time,json
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np,cv2,pandas as pd
import stage3_temporal_motion_experiment as e
import compare_stage3_representations as c

def sparse(path,ends):
 need={i for t in ends for i in range(max(0,int(t)-30),int(t)+1)}
 cap=cv2.VideoCapture(str(path));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert cap.isOpened();flow_rows=np.zeros((n,134),np.float32);prev=None;prev_i=-2
 for i in range(n):
  ok=cap.grab();assert ok
  if i not in need:continue
  ok,bgr=cap.retrieve();assert ok
  rgb=cv2.resize(c.p.prepare_frame(bgr).transpose(1,2,0),(96,96),interpolation=cv2.INTER_AREA);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
  if prev is not None and prev_i==i-1:
   flow=np.zeros((96,96,2),np.float32) if np.array_equal(prev,gray) else cv2.calcOpticalFlowFarneback(prev,gray,None,.5,3,15,3,5,1.2,0)
   flow_rows[i]=np.r_[cv2.resize(flow,(8,8),interpolation=cv2.INTER_AREA).reshape(-1),np.percentile(np.linalg.norm(flow,axis=2),[10,50,90,99]),flow.mean((0,1))].astype(np.float32)
  prev=gray;prev_i=i
 cap.release();return flow_rows

def main():
 cv2.setNumThreads(1);root=e.ROOT;plan=json.loads((root/'plan.json').read_text());assert plan['script_sha256']==c.sha(e.__file__)
 amendment=root/'execution-optimization.json'
 if not amendment.exists():c.save(amendment,dict(reason='Reduce decoded preprocessing to required causal31frame unions; no model, sample, temporal or training change',script_sha256=c.sha(__file__),original_script_sha256=c.sha(e.__file__),workers=2,result_status_before_change='no models trained',parity='All retained baseline rows exact, sparse vs dense replay on one existing train video before launch'))
 else:assert json.loads(amendment.read_text())['script_sha256']==c.sha(__file__)
 groups=[]
 for part in ('train','validation'):
  for sid,g in pd.read_csv(root/(part+'-samples.csv')).groupby('ID'):groups.append((part,sid,g))
 first=groups[0];g=first[2];ends=sorted(g.endpoint.tolist());dense=e.dense_train((c.DATA/g.video.iloc[0]).resolve());reduced=sparse((c.DATA/g.video.iloc[0]).resolve(),ends)
 for t in ends:
  for _,(w1,w2,_) in e.ARMS.items():np.testing.assert_array_equal(e.motion_at(dense,int(t),w1,w2),e.motion_at(reduced,int(t),w1,w2))
 print('PASS sparse/dense real video parity',flush=True)
 def worker(task):
  part,sid,g=task;target=root/'cache'/(sid+'.npz');info=root/'cache'/(sid+'.json')
  if target.exists() and info.exists():
   meta=json.loads(info.read_text());assert meta['sha256']==c.sha(target) and meta['script_sha256']==c.sha(e.__file__);return sid,'cached'
  source=c.ROOT/'cache'/(sid+'.npz');meta=json.loads((c.ROOT/'cache'/(sid+'.json')).read_text());assert c.sha(source)==meta['features_sha256'];path=(c.DATA/g.video.iloc[0]).resolve();assert c.sha(path)==meta['video_sha256']
  with np.load(source) as z:old=z['features'];ends=z['endpoints']
  if part=='validation':assert ends.tolist()==list(range(len(ends)));flow=old[:,1280:1414].copy()
  else:flow=sparse(path,ends)
  np.testing.assert_array_equal(np.stack([e.motion_at(flow,int(t),5,15) for t in ends]),old[:,1280:])
  values={arm:np.concatenate([old[:,:1280],np.stack([e.motion_at(flow,int(t),v[0],v[1]) for t in ends])],axis=1) for arm,v in e.ARMS.items()};np.savez_compressed(target,endpoints=ends,**values)
  c.save(info,dict(ID=sid,sha256=c.sha(target),source_cache_sha256=c.sha(source),video_sha256=meta['video_sha256'],script_sha256=c.sha(e.__file__),extractor_sha256=c.sha(__file__),part=part,frames=len(flow),baseline_exact=True));return sid,'new'
 start=time.monotonic()
 with ThreadPoolExecutor(max_workers=2) as ex:
  for k,future in enumerate(as_completed([ex.submit(worker,t) for t in groups]),1):
   sid,state=future.result();c.save(root/'status.json',dict(status='EXTRACTING',videos_completed=k,total=187,seconds=time.monotonic()-start,workers=2));print(k,187,sid,state,round(time.monotonic()-start),flush=True)
 c.save(root/'status.json',dict(status='FEATURES_COMPLETE',videos=187,sparse_dense_parity=True,seconds=time.monotonic()-start))
if __name__=='__main__':main()
