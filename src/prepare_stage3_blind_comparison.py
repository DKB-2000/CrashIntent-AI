"""Prepare blind source-separated human review of a fixed Stage3 model pair."""
import hashlib,json,random,shutil
from pathlib import Path
import cv2,numpy as np,pandas as pd,torch
import stage3_motion_inference as old
import stage3_short_motion_inference as short
ROOT=Path('artifacts/stage3-blind-review-20260911');DATA=Path('data/stage3-blind-review-20260911')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2),encoding='utf-8')
def main():
 torch.set_num_threads(2);cv2.setNumThreads(1);ROOT.mkdir(exist_ok=True);DATA.mkdir(parents=True,exist_ok=True);(DATA/'videos').mkdir(exist_ok=True);assert not (ROOT/'status.json').exists(), 'Refuse overwrite existing run'
 rows=[]
 for line in Path('data_raw/ccd/Crash-1500.txt').read_text().splitlines():
  sid=line.split(',',1)[0];tail=line.split('],',1)[1].split(',')
  rows.append(dict(original_ID=sid,source_id=tail[1],timing=tail[2],weather=tail[3],ego=tail[4]))
 frame=pd.DataFrame(rows);exclude=set(frame[frame.original_ID.isin([f'{i:06}' for i in range(1,6)])].source_id)
 holdout=pd.read_csv('artifacts/stage1-full-20260910/phone_holdout.csv',dtype=str).source_id
 exclude.update(frame[frame.original_ID.isin(holdout)].source_id)
 groups={str(s):g.to_dict('records') for s,g in frame.groupby('source_id') if s not in exclude}
 rng=random.Random(20260911);sources=sorted(groups);rng.shuffle(sources)
 chosen=[rng.choice(groups[s]) for s in sources[:36]];assert len(chosen)==36
 paths=[Path('artifacts/stage3-motion-submit-candidate-20260910/best.pt'),Path('artifacts/final-stage2-short-stage3-20260911/best.pt')]
 save(ROOT/'plan.json',dict(seed=20260911,pool=chosen,unselected_source_exclusions=sorted(exclude),natural_count=12,diagnostic_count=12,selection='First12 random source-balanced videos before prediction; remaining24 ranked by paired turn-to-straight disagreement, choose12; never hidden data',models={str(p):sha(p) for p in paths},script_sha256=sha(Path(__file__)),labels='None; predictions never become human truth',scope='Stage3 source-disjoint CCD review; CCD may be used by Stage1/2, short crash clips not target natural distribution'))
 models=[old.s3_motion_load(paths[0],torch.device('cpu')),short.s3_motion_load(paths[1],torch.device('cpu'))];allpred={};audit=[]
 for n,r in enumerate(chosen):
  path=Path('data_raw/ccd/videos/Crash-1500')/(r['original_ID']+'.mp4');flow=old.s3_motion_features(path);cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);cap.release();assert abs(fps-10)<.01
  cur=flow[:,:134];new=[]
  for t in range(len(cur)):
   h=cur[max(1,t-4):t+1] if t else np.zeros((1,134),np.float32)
   new.append(np.r_[cur[t],h[-3:].mean(0),h.mean(0)])
  features=[flow,np.asarray(new,np.float32)];pred=[]
  for (model,mean,std),x in zip(models,features):
   a,s=old.s3_motion_logits(model,x,mean,std,torch.device('cpu'));pred.append(pd.DataFrame(dict(sample_index=np.arange(len(a)),accel_label=np.array(old._S3M_ACCEL)[a.argmax(1)],steer_label=np.array(old._S3M_STEER)[s.argmax(1)])))
  direction=pred[0].steer_label.isin(['LEFT','RIGHT']) & pred[1].steer_label.eq('STRAIGHT')
  allpred[r['original_ID']]=pred;audit.append(dict(**r,pool_index=n,frames=len(flow),fps=fps,sha256=sha(path),turn_to_straight=int(direction.sum()),disagreements=int(pred[0].steer_label.ne(pred[1].steer_label).sum())))
  save(ROOT/'status.json',dict(status='PREPARING',completed=n+1,total=36));print('prepared',n+1,36,flush=True)
 natural=audit[:12];diagnostic=sorted(audit[12:],key=lambda r:(-r['turn_to_straight'],-r['disagreements'],r['pool_index']))[:12]
 selected=natural+diagnostic;predfiles=[[],[]];manifest=[]
 for i,r in enumerate(selected,1):
  sid=f'REVIEW_{i:03}';src=Path('data_raw/ccd/videos/Crash-1500')/(r['original_ID']+'.mp4');dst=DATA/'videos'/(sid+'.mp4');shutil.copyfile(src,dst);assert sha(dst)==r['sha256']
  manifest.append(dict(ID=sid,group='source_random' if i<=12 else 'disagreement_diagnostic',**r))
  for k,p in enumerate(allpred[r['original_ID']]):
   p=p.copy();p.insert(0,'ID',sid);predfiles[k].append(p)
 for tag,parts in zip(['old','short'],predfiles):pd.concat(parts,ignore_index=True).to_csv(ROOT/(tag+'-predictions.csv'),index=False)
 pd.DataFrame(manifest).to_csv(ROOT/'source-manifest.csv',index=False);pd.DataFrame(audit).to_csv(ROOT/'pool-selection-private.csv',index=False)
 assert len(set(r['source_id'] for r in manifest))==24
 save(ROOT/'status.json',dict(status='READY_FOR_HUMAN_REVIEW',videos=24,frames=sum(r['frames'] for r in manifest),human_labels=0,models_scored=False))
 print('READY',str(DATA),flush=True)
if __name__=='__main__':main()

