"""Audit exact submitted Stage3 candidate pair and paired errors, without hidden data."""
import ast,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np,pandas as pd,torch
import evaluate_stage3_motion_slices as ev
OUT=Path('artifacts/stage3-drop-87243-20260911');OUT.mkdir(exist_ok=True)
oldzip=Path('artifacts/stage3-motion-submit-candidate-20260910/submit.zip');newzip=Path('artifacts/final-stage2-short-stage3-20260911/submit.zip')
oldrun=Path('artifacts/stage3-representation-20260910/20260910-motion');newrun=Path('artifacts/stage3-temporal-motion-20260910/20260910-short')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def nodes(text):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
with zipfile.ZipFile(oldzip) as a,zipfile.ZipFile(newzip) as b:
 changes=[n for n in a.namelist() if a.read(n)!=b.read(n)]
 na,nb=nodes(a.read('inference.py').decode('utf-8-sig')),nodes(b.read('inference.py').decode('utf-8-sig'))
 changed_nodes=[n for n in na if na[n]!=nb.get(n)]
 assert changed_nodes==['s3_motion_features','s3_motion_load'],changed_nodes
 assert set(changes)=={'inference.py','model/stage2/best.pt','model/stage3/best.pt'},changes
 checkpoints=[]
 for z,run in [(a,oldrun),(b,newrun)]:
  c=torch.load(io.BytesIO(z.read('model/stage3/best.pt')),map_location='cpu',weights_only=True)
  r=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
  assert all(torch.equal(c['model'][k],v) for k,v in r['model'].items())
  assert all(torch.equal(c[k],r[k]) for k in ['mean','std','mask'])
  checkpoints.append(dict(seed=c['seed'],mode=c['mode'],windows=c.get('windows',[5,15]),weights_exact=True))
f,masks=ev.slices();preds=[pd.read_csv(r/'validation-predictions.csv') for r in [oldrun,newrun]]
for p in preds:pd.testing.assert_frame_equal(p[['ID','sample_index']].rename(columns={'sample_index':'endpoint'}),f[['ID','endpoint']])
t=f.steer_label.to_numpy();old=preds[0].steer_label.to_numpy();new=preds[1].steer_label.to_numpy();moving=f.accel_label.ne('STOPPED').to_numpy()
rows=[];metrics=[]
for name,mask in masks.items():
 active=mask&moving
 if not active.any():continue
 for label in ev.STEER:
  m=active&(t==label)
  rows.append(dict(slice=name,label=label,support=int(m.sum()),old_correct=int((m&(old==t)).sum()),new_correct=int((m&(new==t)).sum()),gained=int((m&(old!=t)&(new==t)).sum()),lost=int((m&(old==t)&(new!=t)).sum())))
 for tag,p in [('old',old),('short',new)]:
  cm,support,f1,recall,precision=ev.metrics(t[active],p[active],ev.STEER)
  metrics.append(dict(slice=name,model=tag,samples=int(active.sum()),macro_f1=float(f1.mean()),balanced_accuracy=float(recall.mean()),recall=dict(zip(ev.STEER,recall.tolist())),f1=dict(zip(ev.STEER,f1.tolist())),confusion_matrix=cm.tolist()))
flips=[]
for sid,g in f.groupby('ID',sort=False):
 idx=g.index.to_numpy();valid=moving[idx[1:]]&moving[idx[:-1]]
 stable=valid&(t[idx[1:]]==t[idx[:-1]])
 for tag,p in [('old',old),('short',new)]:
  sw=p[idx[1:]]!=p[idx[:-1]]
  flips.append(dict(ID=sid,model=tag,moving_pairs=int(valid.sum()),switches=int((sw&valid).sum()),stable_truth_pairs=int(stable.sum()),stable_truth_switches=int((sw&stable).sum())))
trans=pd.DataFrame(dict(truth=t[moving],old=old[moving],short=new[moving])).groupby(['truth','old','short']).size().reset_index(name='frames')
pd.DataFrame(rows).to_csv(OUT/'paired-errors.csv',index=False);pd.DataFrame(flips).to_csv(OUT/'switches.csv',index=False);trans.to_csv(OUT/'prediction-transitions.csv',index=False)
public=[]
for folder in [oldzip.parent,newzip.parent]:
 p=pd.read_csv(folder/'expected-stage3.csv');public.append(dict(folder=str(folder),rows=len(p),switches=int(p.steer_label.ne(p.steer_label.shift()).sum()-1),distribution=p.steer_label.value_counts().to_dict()))
report=dict(status='PASS',scope='Exact local candidate pair and external labeled validation; hidden evaluation labels unavailable, server ZIP hash unverified',zip_sha256={'old':sha(oldzip),'short':sha(newzip)},changed_zip_entries=changes,changed_functions=changed_nodes,checkpoint_matches=checkpoints,metrics=metrics,paired_changes=int((old[moving]!=new[moving]).sum()),moving_rows=int(moving.sum()),switches=pd.DataFrame(flips).groupby('model').sum(numeric_only=True).to_dict('index'),public_unlabeled=public,source_prediction_hashes={str(r/'validation-predictions.csv'):sha(r/'validation-predictions.csv') for r in [oldrun,newrun]},script_sha256=sha(Path(__file__)))
(OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['metrics','source_prediction_hashes']},indent=2))
print(json.dumps([m for m in metrics if m['slice']=='all'],indent=2))
print(pd.DataFrame(rows).query('slice == "all"').to_string(index=False))

