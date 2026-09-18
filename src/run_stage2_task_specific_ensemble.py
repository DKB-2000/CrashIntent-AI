"""Direct OOF logit-ensemble comparison of shared and task-specific crop probes."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import pandas as pd
from analyze_stage2_source_cv import summary

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/stage2-task-specific-ensemble-20260917'
SCRIPT=ROOT/'src/compare_stage2_semantic_crops.py'
SEEDS=(20260825,20260826,20260827,20260916,20260917)
METRICS=('evasion_space_accuracy','entry_side_accuracy','development_mean','source_equal_mean')
LOGITS=('pred_evasion_logit_0','pred_evasion_logit_1','pred_side_logit_0','pred_side_logit_1')

def predictions(frame: pd.DataFrame):
 out=[]
 for r in frame.to_dict('records'):
  out.append({'ID':str(r['ID']),'collision_frame':int(r['pred_collision_frame']),'entry_frame':int(r['pred_entry_frame']),
              'evasion_space':int(r['pred_evasion_logit_1']>r['pred_evasion_logit_0']),
              'entry_side':('LEFT','RIGHT')[int(r['pred_side_logit_1']>r['pred_side_logit_0'])]})
 return out

def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir();report={'status':'RUNNING','completed':0,'seeds':list(SEEDS),'results':{},'gate':None}
 (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 try:
  shared_frames=[];task_frames=[]
  for seed in SEEDS:
   rel=f'artifacts/stage2-task-specific-ensemble-20260917/seed-{seed}';env=os.environ.copy();env['STAGE2_SEMANTIC_CROPS_OUT']=rel;env['STAGE2_SEMANTIC_CROPS_SEED']=str(seed);env['STAGE2_CROP_FEATURE_POLICY']='task_specific'
   with (OUT/f'seed-{seed}.log').open('w',encoding='utf-8') as log:subprocess.run([sys.executable,str(SCRIPT)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   child=json.loads((ROOT/rel/'report.json').read_text(encoding='utf-8'))
   if child.get('status')!='COMPLETE_VALIDATED' or 'semantic_shared' not in child.get('conditions',{}):raise RuntimeError(f'Invalid child {seed}')
   shared_frames.append(pd.read_csv(ROOT/rel/'semantic_shared_oof.csv',dtype={'ID':str}).sort_values('ID').reset_index(drop=True))
   task_frames.append(pd.read_csv(ROOT/rel/'semantic_crop_oof.csv',dtype={'ID':str}).sort_values('ID').reset_index(drop=True))
   if list(shared_frames[-1].ID)!=list(task_frames[-1].ID) or any(c not in shared_frames[-1] for c in LOGITS):raise RuntimeError(f'Invalid OOF {seed}')
   report['completed']+=1;(OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
  ids=list(shared_frames[0].ID)
  if any(list(f.ID)!=ids for f in shared_frames+task_frames):raise RuntimeError('OOF ID mismatch')
  shared=shared_frames[0].copy();task=task_frames[0].copy()
  for col in LOGITS:
   shared[col]=sum(f[col] for f in shared_frames)/len(SEEDS);task[col]=sum(f[col] for f in task_frames)/len(SEEDS)
  truth_cols=('ID','path','t_collision','t_entry','evasion_space','entry_side','source_id','fps','frames')
  truth=shared.loc[:,truth_cols].to_dict('records')
  shared_pred=predictions(shared);task_pred=predictions(task)
  shared_summary=summary(truth,shared_pred);task_summary=summary(truth,task_pred)
  pd.DataFrame(shared_pred).to_csv(OUT/'shared_ensemble_oof.csv',index=False);pd.DataFrame(task_pred).to_csv(OUT/'task_specific_ensemble_oof.csv',index=False)
  aggregate={k:{'shared':shared_summary[k],'task_specific':task_summary[k],'delta':task_summary[k]-shared_summary[k]} for k in METRICS}
  source_delta={s:task_summary['by_source'][s]['development_mean']-shared_summary['by_source'][s]['development_mean'] for s in shared_summary['by_source']}
  passed=all(aggregate[k]['delta']>=-1e-12 for k in METRICS) and aggregate['development_mean']['delta']>0 and aggregate['source_equal_mean']['delta']>0 and min(source_delta.values())>=-.10-1e-12
  report.update({'aggregate':aggregate,'source_delta':source_delta,'worst_source_delta':min(source_delta.values()),'gate':{'status':'PASS' if passed else 'FAIL','rule':'All four aggregate metrics no worse; development and source-equal strictly improve; worst per-source development delta >= -0.10.'},'status':'COMPLETE_VALIDATED'})
  (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 except Exception as exc:
  report['status']='FAILED';report['error']=repr(exc);(OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');raise
if __name__=='__main__':main()
