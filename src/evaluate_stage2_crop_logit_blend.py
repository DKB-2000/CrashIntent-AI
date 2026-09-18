"""Evaluate fixed shared/task-specific OOF logit blends without retraining."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd
from analyze_stage2_source_cv import summary

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stage2-task-specific-ensemble-20260917'
OUT=ROOT/'artifacts/stage2-crop-logit-blend-20260917'
SEEDS=(20260825,20260826,20260827,20260916,20260917)
ALPHAS=(.25,.5,.75)  # task-specific weight; .50 is the preregistered primary.
LOGITS=('pred_evasion_logit_0','pred_evasion_logit_1','pred_side_logit_0','pred_side_logit_1')
METRICS=('evasion_space_accuracy','entry_side_accuracy','development_mean','source_equal_mean')

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

def predictions(frame:pd.DataFrame):
 return [{'ID':str(r.ID),'collision_frame':int(r.pred_collision_frame),'entry_frame':int(r.pred_entry_frame),
          'evasion_space':int(r.pred_evasion_logit_1>r.pred_evasion_logit_0),
          'entry_side':('LEFT','RIGHT')[int(r.pred_side_logit_1>r.pred_side_logit_0)]}
         for r in frame.itertuples(index=False)]

def main():
 if OUT.exists():raise FileExistsError(OUT)
 parent=json.loads((BASE/'report.json').read_text(encoding='utf-8'))
 if parent.get('status')!='COMPLETE_VALIDATED' or parent.get('completed')!=5:raise RuntimeError('Parent ensemble is not validated')
 OUT.mkdir();shared_frames=[];task_frames=[];inputs={'parent_report':sha(BASE/'report.json')}
 for seed in SEEDS:
  folder=BASE/f'seed-{seed}';sp=folder/'semantic_shared_oof.csv';tp=folder/'semantic_crop_oof.csv'
  s=pd.read_csv(sp,dtype={'ID':str,'source_id':str}).sort_values('ID').reset_index(drop=True)
  t=pd.read_csv(tp,dtype={'ID':str,'source_id':str}).sort_values('ID').reset_index(drop=True)
  if list(s.ID)!=list(t.ID) or any(c not in s or c not in t for c in LOGITS):raise RuntimeError(f'Invalid seed OOF {seed}')
  shared_frames.append(s);task_frames.append(t);inputs[f'shared_{seed}']=sha(sp);inputs[f'task_{seed}']=sha(tp)
 ids=list(shared_frames[0].ID)
 if any(list(f.ID)!=ids for f in shared_frames+task_frames):raise RuntimeError('Cross-seed ID mismatch')
 shared=shared_frames[0].copy();task=task_frames[0].copy()
 for c in LOGITS:
  shared[c]=sum(f[c] for f in shared_frames)/len(SEEDS);task[c]=sum(f[c] for f in task_frames)/len(SEEDS)
 truth_cols=('ID','path','t_collision','t_entry','evasion_space','entry_side','source_id','fps','frames')
 truth=shared.loc[:,truth_cols].to_dict('records');shared_pred=predictions(shared);shared_sum=summary(truth,shared_pred)
 conditions={};frames={}
 for alpha in ALPHAS:
  blend=shared.copy()
  for c in LOGITS:blend[c]=(1-alpha)*shared[c]+alpha*task[c]
  pred=predictions(blend);score=summary(truth,pred)
  source_delta={s:score['by_source'][s]['development_mean']-shared_sum['by_source'][s]['development_mean'] for s in shared_sum['by_source']}
  conditions[str(alpha)]={'metrics':{k:score[k] for k in METRICS},'delta':{k:score[k]-shared_sum[k] for k in METRICS},
                          'worst_source_delta':min(source_delta.values()),'source_delta':source_delta}
  frames[str(alpha)]=pred
 primary=conditions['0.5'];passed=(all(primary['delta'][k]>=-1e-12 for k in METRICS) and
   primary['delta']['development_mean']>0 and primary['delta']['source_equal_mean']>0 and primary['worst_source_delta']>=-.10-1e-12)
 for alpha,pred in frames.items():pd.DataFrame(pred).to_csv(OUT/f'blend-task-weight-{alpha}.csv',index=False)
 report={'status':'COMPLETE_VALIDATED','primary_task_weight':.5,'diagnostic_task_weights':[.25,.75],
         'shared_metrics':{k:shared_sum[k] for k in METRICS},'conditions':conditions,
         'gate':{'status':'PASS' if passed else 'FAIL','rule':'Primary alpha=.50: all four metrics no worse; development and source-equal strictly improve; worst per-source delta >= -0.10.'},
         'input_sha256':inputs}
 (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
