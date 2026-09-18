"""Compare ordered-joint crop probes against the frozen argmax crop seed results."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/stage2-crop-ordered-seeds-20260916'
BASE=ROOT/'artifacts/stage2-semantic-crop-seeds-20260916/report.json'
SCRIPT=ROOT/'src/compare_stage2_semantic_crops.py'
SEEDS=(20260825,20260826,20260827,20260916,20260917)
METRICS=('collision_frame_within_0.3s','entry_frame_within_0.3s','evasion_space_accuracy','entry_side_accuracy','development_mean','source_equal_mean')

def main():
 if OUT.exists():raise FileExistsError(OUT)
 baseline=json.loads(BASE.read_text(encoding='utf-8'))
 if baseline.get('status')!='COMPLETE_VALIDATED' or baseline.get('completed')!=5:raise RuntimeError('Baseline seed experiment is not validated')
 OUT.mkdir();report={'status':'RUNNING','completed':0,'seeds':list(SEEDS),'event_policy':'ordered_joint','results':{},'gate':None}
 (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 try:
  for seed in SEEDS:
   rel=f'artifacts/stage2-crop-ordered-seeds-20260916/seed-{seed}';env=os.environ.copy();env['STAGE2_SEMANTIC_CROPS_OUT']=rel;env['STAGE2_SEMANTIC_CROPS_SEED']=str(seed);env['STAGE2_EVENT_POLICY']='ordered_joint'
   with (OUT/f'seed-{seed}.log').open('w',encoding='utf-8') as log:subprocess.run([sys.executable,str(SCRIPT)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   child=json.loads((ROOT/rel/'report.json').read_text(encoding='utf-8'))
   if child.get('status')!='COMPLETE_VALIDATED' or child.get('seed')!=seed or child.get('event_policy')!='ordered_joint':raise RuntimeError(f'Invalid child result {seed}')
   ordered={k:child['conditions']['semantic_crop'][k] for k in METRICS}
   argmax_child=json.loads((ROOT/f'artifacts/stage2-semantic-crop-seeds-20260916/seed-{seed}/report.json').read_text(encoding='utf-8'))
   argmax={k:argmax_child['conditions']['semantic_crop'][k] for k in METRICS}
   report['results'][str(seed)]={'argmax_crop':argmax,'ordered_crop':ordered,'delta':{k:ordered[k]-argmax[k] for k in METRICS}}
   report['completed']+=1;(OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
  required=('entry_frame_within_0.3s','evasion_space_accuracy','entry_side_accuracy','development_mean','source_equal_mean')
  report['aggregate']={k:{'argmax_mean':sum(report['results'][str(s)]['argmax_crop'][k] for s in SEEDS)/5,'ordered_mean':sum(report['results'][str(s)]['ordered_crop'][k] for s in SEEDS)/5,'delta_mean':sum(report['results'][str(s)]['delta'][k] for s in SEEDS)/5,'ordered_no_worse_all_seeds':all(report['results'][str(s)]['delta'][k]>=-1e-12 for s in SEEDS)} for k in METRICS}
  passed=all(report['aggregate'][k]['ordered_no_worse_all_seeds'] for k in required) and report['aggregate']['development_mean']['delta_mean']>0 and report['aggregate']['source_equal_mean']['delta_mean']>0
  report['gate']={'status':'PASS' if passed else 'FAIL','rule':'Entry, evasion, side, development mean, and source-equal mean no worse in every seed; mean development and source-equal strictly improve.'}
  report['status']='COMPLETE_VALIDATED';(OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 except Exception as exc:
  report['status']='FAILED';report['error']=repr(exc);(OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');raise

if __name__=='__main__':main()
