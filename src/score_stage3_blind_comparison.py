"""Compare fixed predictions only after blind human labels; keep two sampling groups separate."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
import evaluate_stage3_motion_slices as ev
ROOT=Path('artifacts/stage3-blind-review-20260911');DATA=Path('data/stage3-blind-review-20260911')
subprocess.run([sys.executable,'src/review_stage3_human.py','--check','--video-dir',str(DATA/'videos'),'--output-dir',str(DATA/'labels')],check=True)
labels=pd.read_csv(DATA/'labels/labels_review.csv');manifest=pd.read_csv(ROOT/'source-manifest.csv')
keys=['ID','sample_index'];assert not labels.duplicated(keys).any()
labels=labels.merge(manifest[['ID','group']],on='ID',validate='many_to_one')
valid=labels[(labels.reviewed==1)&(labels.motion_status=='MOVING')&labels.steer_label.isin(ev.STEER)]
if not len(valid):
 print('WAITING_FOR_HUMAN_LABELS: No eligible human-reviewed frames; no model scores calculated.');sys.exit(0)
results=[]
for name in ['old','short']:
 pred=pd.read_csv(ROOT/(name+'-predictions.csv'));assert not pred.duplicated(keys).any()
 joined=valid.merge(pred[keys+['steer_label']],on=keys,how='left',suffixes=('_truth','_pred'),validate='one_to_one');assert joined.steer_label_pred.isin(ev.STEER).all()
 for group,g in joined.groupby('group'):
  cm,support,f1,recall,precision=ev.metrics(g.steer_label_truth,g.steer_label_pred,ev.STEER)
  results.append(dict(model=name,group=group,frames=len(g),videos=g.ID.nunique(),macro_f1=float(f1.mean()),per_class={label:dict(support=int(support[k]),f1=float(f1[k]),recall=float(recall[k])) for k,label in enumerate(ev.STEER)},confusion=cm.tolist()))
out=dict(status='HUMAN_REVIEWED_SUBSET_SCORED',scope='Human development review; report random and disagreement groups separately; no low-speed ground truth assumed',eligible_frames=len(valid),total_frames=len(labels),results=results,labels_sha256=hashlib.sha256((DATA/'labels/labels_review.csv').read_bytes()).hexdigest())
(ROOT/'human-comparison.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
