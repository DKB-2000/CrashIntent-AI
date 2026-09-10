"""Score predictions only on explicitly reviewed, known, moving human labels."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labels',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
l=pd.read_csv(a.labels);r=pd.read_csv(a.predictions);keys=['ID','sample_index'];classes=['LEFT','STRAIGHT','RIGHT']
assert not l.duplicated(keys).any() and not r.duplicated(keys).any()
valid=l[(l.reviewed==1)&(l.motion_status=='MOVING')&l.steer_label.isin(classes)]
assert len(valid)>0,'No eligible human-reviewed moving frames'
m=valid.merge(r[keys+['steer_label']],on=keys,how='left',suffixes=('_truth','_pred'),validate='one_to_one');assert m.steer_label_pred.isin(classes).all(),'Missing or invalid predictions'
c=pd.crosstab(m.steer_label_truth,m.steer_label_pred).reindex(index=classes,columns=classes,fill_value=0).to_numpy();den=c.sum(0)+c.sum(1);f=np.divide(2*np.diag(c),den,out=np.zeros(3),where=den!=0)
result=dict(scope='Human development set, not official competition score',rows_total=len(l),rows_reviewed=int((l.reviewed==1).sum()),rows_scored=len(m),macro_f1=float(f.mean()),class_support=dict(zip(classes,c.sum(1).tolist())),per_class_f1=dict(zip(classes,f.tolist())),confusion=c.tolist())
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
