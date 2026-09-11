"""Independently regenerate Stage2 context inputs and saved scene predictions."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import stage2_pipeline as p

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/stage2-scene-context-20260910'
CV=ROOT/'artifacts/stage2-source-cv-20260910'
torch.set_num_threads(2)
r=json.loads((OUT/'report.json').read_text())
assert r['status']=='COMPLETED'
protocol=json.loads((OUT/'protocol.json').read_text())
cache=ROOT/'artifacts/stage2-controls-20260909/features.pt'
assert hashlib.sha256(cache.read_bytes()).hexdigest()==protocol['feature_sha256']
features=torch.load(cache,weights_only=True,map_location='cpu')
frames={name:pd.read_csv(OUT/f'{name}_oof.csv',dtype={'ID':str,'source_id':str}) for name in r['results'] if name!='original'}
seen=set()
with torch.no_grad():
 for fold,held in enumerate(protocol['folds']):
  train=pd.read_csv(OUT/f'fold-{fold}/train.csv',dtype={'ID':str,'source_id':str})
  val=pd.read_csv(OUT/f'fold-{fold}/validation.csv',dtype={'ID':str,'source_id':str})
  assert not set(train.source_id)&set(val.source_id)
  assert set(val.source_id)==set(held) and not seen&set(val.ID)
  seen.update(val.ID)
  path=CV/f'fold-{fold}/soft_mixed_15.pt'
  assert hashlib.sha256(path.read_bytes()).hexdigest()==r['checkpoint_sha256'][str(fold)]
  model=p.Stage2Temporal();model.load_state_dict(torch.load(path,weights_only=True)['model'],strict=True);model.eval()
  heads={name:copy.deepcopy(model.scene) for name in frames}
  for name,head in heads.items():
   head.load_state_dict(torch.load(OUT/f'fold-{fold}/{name}.pt',weights_only=True),strict=True);head.eval()
  for row in val.to_dict('records'):
   cl,el,h=model.logits(features[row['ID']].unsqueeze(0))
   ci,ei=int(cl.argmax()),int(el.argmax())
   xs={}
   xs['point']=torch.cat([h[0,ci],h[0,ei]])
   vectors=[]
   for center in [ci,ei]:
    indices=[i for i in range(len(h[0])) if abs(i-center)/row['fps']<=.5+1e-9]
    vectors.append(sum(h[0,i] for i in indices)/len(indices))
   xs['window']=torch.cat(vectors)
   for name,head in heads.items():
    score=head(xs[name.split('_')[0]][None,:])[0]
    q=frames[name].set_index('ID').loc[row['ID']]
    assert (ci,ei)==(int(q.pred_collision_frame),int(q.pred_entry_frame))
    assert int(score[:2].argmax())==q.pred_evasion_space
    assert ['LEFT','RIGHT'][int(score[2:].argmax())]==q.pred_entry_side
assert seen==set(features)
checks={}
for name,frame in frames.items():
 assert frame.ID.is_unique and set(frame.ID)==seen
 vals=[]
 for key,target in [('collision_frame','t_collision'),('entry_frame','t_entry')]:
  value=float(((frame['pred_'+key]-frame[target]).abs()/frame.fps<=.3+1e-9).mean())
  assert abs(value-r['results'][name][key+'_within_0.3s'])<1e-12
  vals.append(value)
 for key in ['evasion_space','entry_side']:
  value=float((frame['pred_'+key]==frame[key]).mean());vals.append(value)
  assert abs(value-r['results'][name][key+'_accuracy'])<1e-12
 assert abs(np.mean(vals)-r['results'][name]['development_mean'])<1e-12
 checks[name]=float(np.mean(vals))
result={'status':'PASS','checks':['Raw feature to frozen temporal logits regeneration','Independent pooling by explicit timestamp indices','All 30 saved heads reproduce OOF predictions','Source exclusion and exact OOF coverage','Independent four metric recomputation'],'development_means':checks}
(OUT/'independent-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
