"""Independent nested selection, source separation and checkpoint verification."""
import json,time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import stage2_pipeline as p
from train_stage2_controls import predictions
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/stage2-regularization-20260911'


def score(rows,pred):
 truth=pd.DataFrame(rows).set_index('ID');q=pd.DataFrame(pred).set_index('ID')
 assert q.index.is_unique and set(q.index)==set(truth.index);q=q.loc[truth.index]
 correct=pd.DataFrame(index=truth.index)
 for key,target in [('collision_frame','t_collision'),('entry_frame','t_entry')]:correct[key]=(q[key]-truth[target]).abs()/truth.fps<=.3+1e-9
 for key in ['evasion_space','entry_side']:correct[key]=q[key]==truth[key]
 values=correct.mean(axis=1)
 return float(values.mean()),float(values.groupby(truth.source_id).mean().mean())


def main():
 deadline=time.monotonic()+43200
 while not (OUT/'report.json').exists():
  if (OUT/'status.json').exists():
   s=json.loads((OUT/'status.json').read_text())
   if s['status']=='FAILED':raise RuntimeError(s)
  if time.monotonic()>deadline:raise TimeoutError('Training result not available')
  time.sleep(30)
 torch.set_num_threads(2)
 r=json.loads((OUT/'report.json').read_text());assert r['status']=='COMPLETED'
 protocol=json.loads((OUT/'protocol.json').read_text())
 features=torch.load(ROOT/'artifacts/stage2-controls-20260909/features.pt',weights_only=True,map_location='cpu')
 original=torch.load(ROOT/'artifacts/stage2-ccd-pretrain-20260910/collision-pretrain.pt',weights_only=True,map_location='cpu')['model']
 frames={mode:pd.read_csv(OUT/f'{mode}_oof.csv',dtype={'ID':str,'source_id':str}) for mode in r['results']}
 seen=set()
 for fold in range(5):
  folder=OUT/f'fold-{fold}'
  data={name:pd.read_csv(folder/f'{name}.csv',dtype={'ID':str,'source_id':str}) for name in ['train','validation','inner_train','inner_validation']}
  assert set(data['train'].ID)==set(data['inner_train'].ID)|set(data['inner_validation'].ID)
  groups=[set(data[k].source_id) for k in ['inner_train','inner_validation','validation']]
  assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
  assert not seen&set(data['validation'].ID);seen.update(data['validation'].ID)
  recomputed=[]
  for mode in protocol['modes']:
   history=json.loads((folder/f'{mode}_inner.json').read_text())
   assert [h['epoch'] for h in history]==list(range(1,16))
   for h in history:
    _,v=score(data['inner_validation'].to_dict('records'),h['predictions']);assert abs(v-h['score'])<1e-12
   best=max(history,key=lambda h:h['score']);recomputed.append({'mode':mode,'epoch':best['epoch'],'score':best['score']})
   cp=torch.load(folder/f'{mode}.pt',weights_only=True,map_location='cpu');assert cp['epoch']==best['epoch']
   if mode=='frozen_encoder':assert all(torch.equal(v,original[k]) for k,v in cp['model'].items() if k.startswith('r.'))
   model=p.Stage2Temporal();model.load_state_dict(cp['model'],strict=True)
   expected=frames[mode].set_index('ID')
   for q in predictions(model,data['validation'].to_dict('records'),features,'cpu'):
    assert all(q[k]==expected.loc[q['ID'],'pred_'+k] for k in p.OUTPUT_COLUMNS[1:])
  decision=json.loads((folder/'selection.json').read_text())
  assert decision['modes']==recomputed and decision['selected']==max(recomputed,key=lambda d:d['score'])
  chosen=frames[decision['selected']['mode']].set_index('ID').loc[data['validation'].ID]
  actual=frames['inner_selected'].set_index('ID').loc[data['validation'].ID]
  assert chosen.equals(actual)
 assert seen==set(features)
 summary={}
 for mode,frame in frames.items():
  assert len(frame)==66 and frame.ID.is_unique and set(frame.ID)==seen
  pred=[dict(ID=row['ID'],**{k:row['pred_'+k] for k in p.OUTPUT_COLUMNS[1:]}) for row in frame.to_dict('records')]
  a,b=score(frame.to_dict('records'),pred)
  assert abs(a-r['results'][mode]['development_mean'])<1e-12 and abs(b-r['results'][mode]['source_equal_mean'])<1e-12
  summary[mode]={'development_mean':a,'source_equal_mean':b}
 report={'status':'PASS','checks':['Three-way source disjointness','Inner scores and earliest-epoch selection independently recalculated','Setting selection uses only inner results','All 15 outer models reproduce predictions','Frozen encoder weights unchanged','Complete OOF and independent aggregate metrics'],'results':summary}
 (OUT/'independent-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
 try:main()
 except Exception as e:
  (OUT/'independent-validation.json').write_text(json.dumps({'status':'FAILED','error':repr(e)}));raise
