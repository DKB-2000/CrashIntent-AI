"""Watch and independently validate the approved CPU CCD pretraining comparison."""
import hashlib,json,re,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import stage2_pipeline as p
from train_stage2_controls import predictions
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/stage2-ccd-pretrain-20260910'
CV=ROOT/'artifacts/stage2-source-cv-20260910'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 deadline=time.monotonic()+86400
 while not (OUT/'report.json').exists():
  if (OUT/'status.json').exists():
   state=json.loads((OUT/'status.json').read_text())
   if state['status']=='FAILED':raise RuntimeError(state)
  if time.monotonic()>deadline:raise TimeoutError('Result not available after 24h; inspect process before restart')
  time.sleep(30)
 torch.set_num_threads(2)
 report=json.loads((OUT/'report.json').read_text());assert report['status']=='COMPLETED'
 config=json.loads((OUT/'protocol.json').read_text())
 labels=ROOT/'data_raw/ccd/Crash-1500.txt';assert sha(labels)==config['label_sha256']
 official={}
 for line in labels.read_text().splitlines():
  m=re.fullmatch(r'(\d+),\[([^\]]+)\],(\d+),([^,]+),.*',line);assert m
  official[m[1]]=(m[4],list(map(int,m[2].split(','))).index(1))
 pre=pd.read_csv(OUT/'pretrain-labels.csv',dtype={'ID':str,'source_id':str})
 assert len(pre)==1296 and pre.ID.is_unique
 excluded=set(config['excluded_manual_sources'])
 assert set(pre.ID)=={i for i,(s,_) in official.items() if s not in excluded}
 for row in pre.to_dict('records'):assert (row['source_id'],row['t_collision'])==official[row['ID']]
 provenance=json.loads((OUT/'feature-provenance.json').read_text());assert len(provenance)==1296
 for row in provenance:
  assert row['source_id']==official[row['ID']][0]
  assert sha(OUT/'features'/f"{row['ID']}.pt")==row['feature_sha256']
  assert sha(ROOT/'data_raw/ccd/videos/Crash-1500'/f"{row['ID']}.mp4")==row['video_sha256']
 cache=ROOT/'artifacts/stage2-controls-20260909/features.pt'
 assert sha(cache)==config['manual_features_sha256']
 features=torch.load(cache,weights_only=True,map_location='cpu')
 assert not set(features)&set(pre.ID)
 checked={}
 for arm in ['scratch','pretrained']:
  frame=pd.read_csv(OUT/f'{arm}_oof.csv',dtype={'ID':str,'source_id':str})
  assert len(frame)==66 and frame.ID.is_unique and set(frame.ID)==set(features)
  expected=frame.set_index('ID');seen=set()
  for fold,held in enumerate(config['folds']):
   train=pd.read_csv(CV/f'fold-{fold}/train.csv',dtype={'ID':str,'source_id':str})
   val=pd.read_csv(CV/f'fold-{fold}/validation.csv',dtype={'ID':str,'source_id':str})
   assert not set(held)&(set(train.source_id)|set(pre.source_id))
   assert set(val.source_id)==set(held) and not seen&set(val.ID)
   seen.update(val.ID)
   model=p.Stage2Temporal();model.load_state_dict(torch.load(OUT/f'fold-{fold}/{arm}.pt',weights_only=True,map_location='cpu')['model'],strict=True)
   for q in predictions(model,val.to_dict('records'),features,'cpu'):
    assert all(q[k]==expected.loc[q['ID'],'pred_'+k] for k in p.OUTPUT_COLUMNS[1:])
  assert seen==set(features)
  values=[]
  for key,target in [('collision_frame','t_collision'),('entry_frame','t_entry')]:
   v=float(((frame['pred_'+key]-frame[target]).abs()/frame.fps<=.3+1e-9).mean());values.append(v)
   assert abs(v-report['results'][arm][key+'_within_0.3s'])<1e-12
  for key in ['evasion_space','entry_side']:
   v=float((frame['pred_'+key]==frame[key]).mean());values.append(v)
   assert abs(v-report['results'][arm][key+'_accuracy'])<1e-12
  assert abs(np.mean(values)-report['results'][arm]['development_mean'])<1e-12
  checked[arm]=values
 result={'status':'PASS','checks':['All 1296 CCD labels and sources independently mapped','All pretraining video and feature hashes','No held-out source in pretraining or fold fine-tuning','All 10 CV checkpoints reproduced','Complete OOF coverage and independent metrics'],'metrics':checked}
 (OUT/'independent-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':
 try:main()
 except Exception as error:
  (OUT/'independent-validation.json').write_text(json.dumps({'status':'FAILED','error':repr(error)}),encoding='utf-8')
  raise
