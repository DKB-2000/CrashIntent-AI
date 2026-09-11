"""Wait for Stage2 CV and independently validate coverage, source exclusion and metrics."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import stage2_pipeline as p
from train_stage2_controls import predictions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/stage2-source-cv-20260910'


def main():
    deadline = time.monotonic()+7500
    while not (OUT/'report.json').exists():
        state = json.loads((OUT/'status.json').read_text())
        if state['status'] == 'FAILED':
            raise RuntimeError(state)
        if time.monotonic()>deadline:
            raise TimeoutError('CV result missing')
        time.sleep(30)
    torch.set_num_threads(2)
    report=json.loads((OUT/'report.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert report['status']=='COMPLETED'
    cache=ROOT/'artifacts/stage2-controls-20260909/features.pt'
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==protocol['features_sha256']
    features=torch.load(cache,map_location='cpu',weights_only=True)
    all_ids=set(features)
    checked={}
    for name, claimed in report['results'].items():
        frame=pd.read_csv(OUT/f'{name}_oof.csv',dtype={'ID':str,'source_id':str})
        assert len(frame)==len(all_ids) and set(frame.ID)==all_ids and frame.ID.is_unique
        metric={}
        for key,target in [('collision_frame','t_collision'),('entry_frame','t_entry')]:
            err=(frame['pred_'+key]-frame[target]).abs()/frame.fps
            metric[key+'_within_0.3s']=float((err<=.3+1e-9).mean())
            metric[key+'_mae_seconds']=float(err.mean())
        for key in ['evasion_space','entry_side']:
            metric[key+'_accuracy']=float((frame['pred_'+key]==frame[key]).mean())
        metric['development_mean']=float(np.mean([metric['collision_frame_within_0.3s'],metric['entry_frame_within_0.3s'],metric['evasion_space_accuracy'],metric['entry_side_accuracy']]))
        for key,value in metric.items():
            assert abs(value-claimed[key])<1e-12,(name,key)
        seen=set()
        for fold,groups in enumerate(protocol['folds']):
            folder=OUT/f'fold-{fold}'
            train=pd.read_csv(folder/'train.csv',dtype={'ID':str,'source_id':str})
            val=pd.read_csv(folder/'validation.csv',dtype={'ID':str,'source_id':str})
            assert not set(train.source_id)&set(val.source_id)
            assert set(val.source_id)==set(groups)
            assert set(train.ID)|set(val.ID)==all_ids
            assert not seen&set(val.ID)
            seen.update(val.ID)
            if name!='train_constant':
                model=p.Stage2Temporal()
                model.load_state_dict(torch.load(folder/f'{name}.pt',map_location='cpu',weights_only=True)['model'],strict=True)
                expected=frame.set_index('ID')
                for q in predictions(model,val.to_dict('records'),features,'cpu'):
                    for key in p.OUTPUT_COLUMNS[1:]:
                        assert q[key]==expected.loc[q['ID'],'pred_'+key]
        assert seen==all_ids
        checked[name]=metric
    result={'status':'PASS','checks':['OOF IDs covered exactly once','Held-out source exclusion in every fold','Metrics independently recomputed','All learned checkpoint predictions reproduced'], 'results':checked}
    (OUT/'independent-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        (OUT/'independent-validation.json').write_text(json.dumps({'status':'FAILED','error':repr(error)}),encoding='utf-8')
        raise
