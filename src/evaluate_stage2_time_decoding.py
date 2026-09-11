"""Fixed +/-0.3s posterior mass decoding on existing held-out Stage2 models."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import stage2_pipeline as p
from analyze_stage2_source_cv import details, summary

ROOT=Path(__file__).resolve().parents[1]
CV=ROOT/'artifacts/stage2-source-cv-20260910'
OUT=ROOT/'artifacts/stage2-time-decoding-20260910'


def window_index(logits, fps):
    prob=logits.softmax(-1)
    radius=int(np.floor(.3*fps+1e-9))
    mass=F.conv1d(prob[:,None,:],torch.ones(1,1,2*radius+1),padding=radius)[:,0,:]
    # Independently sum the actual timestamp tolerance, including truncated edge windows.
    pos=np.arange(prob.shape[1])
    oracle=np.array([prob[0].numpy()[np.abs(pos-i)/fps<=.3+1e-9].sum() for i in pos])
    assert np.allclose(mass[0].numpy(),oracle,atol=1e-6)
    chosen=int(mass.argmax(1))
    assert oracle[chosen]>=oracle.max()-1e-6
    return chosen


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    assert json.loads((CV/'independent-validation.json').read_text())['status']=='PASS'
    protocol=json.loads((CV/'protocol.json').read_text())
    cache=ROOT/'artifacts/stage2-controls-20260909/features.pt'
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==protocol['features_sha256']
    features=torch.load(cache,weights_only=True,map_location='cpu')
    report={'status':'RUNNING','tolerance_seconds':.3,'scene_policy':'Existing scene head recomputed at chosen event positions; no training or pooling change', 'scope':'Exploratory on reused development folds; not official score', 'input_feature_sha256':protocol['features_sha256'],'checkpoint_sha256':{},'results':{}}
    # Boundary and noninteger FPS checks before real inference.
    with torch.inference_mode():
        for fps in [10.,29.97,30.]:
            for values in [[10.,0.,0.,0.],[0.,0.,0.,10.],[0.]*12]:
                window_index(torch.tensor([values]),fps)
        for name in protocol['comparison']:
            expected=pd.read_csv(CV/f'{name}_oof.csv',dtype={'ID':str,'source_id':str}).set_index('ID')
            rows=[]
            predictions={v:[] for v in ['argmax','collision_window','entry_window','both_window']}
            for fold,held in enumerate(protocol['folds']):
                folder=CV/f'fold-{fold}'
                train=pd.read_csv(folder/'train.csv',dtype={'ID':str,'source_id':str})
                val=pd.read_csv(folder/'validation.csv',dtype={'ID':str,'source_id':str})
                assert not set(train.source_id)&set(val.source_id)
                assert set(val.source_id)==set(held)
                checkpoint=folder/f'{name}.pt'
                report['checkpoint_sha256'][f'{fold}/{name}']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                model=p.Stage2Temporal()
                model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu')['model'],strict=True)
                model.eval()
                for r in val.to_dict('records'):
                    rows.append(dict(r,fold=fold))
                    cl,el,h=model.logits(features[r['ID']].unsqueeze(0))
                    ca,ea=int(cl.argmax(1)),int(el.argmax(1))
                    cw,ew=window_index(cl,r['fps']),window_index(el,r['fps'])
                    for variant,(ci,ei) in {'argmax':(ca,ea),'collision_window':(cw,ea),'entry_window':(ca,ew),'both_window':(cw,ew)}.items():
                        scene=model.scene(torch.cat([h[:,ci],h[:,ei]],1))
                        q=dict(ID=r['ID'],collision_frame=ci,entry_frame=ei,evasion_space=int(scene[:,:2].argmax(1)),entry_side=['LEFT','RIGHT'][int(scene[:,2:].argmax(1))])
                        if variant=='argmax':
                            assert all(q[k]==expected.loc[r['ID'],'pred_'+k] for k in p.OUTPUT_COLUMNS[1:])
                        predictions[variant].append(q)
            assert len(rows)==len(features) and {r['ID'] for r in rows}==set(features)
            report['results'][name]={}
            for variant,pred in predictions.items():
                data=details(rows,pred)
                frame=pd.DataFrame(data)
                frame.to_csv(OUT/f'{name}_{variant}.csv',index=False)
                score=summary(rows,pred)
                # Independent correctness column mean must match aggregate scoring.
                checks=[float(frame[k+'_ok'].mean()) for k in p.OUTPUT_COLUMNS[1:]]
                assert abs(np.mean(checks)-score['development_mean'])<1e-12
                score['fold_means']={str(f):float(frame[frame.fold==f][[k+'_ok' for k in p.OUTPUT_COLUMNS[1:]]].to_numpy().mean()) for f in range(5)}
                report['results'][name][variant]=score
            print(name,{v:round(m['development_mean'],4) for v,m in report['results'][name].items()},flush=True)
    report['status']='COMPLETED'
    report['validation']='PASS: source exclusion, OOF coverage, original predictions reproduced, direct window sums, aggregate metric cross-check'
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
