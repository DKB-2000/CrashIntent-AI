"""Matched feature interventions, not physically rendered scene changes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import compare_stage3_representations as base
import stage3_motion_inference as inference

P=Path(__file__).resolve().parents[1]
REF=P/'artifacts/stage3-representation-20260910'
OUT=P/'artifacts/stage3-region-attenuation-20260914-v3'
SEEDS=[20260910,20260911,20260912]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=False)
    f=pd.read_csv(REF/'validation-samples.csv');assert len(f)==18603 and f.route.nunique()==4
    masks={}
    for arm,cols in [('sides_half',[0,1,6,7]),('center_half',[2,3,4,5])]:
        mask=np.zeros(402,bool)
        for window in range(3):
            for row in range(8):
                for col in cols:mask[window*134+(row*8+col)*2:window*134+(row*8+col)*2+2]=True
        assert mask.sum()==192;masks[arm]=mask
    assert not (masks['sides_half']&masks['center_half']).any()
    inputs={str(p.relative_to(P)):sha(p) for p in [Path(__file__),REF/'validation-samples.csv',Path(inference.__file__),Path(base.__file__)]}
    for seed in SEEDS:
        for name in ['model.pt','report.json','validation-logits.npz']:
            path=REF/f'{seed}-motion'/name;inputs[str(path.relative_to(P))]=sha(path)
    plan=dict(arms=['baseline',*masks],seeds=SEEDS,
       operation='Raw optical-flow grid x/y scaled0.5 in selected32of64cells,all1/5/15windows;192dimensions per arm; rest unchanged.',
       limitation='Global6summary dimensions/window intentionally unchanged; this is an off-manifold feature sensitivity probe, not an actual scene removal or deployment candidate.',
       data='Existing4development routes only,all18603timesteps;no Civic selection,no training',inputs_sha256=inputs)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    x=np.zeros((len(f),402),np.float32)
    for sid,g in f.groupby('ID'):
        path=REF/'cache'/(sid+'.npz');meta=json.loads(path.with_suffix('.json').read_text());assert sha(path)==meta['features_sha256']
        inputs[str(path.relative_to(P))]=sha(path)
        with np.load(path) as z:
            idx={int(v):i for i,v in enumerate(z['endpoints'])}
            x[g.index]=z['features'][[idx[int(v)] for v in g.endpoint],1280:]
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    results=[];routes=[];allpred=[]
    for seed in SEEDS:
        ckpath=REF/f'{seed}-motion/model.pt';ck=torch.load(ckpath,weights_only=True,map_location='cpu')
        saved_report=json.loads((ckpath.parent/'report.json').read_text());assert sha(ckpath)==saved_report['checkpoint_sha256']
        model=base.Control();model.load_state_dict(ck['model']);model.eval()
        baseline_pred=None
        for arm in plan['arms']:
            features=x.copy()
            if arm!='baseline':
                mask=masks[arm];features[:,mask]*=.5
                np.testing.assert_array_equal(features[:,~mask],x[:,~mask])
            a,s=inference.s3_motion_logits(model,features,ck['mean'].numpy(),ck['std'].numpy(),torch.device('cpu'))
            pred=s.argmax(1)
            if arm=='baseline':
                with np.load(ckpath.parent/'validation-logits.npz') as z:
                    np.testing.assert_array_equal(a,z['accel']);np.testing.assert_array_equal(s,z['steer'])
                baseline_pred=pred.copy()
            metrics=base.score(f,a.argmax(1),pred)['steer']
            matrix=np.array(metrics['confusion_matrix']);support=matrix.sum(1)
            metrics['recall']=np.divide(matrix.diagonal(),support,out=np.zeros(3),where=support!=0).tolist()
            moving=f.accel.to_numpy()!=3;truth=f.steer.to_numpy()
            turn=moving&(truth!=1);straight=moving&(truth==1)
            correct_before=baseline_pred==truth;correct_after=pred==truth
            results.append(dict(seed=seed,arm=arm,metrics=metrics,
               fixed_straight=int((straight&~correct_before&correct_after).sum()),
               broken_straight=int((straight&correct_before&~correct_after).sum()),
               fixed_turn=int((turn&~correct_before&correct_after).sum()),
               broken_turn=int((turn&correct_before&~correct_after).sum()),changed=int((moving&(pred!=baseline_pred)).sum())))
            for route,g in f.groupby('route'):
                routes.append(dict(seed=seed,arm=arm,route=route,metrics=base.score(g,a[g.index].argmax(1),pred[g.index])['steer']))
            table=f[['ID','endpoint','accel','steer']].copy();table['seed']=seed;table['arm']=arm;table['prediction']=pred;allpred.append(table)
    pd.concat(allpred,ignore_index=True).to_csv(OUT/'predictions.csv',index=False)
    # Recompute confusion from saved output using the stdlib reader.
    import csv
    cm={(s,a):[[0]*3 for _ in range(3)] for s in SEEDS for a in plan['arms']}
    with (OUT/'predictions.csv').open(newline='') as stream:
        for r in csv.DictReader(stream):
            if int(r['accel'])!=3:cm[int(r['seed']),r['arm']][int(r['steer'])][int(r['prediction'])]+=1
    for r in results:
        matrix=cm[r['seed'],r['arm']];assert matrix==r['metrics']['confusion_matrix']
        val=sum(2*matrix[i][i]/(sum(matrix[i])+sum(row[i] for row in matrix)) for i in range(3))/3
        assert abs(val-r['metrics']['macro_f1'])<1e-12
    averages={arm:dict(f1=float(np.mean([r['metrics']['macro_f1'] for r in results if r['arm']==arm])),
                      recall=np.mean([r['metrics']['recall'] for r in results if r['arm']==arm],axis=0).tolist()) for arm in plan['arms']}
    for p,h in inputs.items():assert sha(P/p)==h
    report=dict(status='COMPLETE_VALIDATED',averages=averages,results=results,routes=routes,
                validation='Baseline logits exact,192equal disjoint dimensions,unmodified complementary features,input hashes and independent CSV confusion/F1 PASS',
                decision='DIAGNOSTIC_ONLY_KEEP_BASELINE',scope=plan['limitation'],predictions_sha256=sha(OUT/'predictions.csv'))
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(averages=averages,seed20260910=[r for r in results if r['seed']==20260910]),indent=2))

if __name__=='__main__':main()
