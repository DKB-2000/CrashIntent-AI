"""Verify paired outputs and finish fixed-sample experiment review."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from compare_stage3_causal_smoothing import metric, sha

P=Path(__file__).resolve().parents[1]
ROOT=P/'artifacts/stage3-straight-sample-control-20260914'
REF=P/'artifacts/stage3-representation-20260910'

def main():
    comparison=json.loads((ROOT/'comparison.json').read_text())
    assert comparison['baseline_exact']
    assert sha(ROOT/'report.json')==comparison['input_plan_sha256']
    assert sha(P/'src/train_stage3_straight_sample_control.py')==comparison['script_sha256']
    f=pd.read_csv(REF/'validation-samples.csv');events=[];pooled=[];hashes={}
    for seed in [20260910,20260911,20260912]:
        for arm in ['baseline','candidate']:
            folder=ROOT/arm/f'{seed}-motion'
            report=json.loads((folder/'report.json').read_text())
            assert sha(folder/'model.pt')==report['checkpoint_sha256']
            with np.load(folder/'validation-logits.npz') as z:
                logits=z['steer'];assert logits.shape==(len(f),3) and np.isfinite(logits).all()
                if arm=='baseline':
                    with np.load(REF/f'{seed}-motion/validation-logits.npz') as old:
                        for key in ['accel','steer']:np.testing.assert_array_equal(z[key],old[key])
            pred=logits.argmax(1);csv=pd.read_csv(folder/'validation-predictions.csv')
            assert np.array_equal(csv.ID,f.ID) and np.array_equal(csv.sample_index,f.endpoint)
            assert np.array_equal(csv.steer_label,np.array(['LEFT','STRAIGHT','RIGHT'])[pred])
            for route in ['ALL']+sorted(f.route.unique()):
                g=f if route=='ALL' else f[f.route==route]
                m=metric(g,pred[g.index]);saved=next(r for r in comparison['rows'] if (r['seed'],r['arm'],r['route'])==(seed,arm,route))
                assert abs(m['f1']-saved['macro_f1'])<1e-12
                np.testing.assert_allclose(m['recall'],saved['recall'],atol=1e-12,rtol=0)
                if route=='ALL':pooled.append(dict(seed=seed,arm=arm,**m))
            for sid,g in f.groupby('ID',sort=False):
                assert np.all(np.diff(g.endpoint)==1)
                truth=g.steer.to_numpy();moving=g.accel.to_numpy()!=3;p=pred[g.index]
                for t in range(10,len(g)-12):
                    before,after=truth[t-1],truth[t]
                    if before==after or (before!=1 and after!=1):continue
                    if not (moving[t-10:t+10].all() and np.all(truth[t-10:t]==before) and np.all(truth[t:t+10]==after)):continue
                    hits=[k for k in range(11) if np.all(truth[t:t+k+3]==after) and np.all(p[t+k:t+k+3]==after)]
                    events.append(dict(seed=seed,arm=arm,ID=sid,endpoint=int(g.endpoint.iloc[t]),kind='turn_start' if before==1 else 'turn_end',detected=bool(hits),delay_s=hits[0]/10 if hits else None))
            for name in ['model.pt','validation-logits.npz','validation-predictions.csv']:hashes[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    e=pd.DataFrame(events);e.to_csv(ROOT/'transitions.csv',index=False)
    transitions=[]
    for (seed,arm,kind),g in e.groupby(['seed','arm','kind']):
        transitions.append(dict(seed=int(seed),arm=arm,kind=kind,events=len(g),missed=int((~g.detected).sum()),mean_delay_s=float(g.loc[g.detected,'delay_s'].mean())))
    average={a:float(np.mean([r['f1'] for r in pooled if r['arm']==a])) for a in ['baseline','candidate']}
    recall={a:np.mean([r['recall'] for r in pooled if r['arm']==a],0).tolist() for a in average}
    seed_delta=[next(r['f1'] for r in pooled if r['seed']==s and r['arm']=='candidate')-next(r['f1'] for r in pooled if r['seed']==s and r['arm']=='baseline') for s in [20260910,20260911,20260912]]
    route_delta={route:float(np.mean([r['macro_f1'] for r in comparison['rows'] if r['route']==route and r['arm']=='candidate'])-np.mean([r['macro_f1'] for r in comparison['rows'] if r['route']==route and r['arm']=='baseline'])) for route in sorted(f.route.unique())}
    gate=dict(all_seeds_improve=all(x>0 for x in seed_delta),three_routes_improve=sum(x>0 for x in route_delta.values())>=3,recall_loss_within_2pp=min(np.array(recall['candidate'])-recall['baseline'])>=-.02)
    gate={k:bool(v) for k,v in gate.items()}
    assert not all(gate.values()),'Passing candidate requires separate adoption review'
    result=dict(status='COMPLETE_VALIDATED',decision='KEEP_BASELINE',averages=average,recall=recall,seed_f1_deltas=seed_delta,route_f1_deltas=route_delta,gate=gate,transitions=transitions,
        transition_definition='Same prior diagnostic: moving,10constant samples each side; straight/turn transition; first3consecutive target predictions within0..1s, missed separate, delay conditional on detected.',
        validation='All6 checkpoint hashes; CSV/logit row mapping; pooled/per-route metrics recomputed; all3 baseline logits exact; frozen script/plan hashes PASS',
        hashes=hashes,transitions_sha256=sha(ROOT/'transitions.csv'),scope='Reused4development routes and sensor labels, not official or human evaluation. No deployment.')
    (ROOT/'final-review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (ROOT/'status.json').write_text(json.dumps(dict(status='COMPLETE_VALIDATED',completed=6,total=6,decision='KEEP_BASELINE'),indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','averages','recall','gate','transitions']},indent=2))

if __name__=='__main__':main()
