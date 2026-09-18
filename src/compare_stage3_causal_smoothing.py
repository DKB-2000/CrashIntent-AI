"""Fixed causal smoothing on existing development routes; no new training."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import torch

P=Path(__file__).resolve().parents[1]
REF=P/'artifacts/stage3-representation-20260910'
OUT=P/'artifacts/stage3-causal-smoothing-20260914'
SEEDS=[20260910,20260911,20260912]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def apply(prob,arm):
    raw=prob.argmax(1)
    if arm=='baseline':return raw
    if arm=='ema_0p5':
        state=prob[0].copy();out=[]
        for row in prob:
            state=.5*row+.5*state;out.append(int(state.argmax()))
        return np.array(out)
    out=[]
    for i in range(len(raw)):
        history=raw[max(0,i-2):i+1];counts=np.bincount(history,minlength=3)
        tied=set(np.flatnonzero(counts==counts.max()))
        out.append(next(int(v) for v in history[::-1] if v in tied))
    return np.array(out)


def metric(f,pred):
    use=f.accel.to_numpy()!=3;true=f.steer.to_numpy()[use];pred=pred[use]
    cm=np.zeros((3,3),int);np.add.at(cm,(true,pred),1)
    den=cm.sum(0)+cm.sum(1);rec=cm.sum(1)
    return dict(f1=float(np.divide(2*cm.diagonal(),den,out=np.zeros(3),where=den!=0).mean()),
                recall=np.divide(cm.diagonal(),rec,out=np.zeros(3),where=rec!=0).tolist(),confusion=cm.tolist())


def main():
    OUT.mkdir(exist_ok=False)
    f=pd.read_csv(REF/'validation-samples.csv');assert len(f)==18603 and f.route.nunique()==4
    arms=['baseline','majority3','ema_0p5']
    inputs=[REF/'validation-samples.csv',Path(__file__)]
    for seed in SEEDS:
        inputs += [REF/f'{seed}-motion'/n for n in ['model.pt','validation-logits.npz','validation-predictions.csv']]
    archive=P/'artifacts/stage3-motion-submit-candidate-20260910/submit.zip'
    v=json.loads((archive.parent/'validation.json').read_text());assert sha(archive)==v['candidate_sha256']
    with zipfile.ZipFile(archive) as z:
        weights=z.read('model/stage3/best.pt');assert hashlib.sha256(weights).hexdigest()==v['deploy_checkpoint_sha256']
        deployed=torch.load(io.BytesIO(weights),weights_only=True,map_location='cpu')
    saved=torch.load(REF/'20260910-motion/model.pt',weights_only=True,map_location='cpu')
    for k,value in saved['model'].items():assert torch.equal(value,deployed['model'][k])
    for k in ['mean','std','mask']:assert torch.equal(saved[k],deployed[k])
    plan=dict(arms=arms,seeds=SEEDS,rows=len(f),routes=4,
         rules='Per-video reset. majority3 trailing3 labels, tie most recent; EMA probability alpha0.5. No future frames.',
         transitions='Moving throughout;10samples constant before and10constant after; straight<->turn only. First3consecutive target predictions within onset..+10samples; missed reported separately, lag in nominal0.1s.',
         gate='All3seed F1 improve,>=3/4 mean routes improve,no pooled direction recall loss>2pp; perseed onset/end miss count cannot increase and mean detected delay increase<=0.1s. Diagnostic candidate only.',
         scope='Already reused4development routes, sensor proxy. No Civic rescoring/selection on June11; not independent or official improvement.',
         inputs_sha256={str(p.relative_to(P)):sha(p) for p in inputs},deployed_zip_sha256=sha(archive))
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    pooled=[];routes=[];events=[];predrows=[]
    for seed in SEEDS:
        with np.load(REF/f'{seed}-motion/validation-logits.npz') as z:logits=z['steer']
        assert logits.shape==(len(f),3) and np.isfinite(logits).all()
        recorded=pd.read_csv(REF/f'{seed}-motion/validation-predictions.csv')
        assert np.array_equal(recorded.ID,f.ID) and np.array_equal(recorded.sample_index,f.endpoint)
        assert np.array_equal(recorded.steer_label,np.array(['LEFT','STRAIGHT','RIGHT'])[logits.argmax(1)])
        prob=np.exp(logits-logits.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
        for arm in arms:
            prediction=np.full(len(f),-1,int)
            for sid,g in f.groupby('ID',sort=False):
                idx=g.index.to_numpy();assert np.all(np.diff(g.endpoint)==1)
                q=prob[idx];p=apply(q,arm);prediction[idx]=p
                # Prefix invariance checks causality against suffix perturbation.
                cut=len(q)//2;np.testing.assert_array_equal(apply(q[:cut],arm),p[:cut])
                truth=g.steer.to_numpy();moving=g.accel.to_numpy()!=3
                for t in range(10,len(g)-12):
                    before,after=truth[t-1],truth[t]
                    if before==after or (before!=1 and after!=1):continue
                    if not (moving[t-10:t+10].all() and np.all(truth[t-10:t]==before) and np.all(truth[t:t+10]==after)):continue
                    hits=[k for k in range(11) if np.all(truth[t:t+k+3]==after) and np.all(p[t+k:t+k+3]==after)]
                    events.append(dict(seed=seed,arm=arm,ID=sid,route=g.route.iloc[0],onset=t,
                         kind='turn_start' if before==1 else 'turn_end',detected=bool(hits),delay_s=hits[0]/10 if hits else None))
            pooled.append(dict(seed=seed,arm=arm,**metric(f,prediction)))
            for route,g in f.groupby('route'):routes.append(dict(seed=seed,arm=arm,route=route,**metric(g,prediction[g.index])))
            table=f[['ID','endpoint','accel','steer']].copy();table['seed']=seed;table['arm']=arm;table['prediction']=prediction;predrows.append(table)
    pred=pd.concat(predrows,ignore_index=True);pred.to_csv(OUT/'predictions.csv',index=False)
    e=pd.DataFrame(events);e.to_csv(OUT/'transitions.csv',index=False)
    transition=[]
    for (seed,arm,kind),g in e.groupby(['seed','arm','kind']):
        hit=g[g.detected]
        transition.append(dict(seed=int(seed),arm=arm,kind=kind,events=len(g),missed=int((~g.detected).sum()),
                              mean_delay_s=float(hit.delay_s.mean()) if len(hit) else None))
    decisions={}
    for arm in arms[1:]:
        deltas=[next(r['f1'] for r in pooled if r['seed']==s and r['arm']==arm)-next(r['f1'] for r in pooled if r['seed']==s and r['arm']=='baseline') for s in SEEDS]
        rd=[np.mean([r['f1'] for r in routes if r['route']==route and r['arm']==arm])-np.mean([r['f1'] for r in routes if r['route']==route and r['arm']=='baseline']) for route in f.route.unique()]
        recalls=np.mean([r['recall'] for r in pooled if r['arm']==arm],0)-np.mean([r['recall'] for r in pooled if r['arm']=='baseline'],0)
        latency=True
        for r in [x for x in transition if x['arm']==arm]:
            b=next(x for x in transition if x['arm']=='baseline' and x['seed']==r['seed'] and x['kind']==r['kind'])
            latency &= r['missed']<=b['missed'] and r['mean_delay_s'] is not None and b['mean_delay_s'] is not None and r['mean_delay_s']<=b['mean_delay_s']+.1
        decisions[arm]=dict(seed_f1_deltas=deltas,route_f1_deltas=rd,recall_delta=recalls.tolist(),latency_pass=bool(latency),
             decision='FOLLOWUP_ONLY' if all(d>0 for d in deltas) and sum(d>0 for d in rd)>=3 and min(recalls)>=-.02 and latency else 'KEEP_BASELINE')
    # Saved CSV independent confusion/F1 reconstruction.
    import csv
    cms={(s,a):[[0]*3 for _ in range(3)] for s in SEEDS for a in arms}
    with (OUT/'predictions.csv').open(newline='') as stream:
        for r in csv.DictReader(stream):
            if int(r['accel'])!=3:cms[int(r['seed']),r['arm']][int(r['steer'])][int(r['prediction'])]+=1
    for r in pooled:
        cm=cms[r['seed'],r['arm']];assert cm==r['confusion']
        independent=sum(2*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm)) for i in range(3))/3
        assert abs(independent-r['f1'])<1e-12
    for p,h in plan['inputs_sha256'].items():assert sha(P/p)==h
    summary=dict(status='COMPLETE_VALIDATED',averages={a:float(np.mean([r['f1'] for r in pooled if r['arm']==a])) for a in arms},
          decisions=decisions,pooled=pooled,routes=routes,transitions=transition,
          validation='Frozen deployed seed parity; saved prediction/logit mapping; per-video reset/prefix causality; CSV confusion/F1 independent recomputation; input hashes PASS',
          scope=plan['scope'],output_sha256={p.name:sha(p) for p in OUT.glob('*.csv')})
    (OUT/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['status','averages','decisions','transitions']},indent=2))


if __name__=='__main__':main()
