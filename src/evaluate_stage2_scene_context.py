"""Stage2 step 2: paired point/window scene-head adaptation on fixed source folds."""
import copy
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import stage2_pipeline as p
from analyze_stage2_source_cv import details, summary

ROOT=Path(__file__).resolve().parents[1]
CV=ROOT/'artifacts/stage2-source-cv-20260910'
OUT=ROOT/'artifacts/stage2-scene-context-20260910'
SEEDS=[20260825,20260826,20260827]


def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def context(h,positions,fps,mode):
    vectors=[]
    for position in positions:
        if mode=='point':
            vectors.append(h[position])
        else:
            radius=int(np.floor(.5*fps+1e-9))
            lo,hi=max(0,position-radius),min(len(h),position+radius+1)
            pooled=h[lo:hi].mean(0)
            mask=torch.abs(torch.arange(len(h))-position)/fps<=.5+1e-9
            assert torch.allclose(pooled,h[mask].mean(0),atol=1e-6)
            vectors.append(pooled)
    return torch.cat(vectors)


def predict(head,rows,inputs,positions):
    head.eval()
    with torch.no_grad():
        outputs=head(torch.stack([inputs[r['ID']] for r in rows]))
    return [dict(ID=r['ID'],collision_frame=positions[r['ID']][0],entry_frame=positions[r['ID']][1],evasion_space=int(outputs[i,:2].argmax()),entry_side=['LEFT','RIGHT'][int(outputs[i,2:].argmax())]) for i,r in enumerate(rows)]


def main():
    OUT.mkdir(exist_ok=False)
    started=time.monotonic()
    torch.set_num_threads(2)
    protocol=json.loads((CV/'protocol.json').read_text())
    assert json.loads((CV/'independent-validation.json').read_text())['status']=='PASS'
    cache=ROOT/'artifacts/stage2-controls-20260909/features.pt'
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==protocol['features_sha256']
    features=torch.load(cache,map_location='cpu',weights_only=True)
    source_pred=pd.read_csv(CV/'soft_mixed_15_oof.csv',dtype={'ID':str,'source_id':str}).set_index('ID')
    write('protocol.json',{'seeds':SEEDS,'folds':protocol['folds'],'feature_sha256':protocol['features_sha256'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'epochs':15,'lr':2e-4,'batch_size':8,'window_half_seconds':.5,'model':'Frozen fold-specific soft_mixed_15 temporal encoder and argmax time heads; adapt scene head only from same existing weights','control':'Point head adapted with identical samples/order/optimizer/dropout seed','selection':'Fixed settings, no outer-fold epoch selection','scope':'Reused development data, not independent test or official score'})
    all_rows=[]
    collected={f'{mode}_{seed}':[] for seed in SEEDS for mode in ['point','window']}
    original=[]
    provenance={}
    for fold,held in enumerate(protocol['folds']):
        folder=CV/f'fold-{fold}'
        train=pd.read_csv(folder/'train.csv',dtype={'ID':str,'source_id':str}).to_dict('records')
        val=pd.read_csv(folder/'validation.csv',dtype={'ID':str,'source_id':str}).to_dict('records')
        assert not {r['source_id'] for r in train}&{r['source_id'] for r in val}
        assert {r['source_id'] for r in val}==set(held)
        all_rows.extend([dict(r,fold=fold) for r in val])
        checkpoint=folder/'soft_mixed_15.pt'
        provenance[str(fold)]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        temporal=p.Stage2Temporal()
        temporal.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
        temporal.eval()
        inputs={mode:{} for mode in ['point','window']}
        positions={}
        with torch.no_grad():
            for r in train+val:
                cl,el,h=temporal.logits(features[r['ID']].unsqueeze(0))
                pos=(int(cl.argmax()),int(el.argmax()))
                positions[r['ID']]=pos
                for mode in inputs:
                    inputs[mode][r['ID']]=context(h[0],pos,r['fps'],mode).detach()
        before=predict(temporal.scene,val,inputs['point'],positions)
        for q in before:
            assert all(q[k]==source_pred.loc[q['ID'],'pred_'+k] for k in p.OUTPUT_COLUMNS[1:])
        original.extend(before)
        output=OUT/f'fold-{fold}'
        output.mkdir()
        pd.DataFrame(train).to_csv(output/'train.csv',index=False)
        pd.DataFrame(val).to_csv(output/'validation.csv',index=False)
        torch.save({'inputs':inputs,'positions':positions},output/'context-inputs.pt')
        for seed in SEEDS:
            for mode in ['point','window']:
                head=copy.deepcopy(temporal.scene)
                torch.manual_seed(seed)
                rng=random.Random(seed)
                optimizer=torch.optim.AdamW(head.parameters(),lr=2e-4)
                for epoch in range(15):
                    head.train()
                    order=list(train);rng.shuffle(order)
                    for offset in range(0,len(order),8):
                        batch=order[offset:offset+8]
                        x=torch.stack([inputs[mode][r['ID']] for r in batch])
                        y=head(x)
                        loss=F.cross_entropy(y[:,:2],torch.tensor([r['evasion_space'] for r in batch]))+F.cross_entropy(y[:,2:],torch.tensor([int(r['entry_side']=='RIGHT') for r in batch]))
                        optimizer.zero_grad(set_to_none=True);loss.backward()
                        torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                        optimizer.step()
                q=predict(head,val,inputs[mode],positions)
                path=output/f'{mode}_{seed}.pt'
                torch.save(head.state_dict(),path)
                reloaded=copy.deepcopy(temporal.scene)
                reloaded.load_state_dict(torch.load(path,weights_only=True),strict=True)
                assert predict(reloaded,val,inputs[mode],positions)==q
                collected[f'{mode}_{seed}'].extend(q)
        write('status.json',{'status':'RUNNING','folds_completed':fold+1,'seconds':time.monotonic()-started})
        print('fold',fold+1,'seconds',round(time.monotonic()-started,1),flush=True)
    assert len(all_rows)==len(features) and {r['ID'] for r in all_rows}==set(features)
    result={'original':summary(all_rows,original)}
    for name,pred in collected.items():
        data=details(all_rows,pred)
        pd.DataFrame(data).to_csv(OUT/f'{name}_oof.csv',index=False)
        result[name]=summary(all_rows,pred)
        result[name]['fold_means']={str(f):float(np.mean([sum(d[k+'_ok'] for k in p.OUTPUT_COLUMNS[1:])/4 for d in data if d['fold']==f])) for f in range(5)}
        for a,b in zip(original,pred):
            assert a['collision_frame']==b['collision_frame'] and a['entry_frame']==b['entry_frame']
    write('report.json',{'status':'COMPLETED','seconds':time.monotonic()-started,'checkpoint_sha256':provenance,'results':result,'checks':'OOF coverage, source exclusion, pooling bounds/direct mask comparison, argmax baseline reproduction, strict head reload prediction equality, unchanged time predictions'})
    write('status.json',{'status':'COMPLETED','seconds':time.monotonic()-started})
    for name,m in result.items():
        print(name,round(m['evasion_space_accuracy'],4),round(m['entry_side_accuracy'],4),round(m['development_mean'],4),flush=True)


if __name__=='__main__':main()
