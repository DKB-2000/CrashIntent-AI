"""Predeclared spatial flow controls; reuse audited cache, never submission mutation."""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import compare_stage3_representations as base
from train_stage3_route_trial import balanced_batches

OUT=Path('artifacts/stage3-spatial-motion-20260910')
MODES=('baseline','common_residual','regional_augmented')

def transform(x,mode):
    y=x.copy()
    if mode=='baseline':
        mask=np.r_[np.zeros(1280),np.ones(402)].astype(np.float32)
        return y,mask
    y[:,:1280]=0
    blocks=x[:,1280:].reshape(-1,3,134)
    grid=blocks[:,:,:128].reshape(-1,3,8,8,2)
    common=np.median(grid,axis=(2,3))
    residual=grid-common[:,:,None,None,:]
    if mode=='common_residual':
        y[:,1280:]=np.concatenate([residual.reshape(-1,3,128),blocks[:,:,128:]],axis=2).reshape(-1,402)
        extra=common.reshape(-1,6)
    else:
        regions=(residual[:,:,:4],residual[:,:,4:],residual[:,:,:,:4],residual[:,:,:,4:],residual[:,:,2:6,2:6])
        summaries=[]
        for region in regions:
            mag=np.linalg.norm(region,axis=-1)
            summaries.append(np.concatenate([region.mean((2,3)),mag.mean((2,3))[:,:,None],mag.std((2,3))[:,:,None]],axis=2))
        extra=np.concatenate([common]+summaries,axis=2).reshape(-1,66)
    y[:,:extra.shape[1]]=extra
    mask=np.zeros(1682,np.float32);mask[:extra.shape[1]]=1;mask[1280:]=1
    assert np.isfinite(y).all()
    return y,mask

def loaded():
    frames={part:pd.read_csv(base.ROOT/(part+'-samples.csv')) for part in ('train','validation')}
    values={part:base.array_for(frame) for part,frame in frames.items()}
    return frames,values

def metrics(frame,a,s):
    def one(truth,pred,labels):
        cm=np.zeros((len(labels),len(labels)),dtype=np.int64)
        np.add.at(cm,(truth,pred),1)
        tp=cm.diagonal();support=cm.sum(1);predicted=cm.sum(0)
        recall=np.divide(tp,support,out=np.zeros(len(labels),float),where=support!=0)
        precision=np.divide(tp,predicted,out=np.zeros(len(labels),float),where=predicted!=0)
        f1=np.divide(2*tp,support+predicted,out=np.zeros(len(labels),float),where=support+predicted!=0)
        return dict(samples=int(cm.sum()),macro_f1=float(f1.mean()),confusion_matrix=cm.tolist(),per_class={label:dict(support=int(support[i]),predicted=int(predicted[i]),recall=float(recall[i]),precision=float(precision[i]),f1=float(f1[i])) for i,label in enumerate(labels)})
    moving=frame.accel.to_numpy()!=3
    return dict(accel=one(frame.accel.to_numpy(),a,base.p.ACCEL),steer=one(frame.steer.to_numpy()[moving],s[moving],base.p.STEER))

def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'plan.json').exists(),'Plan already exists; use train/verify'
    source=json.loads((base.ROOT/'independent-validation.json').read_text());assert source['status']=='PASS'
    frames,values=loaded()
    files=[base.ROOT/'results.json',base.ROOT/'plan.json',base.ROOT/'independent-validation.json',Path(base.__file__),Path(__file__),Path('src/train_stage3_route_trial.py')]
    for part in frames:files.append(base.ROOT/(part+'-samples.csv'))
    for sid in sorted(set(frames['train'].ID)|set(frames['validation'].ID)):
        for ext in ('.npz','.json'):files.append(base.ROOT/'cache'/(sid+ext))
        info=json.loads((base.ROOT/'cache'/(sid+'.json')).read_text())
        assert info['features_sha256']==base.sha(base.ROOT/'cache'/(sid+'.npz'))
    for seed in base.SEEDS:files += [base.ROOT/(str(seed)+'-motion')/name for name in ('model.pt','report.json','validation-logits.npz')]
    plan=dict(status='PREDECLARED',modes=list(MODES),seeds=list(base.SEEDS),epochs=12,updates=1200,batch=10,lr=.001,weight_decay=0.,threads=2,
      split='Original fixed 1000 training clips / 18603 validation frames; route-disjoint, reused development validation',
      checkpoint_selection='Fixed final epoch, no validation selection',normalization='Training samples only, std>=0.01, clip +/-10',
      inputs='Same 1682-D MLP. Original motion remains in last 402 slots; extra derived features occupy previously masked image slots.',
      variants=dict(baseline='Original motion mask and original mean/std, identically retrained',common_residual='For current/mean5/mean15 8x8 flow grid, replace grid by grid minus spatial median; retain original six statistics, append median xy (6 dims).',regional_augmented='Retain original motion; append per-block median xy and residual mean xy, mean/std magnitude in top/bottom/left/right/central4x4 regions (66 dims).'),
      limitation='Coarse cached grid only; median of time-averaged grids differs from average per-frame median. Not a physical ego-motion or camera-motion separation.',
      input_hashes={str(f):base.sha(f) for f in files})
    base.save(OUT/'plan.json',plan)
    print('PREDECLARED',len(files),'hashed files',flush=True)

def train():
    plan=json.loads((OUT/'plan.json').read_text())
    assert plan['input_hashes'][str(Path(__file__))]==base.sha(__file__)
    frames,values=loaded();ta=torch.tensor(frames['train'].accel.to_numpy());ts=torch.tensor(frames['train'].steer.to_numpy())
    batches={e:list(balanced_batches(frames['train'].to_dict('records'),e)) for e in range(1,13)}
    results=[]
    for mode in MODES:
        transformed={part:transform(x,mode)[0] for part,x in values.items()}
        mask=torch.from_numpy(transform(values['train'][:1],mode)[1])
        mean=transformed['train'].mean(0);std=transformed['train'].std(0).clip(.01)
        normalized={part:torch.from_numpy(np.clip((x-mean)/std,-10,10)) for part,x in transformed.items()}
        for seed in base.SEEDS:
            out=OUT/(str(seed)+'-'+mode);out.mkdir(exist_ok=True)
            assert not (out/'report.json').exists(),'Existing completed run; refuse overwrite'
            start=time.monotonic();torch.manual_seed(seed);model=base.Control();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
            x=normalized['train']*mask;history=[]
            for epoch in range(1,13):
                losses=[];model.train()
                for ids in batches[epoch]:
                    a,s=model(x[ids]);moving=ta[ids]!=3
                    loss=F.cross_entropy(a,ta[ids])+F.cross_entropy(s[moving],ts[ids][moving])
                    opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
                history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses)))
            model.eval();all_metrics={};routes={}
            for part in ('train','validation'):
                with torch.inference_mode():
                    aa=[];ss=[]
                    for chunk in normalized[part].split(256):
                        a,s=model(chunk*mask);aa.append(a);ss.append(s)
                    a=torch.cat(aa).numpy();s=torch.cat(ss).numpy()
                np.savez_compressed(out/(part+'-logits.npz'),accel=a,steer=s)
                pred=frames[part][['ID','endpoint']].rename(columns={'endpoint':'sample_index'}).copy()
                pred['accel_label']=np.array(base.p.ACCEL)[a.argmax(1)];pred['steer_label']=np.array(base.p.STEER)[s.argmax(1)]
                pred.to_csv(out/(part+'-predictions.csv'),index=False)
                all_metrics[part]=metrics(frames[part],a.argmax(1),s.argmax(1))
                if part=='validation':
                    for route,indices in frames[part].groupby('route').groups.items():
                        indices=np.asarray(indices);routes[route]=metrics(frames[part].iloc[indices],a[indices].argmax(1),s[indices].argmax(1))
            ck=dict(model=model.state_dict(),mean=torch.from_numpy(mean),std=torch.from_numpy(std),mask=mask,mode=mode,seed=seed)
            torch.save(ck,out/'model.pt')
            report=dict(status='COMPLETE',mode=mode,seed=seed,metrics=all_metrics,routes=routes,history=history,checkpoint_sha256=base.sha(out/'model.pt'),seconds=time.monotonic()-start)
            base.save(out/'report.json',report);results.append(report)
            base.save(OUT/'status.json',dict(status='TRAINING',completed=len(results),expected=9))
            print(seed,mode,'accel',all_metrics['validation']['accel']['macro_f1'],'steer',all_metrics['validation']['steer']['macro_f1'],flush=True)
    base.save(OUT/'results.json',dict(status='COMPLETE',runs=results))

def verify():
    plan=json.loads((OUT/'plan.json').read_text())
    for path,expected in plan['input_hashes'].items():assert base.sha(path)==expected,path
    frames,values=loaded();report=json.loads((OUT/'results.json').read_text());checks=[]
    for r in report['runs']:
        mode=r['mode'];seed=r['seed'];out=OUT/(str(seed)+'-'+mode)
        assert base.sha(out/'model.pt')==r['checkpoint_sha256']
        ck=torch.load(out/'model.pt',map_location='cpu',weights_only=True);model=base.Control();model.load_state_dict(ck['model'],strict=True);model.eval()
        original=torch.load(base.ROOT/(str(seed)+'-motion')/'model.pt',map_location='cpu',weights_only=True) if mode=='baseline' else None
        if original:
            for key in ck['model']:assert torch.equal(ck['model'][key],original['model'][key]),key
            for key in ('mean','std','mask'):assert torch.equal(ck[key],original[key]),key
        errors=[]
        for part,frame in frames.items():
            x,_=transform(values[part],mode)
            norm=torch.from_numpy(np.clip((x-ck['mean'].numpy())/ck['std'].numpy(),-10,10))*ck['mask']
            with torch.inference_mode():
                preds=[model(chunk) for chunk in norm.split(256)]
            a=torch.cat([t[0] for t in preds]).numpy();s=torch.cat([t[1] for t in preds]).numpy()
            with np.load(out/(part+'-logits.npz')) as saved:
                for key,val in (('accel',a),('steer',s)):
                    assert np.array_equal(saved[key],val);errors.append(float(np.max(np.abs(saved[key]-val))))
            expected=base.score(frame,a.argmax(1),s.argmax(1))
            for task in ('accel','steer'):
                assert expected[task]['confusion_matrix']==r['metrics'][part][task]['confusion_matrix']
                assert abs(expected[task]['macro_f1']-r['metrics'][part][task]['macro_f1'])<1e-12
            csv=pd.read_csv(out/(part+'-predictions.csv'))
            assert csv.ID.tolist()==frame.ID.tolist() and csv.sample_index.tolist()==frame.endpoint.tolist()
            assert csv.accel_label.tolist()==np.array(base.p.ACCEL)[a.argmax(1)].tolist()
            assert csv.steer_label.tolist()==np.array(base.p.STEER)[s.argmax(1)].tolist()
            if original:
                with np.load(base.ROOT/(str(seed)+'-motion')/(part+'-logits.npz')) as old:
                    assert np.array_equal(a,old['accel']) and np.array_equal(s,old['steer'])
            if part=='validation':
                for route,indices in frame.groupby('route').groups.items():
                    indices=np.asarray(indices);m=base.score(frame.iloc[indices],a[indices].argmax(1),s[indices].argmax(1))
                    for task in ('accel','steer'):assert m[task]['confusion_matrix']==r['routes'][route][task]['confusion_matrix']
        checks.append(dict(mode=mode,seed=seed,checkpoint_sha256=r['checkpoint_sha256'],max_logits_error=max(errors),baseline_original_exact=bool(original)))
    rows=[];classrows=[];routerows=[]
    for mode in MODES:
        runs=[r for r in report['runs'] if r['mode']==mode]
        for task in ('accel','steer'):
            vals=[r['metrics']['validation'][task]['macro_f1'] for r in runs]
            rows.append(dict(mode=mode,task=task,mean=np.mean(vals),minimum=min(vals),maximum=max(vals)))
            for label in runs[0]['metrics']['validation'][task]['per_class']:
                classrows.append(dict(mode=mode,task=task,label=label,**{m:float(np.mean([r['metrics']['validation'][task]['per_class'][label][m] for r in runs])) for m in ('support','predicted','recall','precision','f1')}))
            for route in runs[0]['routes']:
                routerows.append(dict(mode=mode,task=task,route=route,macro_f1=float(np.mean([r['routes'][route][task]['macro_f1'] for r in runs]))))
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(classrows).to_csv(OUT/'class-summary.csv',index=False);pd.DataFrame(routerows).to_csv(OUT/'route-summary.csv',index=False)
    base.save(OUT/'independent-validation.json',dict(status='PASS',input_files_verified=len(plan['input_hashes']),checks=checks,results_sha256=base.sha(OUT/'results.json'),script_sha256=base.sha(__file__),verification='Reloaded all checkpoints, recomputed all 19603 logits and CSV labels exactly; independent base metric confusion matrices and F1 agree, all 3 baseline checkpoints/normalization/logits exactly match original.'))
    base.save(OUT/'status.json',dict(status='COMPLETE_VALIDATED',completed=9,expected=9));print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','train','verify']);args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    {'prepare':prepare,'train':train,'verify':verify}[args.command]()
