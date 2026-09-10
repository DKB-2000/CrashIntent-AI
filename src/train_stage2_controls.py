"""Source-held-out Stage2 controls using cached ResNet features and four heads."""
import argparse, json, time, random, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import stage2_pipeline as p


def metrics(rows,pred):
    result={}
    for key,target in [('collision_frame','t_collision'),('entry_frame','t_entry')]:
        errors=[abs(a[key]-int(b[target]))/float(b['fps']) for a,b in zip(pred,rows)]
        result[key+'_within_0.3s']=float(np.mean(np.array(errors)<=.3+1e-9))
        result[key+'_mae_seconds']=float(np.mean(errors))
    for key in ['evasion_space','entry_side']:
        result[key+'_accuracy']=float(np.mean([a[key]==b[key] for a,b in zip(pred,rows)]))
    result['development_mean']=float(np.mean([result[k] for k in ['collision_frame_within_0.3s','entry_frame_within_0.3s','evasion_space_accuracy','entry_side_accuracy']]))
    return result


def predictions(model,rows,features,device):
    model.eval();out=[]
    with torch.inference_mode():
        for row in rows:
            c,e,s=model(features[row['ID']].unsqueeze(0).to(device))
            out.append(dict(ID=row['ID'],collision_frame=int(c),entry_frame=int(e),evasion_space=int(s[:,:2].argmax(1)),entry_side=['LEFT','RIGHT'][int(s[:,2:].argmax(1))]))
    return out


def temporal_loss(logits,target,sigma):
    if sigma==0:return F.cross_entropy(logits,target)
    distance=torch.arange(logits.shape[1],device=logits.device)[None,:]-target[:,None]
    soft=torch.exp(-.5*(distance/sigma)**2);soft=soft/soft.sum(1,keepdim=True)
    return -(soft*F.log_softmax(logits,1)).sum(1).mean()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir',type=Path,required=True)
    parser.add_argument('--video-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--max-seconds',type=int,default=3000)
    args=parser.parse_args();started=time.monotonic()
    if not 0<args.max_seconds<=3300:raise ValueError('Experiment limit must be <=3300 seconds')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);p.cv2.setNumThreads(1);device=p._device(args.device)
    rows={s:pd.read_csv(args.split_dir/f'labels_{s}.csv',dtype={'ID':str,'source_id':str}).to_dict('records') for s in ['train','validation']}
    if {r['source_id'] for r in rows['train']}&{r['source_id'] for r in rows['validation']}:raise ValueError('Source leakage')
    backbone,transform=p._backbone('none',device)
    weights=args.split_dir/'training/model/stage2/resnet18-f37072fd.pth'
    backbone.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));backbone.fc=nn.Identity();backbone.to(device).eval()
    features={}
    for row in rows['train']+rows['validation']:
        path=args.video_dir/(row['ID']+'.mp4')
        if time.monotonic()-started>args.max_seconds:raise TimeoutError('Feature extraction budget exhausted')
        features[row['ID']]=p._features(p._read_video(path),backbone,transform,device,16)
        if len(features[row['ID']])!=int(row['frames']):raise ValueError('Feature/frame mismatch')
        print('features',row['ID'],flush=True)
    torch.save(features,out/'features.pt');del backbone
    modes=[('baseline_5epoch',5,0.,False),('longer_exact',25,0.,False),('soft_mixed_scene',25,1.,True)]
    report=dict(status='RUNNING',scope='Same 51/15-source-held-out development split. Not official Stage2 score.',split_sha256={s:hashlib.sha256((args.split_dir/f'labels_{s}.csv').read_bytes()).hexdigest() for s in rows},candidates=[])
    best_score=-1
    for name,epochs,sigma,mixed in modes:
        random.seed(p.SEED);np.random.seed(p.SEED);torch.manual_seed(p.SEED)
        model=p.Stage2Temporal().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
        history=[];best=-1;best_state=None;best_pred=None
        for epoch in range(1,epochs+1):
            if time.monotonic()-started>args.max_seconds:break
            model.train();order=list(rows['train']);random.shuffle(order);losses=[]
            for row in order:
                if time.monotonic()-started>args.max_seconds:break
                seq=features[row['ID']].unsqueeze(0).to(device)
                ct=torch.tensor([row['t_collision']],device=device);et=torch.tensor([row['t_entry']],device=device)
                cl,el,h=model.logits(seq)
                # Gradually expose scene heads to inferred positions; targets remain manual.
                use_pred=mixed and random.random()<min(.5,epoch/20)
                ci=cl.argmax(1) if use_pred else ct;ei=el.argmax(1) if use_pred else et
                scene=model.scene(torch.cat([h[0,ci],h[0,ei]],dim=1))
                loss=temporal_loss(cl,ct,sigma)+temporal_loss(el,et,sigma)
                loss=loss+F.cross_entropy(scene[:,:2],torch.tensor([row['evasion_space']],device=device))
                loss=loss+F.cross_entropy(scene[:,2:],torch.tensor([int(row['entry_side']=='RIGHT')],device=device))
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss))
            pred=predictions(model,rows['validation'],features,device);score=metrics(rows['validation'],pred)
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),**score));print(name,history[-1],flush=True)
            # Baseline control is its final epoch, matching previous report.
            if (name=='baseline_5epoch' and epoch==epochs) or (name!='baseline_5epoch' and score['development_mean']>best):
                best=score['development_mean'];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_pred=pred
        if best_state is None:break
        folder=out/name;folder.mkdir();torch.save({'model':best_state,'labels':p.OUTPUT_COLUMNS},folder/'best.pt')
        reloaded=p.Stage2Temporal().to(device);reloaded.load_state_dict(torch.load(folder/'best.pt',map_location=device,weights_only=True)['model'],strict=True)
        check=predictions(reloaded,rows['validation'],features,device)
        if check!=best_pred:raise ValueError('Reload predictions changed')
        pd.DataFrame(check)[p.OUTPUT_COLUMNS].to_csv(folder/'validation_predictions.csv',index=False)
        summary=dict(name=name,history=history,selected_metrics=metrics(rows['validation'],check),reload_matches=True)
        report['candidates'].append(summary)
        if best>best_score:best_score=best;report['selected']=name
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    report['status']='COMPLETED' if len(report['candidates'])==len(modes) else 'TIME_LIMIT';report['seconds']=time.monotonic()-started
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
if __name__=='__main__':main()
