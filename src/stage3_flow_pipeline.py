"""Stage3 causal optical-flow + compact MLP control, using only video pixels."""
import argparse,json,time,hashlib
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
S3F_ACCEL=['ACCELERATING','DECELERATING','CONSTANT','STOPPED']
S3F_STEER=['LEFT','STRAIGHT','RIGHT']
S3F_COLUMNS=['ID','sample_index','accel_label','steer_label']
S3F_FORMAT='stage3-flow-grid-v1'

class Stage3Flow(nn.Module):
    def __init__(self,dim):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,64),nn.ReLU());self.accel=nn.Linear(64,4);self.steer=nn.Linear(64,3)
    def forward(self,x):
        h=self.net(x);return self.accel(h),self.steer(h)

def flow_features(path):
    cap=cv2.VideoCapture(str(path));total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened():raise ValueError('Cannot decode video')
    previous=None;history=deque(maxlen=15);result=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            # Exclude hood and preserve road geometry. Fixed crop, no labels or sensors.
            h,w=frame.shape[:2];gray=cv2.cvtColor(frame[int(.15*h):int(.85*h)],cv2.COLOR_BGR2GRAY)
            gray=cv2.resize(gray,(96,72),interpolation=cv2.INTER_AREA)
            flow=np.zeros((72,96,2),np.float32) if previous is None else cv2.calcOpticalFlowFarneback(previous,gray,None,.5,3,15,3,5,1.2,0)
            grid=cv2.resize(flow,(8,6),interpolation=cv2.INTER_AREA).reshape(-1)
            magnitude=np.linalg.norm(flow,axis=2)
            current=np.concatenate([grid,np.percentile(magnitude,[10,50,90,99]),flow.mean((0,1))]).astype(np.float32)
            history.append(current)
            image=cv2.resize(gray,(8,6),interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32)/255
            result.append(np.concatenate([current,np.mean(list(history)[-5:],axis=0),np.mean(history,axis=0),image]))
            previous=gray
    finally:cap.release()
    if len(result)!=total or not result:raise ValueError('Incomplete video decode')
    return np.stack(result).astype(np.float32)

def flow_load(path,device):
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    if ckpt.get('format')!=S3F_FORMAT or ckpt['accel_classes']!=S3F_ACCEL or ckpt['steer_classes']!=S3F_STEER:raise ValueError('Flow checkpoint contract mismatch')
    model=Stage3Flow(len(ckpt['mean']));model.load_state_dict(ckpt['model'],strict=True)
    return model.to(device).eval(),ckpt['mean'].to(device),ckpt['std'].to(device)

def flow_predict_array(model,features,mean,std,device):
    aa,ss=[],[]
    with torch.inference_mode():
        for chunk in torch.from_numpy(features).split(512):
            a,s=model(((chunk.to(device)-mean)/std).clamp(-10,10));aa.extend(a.argmax(1).cpu().tolist());ss.extend(s.argmax(1).cpu().tolist())
    return aa,ss

def predict_stage3(data_dir,model_dir):
    if not torch.cuda.is_available():raise RuntimeError('Competition inference requires CUDA')
    device=torch.device('cuda');model,mean,std=flow_load(Path(model_dir)/'best.pt',device);result=[]
    paths=sorted(p for p in (Path(data_dir)/'videos').iterdir() if p.suffix.lower() in {'.mp4','.avi','.mov','.mkv','.hevc'})
    if len({p.stem for p in paths})!=len(paths):raise ValueError('Duplicate IDs')
    for path in paths:
        a,s=flow_predict_array(model,flow_features(path),mean,std,device)
        result.extend(dict(ID=path.stem,sample_index=i,accel_label=S3F_ACCEL[ai],steer_label=S3F_STEER[si]) for i,(ai,si) in enumerate(zip(a,s)))
    return pd.DataFrame(result,columns=S3F_COLUMNS)

def train(args):
    import stage3_pipeline as original
    started=time.monotonic();torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(20260909)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False);records=original.load_records(args.dataset_dir)
    values={};targets={};row_ids={}
    for part,rows in records.items():
        chunks=[];a=[];s=[];ids=[]
        for row in rows:
            if time.monotonic()-started>args.max_seconds:raise TimeoutError('Feature budget exhausted')
            features=flow_features(row['path'])
            if len(features)!=len(row['accel']):raise ValueError('Label count mismatch')
            chunks.append(features);a.extend(row['accel']);s.extend(row['steer']);ids.extend((row['ID'],i) for i in range(len(features)))
            print('features',part,row['ID'],flush=True)
        values[part]=np.concatenate(chunks);targets[part]=(torch.tensor(np.array(a)),torch.tensor(np.array(s)));row_ids[part]=ids
    mean=torch.from_numpy(values['train'].mean(0));std=torch.from_numpy(values['train'].std(0)).clamp_min(.01)
    x=((torch.from_numpy(values['train'])-mean)/std).clamp(-10,10);ta,ts=targets['train']
    model=Stage3Flow(x.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    wa=(torch.bincount(ta,minlength=4).float().clamp_min(1).sum()/torch.bincount(ta,minlength=4).float().clamp_min(1)).sqrt();wa/=wa.mean()
    moving=ta!=3;ws=(torch.bincount(ts[moving],minlength=3).float().clamp_min(1).sum()/torch.bincount(ts[moving],minlength=3).float().clamp_min(1)).sqrt();ws/=ws.mean()
    report=dict(status='RUNNING',architecture=S3F_FORMAT,scope='Full route-held-out comma2k19 proxy labels; not official ground truth',history=[],samples={p:len(v) for p,v in values.items()});best=-1
    for epoch in range(1,args.epochs+1):
        if time.monotonic()-started>args.max_seconds:break
        model.train();order=torch.randperm(len(x));losses=[]
        for idx in order.split(512):
            if time.monotonic()-started>args.max_seconds:break
            a,s=model(x[idx]);valid=ta[idx]!=3
            loss=F.cross_entropy(a,ta[idx],weight=wa)+(F.cross_entropy(s[valid],ts[idx][valid],weight=ws) if valid.any() else s.sum()*0)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        model.eval();a,s=flow_predict_array(model,values['validation'],mean,std,torch.device('cpu'));va,vs=targets['validation'];mask=va.numpy()!=3
        metrics=dict(accel=original.classification_metrics(va.numpy(),a,S3F_ACCEL),moving_steer=original.classification_metrics(vs.numpy()[mask],np.array(s)[mask],S3F_STEER))
        row=dict(epoch=epoch,loss=float(np.mean(losses)),validation=metrics);report['history'].append(row);score=metrics['moving_steer']['macro_f1'];print('epoch',epoch,'steer_f1',score,flush=True)
        if score>best:
            best=score;report['best_epoch']=epoch
            torch.save(dict(format=S3F_FORMAT,model=model.state_dict(),mean=mean,std=std,accel_classes=S3F_ACCEL,steer_classes=S3F_STEER),out/'best.pt')
            pd.DataFrame([dict(ID=rid,sample_index=i,accel_label=S3F_ACCEL[ai],steer_label=S3F_STEER[si]) for (rid,i),ai,si in zip(row_ids['validation'],a,s)],columns=S3F_COLUMNS).to_csv(out/'validation_predictions.csv',index=False)
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    loaded,mn,sd=flow_load(out/'best.pt',torch.device('cpu'));a,s=flow_predict_array(loaded,values['validation'],mn,sd,torch.device('cpu'))
    frame=pd.read_csv(out/'validation_predictions.csv')
    if frame.accel_label.tolist()!=[S3F_ACCEL[i] for i in a] or frame.steer_label.tolist()!=[S3F_STEER[i] for i in s]:raise ValueError('Reload mismatch')
    report.update(status='COMPLETED' if len(report['history'])==args.epochs else 'TIME_LIMIT',best_moving_steer_f1=best,reload_matches=True,seconds=time.monotonic()-started)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--epochs',type=int,default=30);parser.add_argument('--max-seconds',type=int,default=3300)
    args=parser.parse_args()
    if not 0<args.max_seconds<=3300 or args.epochs<1:raise ValueError('Invalid experiment budget')
    train(args)
if __name__=='__main__':main()
