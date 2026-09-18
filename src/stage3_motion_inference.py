"""Standalone Stage3 motion-only deployment of the fixed 20260910 control."""
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

_S3M_FORMAT='stage3-motion-control-v1'
_S3M_ACCEL=['ACCELERATING','DECELERATING','CONSTANT','STOPPED']
_S3M_STEER=['LEFT','STRAIGHT','RIGHT']
_S3M_COLUMNS=['ID','sample_index','accel_label','steer_label']

class S3MotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1682,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
        self.accel=nn.Linear(64,4);self.steer=nn.Linear(64,3)
    def forward(self,x):
        z=self.net(x);return self.accel(z),self.steer(z)

def s3_motion_features(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError(f'Cannot decode {path}')
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    previous=None;history=deque(maxlen=15);rows=[]
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            h,w=rgb.shape[:2];scale=256/min(h,w)
            rgb=cv2.resize(rgb,(max(256,round(w*scale)),max(256,round(h*scale))),interpolation=cv2.INTER_AREA)
            y,x=(rgb.shape[0]-224)//2,(rgb.shape[1]-224)//2
            rgb=cv2.resize(rgb[y:y+224,x:x+224],(96,96),interpolation=cv2.INTER_AREA)
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            if previous is not None:
                flow=np.zeros((96,96,2),np.float32) if np.array_equal(previous,gray) else cv2.calcOpticalFlowFarneback(previous,gray,None,.5,3,15,3,5,1.2,0)
                grid=cv2.resize(flow,(8,8),interpolation=cv2.INTER_AREA).reshape(-1)
                magnitude=np.linalg.norm(flow,axis=2)
                current=np.r_[grid,np.percentile(magnitude,[10,50,90,99]),flow.mean((0,1))].astype(np.float32)
                history.append(current)
            values=list(history) if history else [np.zeros(134,np.float32)]
            rows.append(np.r_[values[-1],np.mean(values[-5:],axis=0),np.mean(values,axis=0)])
            previous=gray
    finally:cap.release()
    if not rows or len(rows)!=total:raise ValueError(f'Incomplete video decode: {path}')
    result=np.asarray(rows,dtype=np.float32)
    if result.shape!=(total,402) or not np.isfinite(result).all():raise ValueError('Invalid motion features')
    return result

def s3_motion_load(path,device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('format')!=_S3M_FORMAT or checkpoint.get('accel_classes')!=_S3M_ACCEL or checkpoint.get('steer_classes')!=_S3M_STEER:
        raise ValueError('Motion checkpoint contract mismatch')
    mask=checkpoint['mask']
    if checkpoint['mode']!='motion' or mask.shape!=(1682,) or torch.any(mask[:1280]!=0) or torch.any(mask[1280:]!=1):
        raise ValueError('Expected motion-only checkpoint')
    mean=checkpoint['mean'].numpy();std=checkpoint['std'].numpy()
    if mean.shape!=(1682,) or std.shape!=(1682,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('Invalid normalization')
    model=S3MotionModel();model.load_state_dict(checkpoint['model'],strict=True)
    return model.to(device).eval(),mean,std

def s3_motion_logits(model,features,mean,std,device):
    outputs=[[],[]]
    tf32=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        with torch.inference_mode():
            for start in range(0,len(features),256):
                chunk=features[start:start+256]
                x=np.zeros((len(chunk),1682),np.float32)
                x[:,1280:]=np.clip((chunk-mean[1280:])/std[1280:],-10,10)
                logits=model(torch.from_numpy(x).to(device))
                for k,value in enumerate(logits):
                    if not torch.isfinite(value).all():raise ValueError('Nonfinite motion prediction')
                    outputs[k].append(value.cpu().numpy())
    finally:torch.backends.cuda.matmul.allow_tf32=tf32
    return tuple(np.concatenate(parts) for parts in outputs)

def s3_motion_predict_video(model,path,mean,std,device):
    features=s3_motion_features(path)
    a,s=s3_motion_logits(model,features,mean,std,device)
    return pd.DataFrame(dict(ID=path.stem,sample_index=np.arange(len(a)),accel_label=np.array(_S3M_ACCEL)[a.argmax(1)],steer_label=np.array(_S3M_STEER)[s.argmax(1)]),columns=_S3M_COLUMNS)

def predict_stage3(data_dir,model_dir):
    if not torch.cuda.is_available():raise RuntimeError('Competition inference requires CUDA')
    device=torch.device('cuda')
    path=Path(model_dir)/'best.pt'
    if not path.is_file():path=Path(model_dir)/'stage3'/'best.pt'
    model,mean,std=s3_motion_load(path,device)
    paths=sorted(p for p in (Path(data_dir)/'videos').iterdir() if p.suffix.lower() in {'.mp4','.avi','.mov','.mkv','.hevc','.m4v','.3gp','.3gpp','.wmv'})
    if len({p.stem for p in paths})!=len(paths):raise ValueError('Duplicate video IDs')
    parts=[s3_motion_predict_video(model,p,mean,std,device) for p in paths]
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=_S3M_COLUMNS)
