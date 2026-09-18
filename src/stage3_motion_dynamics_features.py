"""Derived dynamics from fixed 1/5/15 flow summaries; no image or sensor input."""
import numpy as np

def features(x,endpoints):
    if np.any(np.asarray(endpoints)<15):raise ValueError('This experiment requires a complete15-pair history; startup deployment is out of scope')
    z=x[:,1280:].reshape(-1,3,134);current,recent,full=z[:,0],z[:,1],z[:,2]
    previous=(15*full-5*recent)/10
    delta=np.concatenate([current-recent,recent-previous],axis=1)
    a=recent[:,:128].reshape(-1,64,2);b=previous[:,:128].reshape(-1,64,2);m=full[:,:128].reshape(-1,64,2)
    an=np.linalg.norm(a,axis=2);bn=np.linalg.norm(b,axis=2)
    denom=an*bn;cos=np.divide((a*b).sum(2),denom,out=np.zeros_like(denom),where=denom>1e-8).clip(-1,1)
    size_change=an-bn
    denom=(5*an+10*bn)/15
    consistency=np.divide(np.linalg.norm(m,axis=2),denom,out=np.zeros_like(denom),where=denom>1e-8).clip(0,1)
    extra=np.concatenate([delta,size_change,cos,consistency],axis=1).astype(np.float32)
    assert extra.shape==(len(x),460) and np.isfinite(extra).all()
    result=x.copy();result[:,:1280]=0;result[:,:460]=extra
    return result

def mask(arm):
    m=np.zeros(1682,np.float32);m[1280:]=1
    if arm=='dynamics':m[:460]=1
    elif arm!='baseline':raise ValueError(arm)
    return m
