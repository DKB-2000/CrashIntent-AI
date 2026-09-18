import json,time
import numpy as np,pandas as pd,torch
import compare_stage3_representations as c
from torch import nn
from torch.nn import functional as F
from stage3_temporal_motion_experiment import ROOT
torch.set_num_threads(1)
f=pd.read_csv(c.ROOT/'train-samples.csv');v=c.array_for(f);mean=v.mean(0);std=v.std(0).clip(.01);x=torch.from_numpy(np.clip((v-mean)/std,-10,10));x[:,:1280]*=0
ta=torch.tensor(f.accel.to_numpy());ts=torch.tensor(f.steer.to_numpy());torch.manual_seed(20260910);model=c.Control();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
for epoch in range(1,13):
 for ids in c.balanced_batches(f.to_dict('records'),epoch):
  a,s=model(x[ids]);moving=ta[ids]!=3;loss=F.cross_entropy(a,ta[ids])+F.cross_entropy(s[moving],ts[ids][moving]);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
old=torch.load(c.ROOT/'20260910-motion/model.pt',weights_only=True);diff={k:float(abs(v-old['model'][k]).max()) for k,v in model.state_dict().items()};result=dict(threads=1,parameter_max_difference=max(diff.values()),exact=all(x==0 for x in diff.values()),differences=diff)
c.save(ROOT/'baseline-thread-repro.json',result);print(result)
