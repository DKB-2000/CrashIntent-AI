"""Leakage-controlled 5-fold CV using frozen ResNet and spectral statistics."""
import hashlib,json,sys,time,zipfile
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torchvision.models import resnet18

def image_tensor(rgb):
 h,w=rgb.shape[:2]; scale=256/min(h,w); value=cv2.resize(rgb,(max(224,round(w*scale)),max(224,round(h*scale))),interpolation=cv2.INTER_AREA); h,w=value.shape[:2]; value=value[(h-224)//2:(h-224)//2+224,(w-224)//2:(w-224)//2+224].astype(np.float32)/255
 return torch.from_numpy(value).permute(2,0,1).float().sub_(torch.tensor([.485,.456,.406])[:,None,None]).div_(torch.tensor([.229,.224,.225])[:,None,None])
def handcrafted(rgb):
 small=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA).astype(np.float32)/255; gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
 fft=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray)))); yy,xx=np.indices(fft.shape); rr=np.sqrt((yy-128)**2+(xx-128)**2); radial=[]
 for lo,hi in zip(np.linspace(0,181,25)[:-1],np.linspace(0,181,25)[1:]): radial.append(float(fft[(rr>=lo)&(rr<hi)].mean()))
 color=[]
 for channel in range(3):
  values=small[:,:,channel]; color += [float(values.mean()),float(values.std())]+[float(x) for x in np.quantile(values,[.05,.25,.5,.75,.95])]
 edges=[]
 for channel in range(3):
  values=small[:,:,channel]; gx=cv2.Sobel(values,cv2.CV_32F,1,0); gy=cv2.Sobel(values,cv2.CV_32F,0,1); mag=np.sqrt(gx*gx+gy*gy); edges += [float(mag.mean()),float(mag.std()),float(np.quantile(mag,.9))]
 return np.asarray(radial+color+edges,dtype=np.float32)
def macro_f1(y,p):
 scores=[]
 for label in (0,1):
  tp=np.sum((y==label)&(p==label)); fp=np.sum((y!=label)&(p==label)); fn=np.sum((y==label)&(p!=label)); precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0; scores.append(2*precision*recall/(precision+recall) if precision+recall else 0)
 return float(np.mean(scores))
def auc(y,s):
 pos=s[y==1]; neg=s[y==0]; return float((sum(a>b for a in pos for b in neg)+.5*sum(a==b for a in pos for b in neg))/(len(pos)*len(neg)))
def fit_predict(train_x,train_y,test_x):
 mean=train_x.mean(0); std=train_x.std(0); std[std<1e-6]=1; x=(train_x-mean)/std; test=(test_x-mean)/std
 model=nn.Linear(x.shape[1],1); optimizer=torch.optim.LBFGS(model.parameters(),lr=.5,max_iter=100,line_search_fn='strong_wolfe'); tx=torch.from_numpy(x).float(); ty=torch.from_numpy(train_y).float()[:,None]
 def closure(): optimizer.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(model(tx),ty)+.05*sum(v.square().sum() for v in model.parameters()); loss.backward(); return loss
 optimizer.step(closure)
 with torch.inference_mode(): return torch.sigmoid(model(torch.from_numpy(test).float())).numpy().ravel()
def main(root):
 root=Path(root); out=root/'feature-cv-result'; out.mkdir(); torch.manual_seed(236753); started=time.time()
 with zipfile.ZipFile(root/'imperial.bin') as archive:
  labels={x.split('\t')[0]:int(x.split('\t')[1]!='Original Captured') for x in archive.read('labels.txt').decode().splitlines()}; names=sorted(labels); images=[cv2.cvtColor(cv2.imdecode(np.frombuffer(archive.read(n),np.uint8),1),cv2.COLOR_BGR2RGB) for n in names]
 backbone=resnet18(weights=None); backbone.load_state_dict(torch.load(root/'resnet18.pth',map_location='cpu',weights_only=True)); backbone.fc=nn.Identity(); backbone.cuda().eval(); deep=[]
 with torch.inference_mode():
  for index in range(0,100,16): deep.append(backbone(torch.stack([image_tensor(v) for v in images[index:index+16]]).cuda()).cpu().numpy())
 deep=np.concatenate(deep); hand=np.stack([handcrafted(v) for v in images]); y=np.asarray([labels[n] for n in names]); fold=np.empty(100,dtype=int)
 for label in (0,1):
  ids=[i for i,n in enumerate(names) if y[i]==label]; ids.sort(key=lambda i:hashlib.sha256(('cv-v1:'+names[i]).encode()).hexdigest())
  for rank,index in enumerate(ids): fold[index]=rank%5
 modes={'resnet':deep,'handcrafted':hand,'combined':np.concatenate([deep,hand],1)}; rows=[]; summaries=[]
 for mode,features in modes.items():
  scores=np.zeros(100,dtype=np.float32)
  for value in range(5):
   test=fold==value; train=~test; scores[test]=fit_predict(features[train],y[train],features[test])
  answers=(scores>=.5).astype(int)
  summaries.append({'mode':mode,'images':100,'accuracy':float(np.mean(answers==y)),'macro_f1':macro_f1(y,answers),'auc':auc(y,scores),'original_accuracy':float(np.mean(answers[y==0]==0)),'rerecorded_accuracy':float(np.mean(answers[y==1]==1))})
  rows += [{'mode':mode,'image':n,'fold':int(fold[i]),'label':int(y[i]),'probability':float(scores[i]),'answer':int(answers[i]),'correct':bool(answers[i]==y[i])} for i,n in enumerate(names)]
 report={'status':'COMPLETE','folds':5,'images':100,'features':{'resnet':int(deep.shape[1]),'handcrafted':int(hand.shape[1]),'combined':int(deep.shape[1]+hand.shape[1])},'summaries':summaries,'seconds':time.time()-started,'limitations':['Imperial sample lacks an explicit SPDX-style license; diagnostic only.','No hyperparameter selection on held-out folds; fixed regularization.','Image identities may not represent deployment video diversity.']}
 (out/'report.json').write_text(json.dumps(report,indent=2)); (out/'predictions.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(report))
if __name__=='__main__': main(Path(sys.argv[1]))
