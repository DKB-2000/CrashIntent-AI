"""ResNet18 spatial-cue pilot on the fixed Imperial split."""
import json,random,sys,time,zipfile
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torchvision.models import resnet18

def crop(image,seed,train):
 rng=np.random.default_rng(seed); h,w=image.shape[:2]; scale=(256/min(h,w))*float(rng.uniform(.92,1.08) if train else 1); value=cv2.resize(image,(max(224,round(w*scale)),max(224,round(h*scale))),interpolation=cv2.INTER_AREA)
 h,w=value.shape[:2]; y=int(rng.integers(0,h-224+1)) if train and h>224 else (h-224)//2; x=int(rng.integers(0,w-224+1)) if train and w>224 else (w-224)//2; value=value[y:y+224,x:x+224].astype(np.float32)/255
 if train: value=np.clip(value*float(rng.uniform(.9,1.1))+float(rng.uniform(-.03,.03)),0,1)
 return torch.from_numpy(value).permute(2,0,1).float().sub_(torch.tensor([.485,.456,.406])[:,None,None]).div_(torch.tensor([.229,.224,.225])[:,None,None])
def f1(rows):
 values=[]
 for label in (0,1):
  tp=sum(r['label']==label and r['answer']==label for r in rows); fp=sum(r['label']!=label and r['answer']==label for r in rows); fn=sum(r['label']==label and r['answer']!=label for r in rows); p=tp/(tp+fp) if tp+fp else 0; q=tp/(tp+fn) if tp+fn else 0; values.append(2*p*q/(p+q) if p+q else 0)
 return sum(values)/2
def main(root):
 root=Path(root); out=root/'spatial-result'; out.mkdir(); random.seed(236753); np.random.seed(236753); torch.manual_seed(236753)
 with zipfile.ZipFile(root/'imperial.bin') as z:
  labels={x.split('\t')[0]:int(x.split('\t')[1]!='Original Captured') for x in z.read('labels.txt').decode().splitlines()}; images={n:cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(n),np.uint8),1),cv2.COLOR_BGR2RGB) for n in labels}
 split=json.loads((root/'split.json').read_text()); train=[(n,labels[n],images[n]) for n in split['train']]; hold=[(n,labels[n],images[n]) for n in split['holdout']]
 model=resnet18(weights=None); model.load_state_dict(torch.load(root/'resnet18.pth',map_location='cpu',weights_only=True)); model.fc=nn.Linear(model.fc.in_features,2)
 for parameter in model.parameters(): parameter.requires_grad=False
 for parameter in model.layer4.parameters(): parameter.requires_grad=True
 for parameter in model.fc.parameters(): parameter.requires_grad=True
 device=torch.device('cuda'); model.to(device); optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4,weight_decay=1e-4); scaler=torch.amp.GradScaler('cuda'); losses=[]
 by={label:[x for x in train if x[1]==label] for label in (0,1)}; started=time.time(); model.train()
 for step in range(240):
  batch=[]; target=[]
  for label in (0,1):
   item=by[label][step%40]; batch.append(crop(item[2],step*101+label,True)); target.append(label)
  optimizer.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16): loss=nn.functional.cross_entropy(model(torch.stack(batch).to(device)),torch.tensor(target,device=device))
  scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss))
 model.eval(); rows=[]
 with torch.inference_mode():
  for name,label,image in hold:
   probs=[]
   for view in range(5):
    with torch.autocast('cuda',dtype=torch.float16): probs.append(float(model(crop(image,90000+int(name[3:6])*13+view,view>0).unsqueeze(0).to(device)).softmax(1)[0,1]))
   probability=float(np.mean(probs)); rows.append(dict(image=name,label=label,probability=probability,answer=int(probability>=.5),correct=int(probability>=.5)==label))
 pos=[r['probability'] for r in rows if r['label']==1]; neg=[r['probability'] for r in rows if r['label']==0]; auc=(sum(a>b for a in pos for b in neg)+.5*sum(a==b for a in pos for b in neg))/100
 report={'status':'COMPLETE','train_images':80,'holdout_images':20,'updates':240,'macro_f1':f1(rows),'accuracy':sum(r['correct'] for r in rows)/20,'auc':auc,'original_accuracy':sum(r['correct'] for r in rows if r['label']==0)/10,'rerecorded_accuracy':sum(r['correct'] for r in rows if r['label']==1)/10,'loss_first':float(np.mean(losses[:20])),'loss_last':float(np.mean(losses[-20:])),'seconds':time.time()-started,'limitations':['Diagnostic only; no submission adoption without video and regression evaluation.']}
 torch.save({'model':model.state_dict(),'architecture':'resnet18','size':224},out/'spatial.pt'); (out/'report.json').write_text(json.dumps(report,indent=2)); (out/'predictions.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(report))
if __name__=='__main__': main(Path(sys.argv[1]))
