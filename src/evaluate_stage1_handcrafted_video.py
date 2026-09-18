"""Fit the fixed Imperial handcrafted model and evaluate 420 labeled videos."""
import json,math,os,tempfile,time,zipfile
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts/stage1-handcrafted-video-20260918'; BUNDLE=ROOT/'artifacts/kaggle-stage1-quality-trial-20260911/dataset/quality.bin'; IMPERIAL=ROOT/'data_raw/imperial-recapture/SubjectiveTestImages.clean.zip'; BASE=ROOT/'artifacts/kaggle-stage1-quarter-mix-trial-20260916/validated/baseline-predictions.jsonl'
def write(path,value):
 temporary=path.with_suffix('.tmp'); temporary.write_text(json.dumps(value,indent=2)); os.replace(temporary,path)
def features(rgb):
 small=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA).astype(np.float32)/255; gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY); spectrum=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray)))); yy,xx=np.indices(spectrum.shape); radius=np.sqrt((yy-128)**2+(xx-128)**2); result=[]; boundaries=np.linspace(0,181,25)
 for low,high in zip(boundaries[:-1],boundaries[1:]): result.append(float(spectrum[(radius>=low)&(radius<high)].mean()))
 for channel in range(3):
  values=small[:,:,channel]; result += [float(values.mean()),float(values.std())]+[float(x) for x in np.quantile(values,[.05,.25,.5,.75,.95])]
 for channel in range(3):
  values=small[:,:,channel]; gx=cv2.Sobel(values,cv2.CV_32F,1,0); gy=cv2.Sobel(values,cv2.CV_32F,0,1); magnitude=np.sqrt(gx*gx+gy*gy); result += [float(magnitude.mean()),float(magnitude.std()),float(np.quantile(magnitude,.9))]
 return np.asarray(result,dtype=np.float32)
def fit(x,y):
 mean=x.mean(0); std=x.std(0); std[std<1e-6]=1; normalized=(x-mean)/std; model=nn.Linear(x.shape[1],1); optimizer=torch.optim.LBFGS(model.parameters(),lr=.5,max_iter=100,line_search_fn='strong_wolfe'); tx=torch.from_numpy(normalized).float(); ty=torch.from_numpy(y).float()[:,None]
 def closure(): optimizer.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(model(tx),ty)+.05*sum(v.square().sum() for v in model.parameters()); loss.backward(); return loss
 optimizer.step(closure); return mean,std,model.weight.detach().numpy().ravel(),float(model.bias.detach())
def probability(x,mean,std,weight,bias): return float(1/(1+math.exp(-float(((x-mean)/std)@weight+bias))))
def metric(rows,key):
 answer=np.asarray([r[key]>=.5 for r in rows]); target=np.asarray([r['label'] for r in rows]); scores=[]
 for label in (0,1):
  tp=np.sum((target==label)&(answer==label)); fp=np.sum((target!=label)&(answer==label)); fn=np.sum((target==label)&(answer!=label)); p=tp/(tp+fp) if tp+fp else 0; q=tp/(tp+fn) if tp+fn else 0; scores.append(2*p*q/(p+q) if p+q else 0)
 return {'videos':len(rows),'accuracy':float(np.mean(answer==target)),'macro_f1':float(np.mean(scores)),'errors':int(np.sum(answer!=target))}
def main():
 OUT.mkdir(exist_ok=False); status=OUT/'status.json'; write(status,{'status':'RUNNING','completed':0,'total':420,'started_unix':time.time()}); torch.manual_seed(236753); cv2.setNumThreads(1)
 with zipfile.ZipFile(IMPERIAL) as archive:
  labels={line.split('\t')[0]:int(line.split('\t')[1]!='Original Captured') for line in archive.read('labels.txt').decode().splitlines()}; x=[]; y=[]
  for name in sorted(labels): x.append(features(cv2.cvtColor(cv2.imdecode(np.frombuffer(archive.read(name),np.uint8),1),cv2.COLOR_BGR2RGB))); y.append(labels[name])
 mean,std,weight,bias=fit(np.stack(x),np.asarray(y)); write(OUT/'classifier.json',{'mean':mean.tolist(),'std':std.tolist(),'weight':weight.tolist(),'bias':bias,'feature_count':54,'training_images':100,'regularization':.05})
 base={row['path']:row for row in (json.loads(line) for line in BASE.read_text().splitlines())}; results=[]; started=time.time()
 with zipfile.ZipFile(BUNDLE) as archive:
  plan=json.loads(archive.read('plan.json')); rows=[row for row in plan['rows'] if row['role']=='evaluation']
  if len(rows)!=420 or set(base)!=set(row['path'] for row in rows): raise ValueError('Evaluation identity mismatch')
  with tempfile.TemporaryDirectory(dir=OUT) as temp:
   temp=Path(temp)
   for index,row in enumerate(rows,1):
    path=temp/'video.mp4'; path.write_bytes(archive.read(row['path'])); capture=cv2.VideoCapture(str(path)); count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); frame_features=[]
    for fraction in (.2,.5,.8):
     capture.set(cv2.CAP_PROP_POS_FRAMES,max(0,min(count-1,round((count-1)*fraction)))); ok,bgr=capture.read()
     if not ok: raise ValueError('Decode failed: '+row['path'])
     frame_features.append(features(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
    capture.release(); path.unlink(); hand=float(np.mean([probability(value,mean,std,weight,bias) for value in frame_features])); mvit=float(base[row['path']]['probability'])
    result={'path':row['path'],'suite':row['suite'],'condition':row['condition'],'label':int(row['label']=='RERECORDED'),'handcrafted_probability':hand,'mvit_probability':mvit}
    for alpha in (.1,.25,.5): result[f'fusion_{alpha:g}']=(1-alpha)*mvit+alpha*hand
    results.append(result)
    if index%10==0: write(status,{'status':'RUNNING','completed':index,'total':420,'updated_unix':time.time()})
 summaries={}
 for key in ('mvit_probability','handcrafted_probability','fusion_0.1','fusion_0.25','fusion_0.5'):
  summaries[key]={'all':metric(results,key),'stress':metric([r for r in results if r['suite']=='stress'],key),'normal':metric([r for r in results if r['suite']=='normal'],key)}
 report={'status':'COMPLETE','training_images':100,'evaluation_videos':420,'frames_per_video':3,'summaries':summaries,'seconds':time.time()-started,'limitations':['Imperial license is not explicit; diagnostic only.','Synthetic CCD development videos are not independent competition data.','Fusion weights are fixed diagnostic comparisons.']}; write(OUT/'report.json',report); write(OUT/'predictions.json',results); write(status,{'status':'COMPLETE_VALIDATED','completed':420,'total':420,'updated_unix':time.time()})
if __name__=='__main__':
 try: main()
 except Exception as error:
  OUT.mkdir(exist_ok=True); write(OUT/'status.json',{'status':'FAILED','error':repr(error),'updated_unix':time.time()}); raise
