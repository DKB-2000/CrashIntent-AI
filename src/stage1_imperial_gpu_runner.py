"""Frozen CUDA diagnostic for the Imperial 100-image recapture sample."""
import hashlib,io,json,sys,time,zipfile
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torchvision.models.video import mvit_v2_s

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clip(data):
 bgr=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
 h,w=rgb.shape[:2];scale=224/min(h,w);nh,nw=max(224,round(h*scale)),max(224,round(w*scale))
 rgb=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA);y,x=(nh-224)//2,(nw-224)//2;f=rgb[y:y+224,x:x+224]
 v=torch.from_numpy(np.stack([f]*16)).permute(3,0,1,2).float()/255
 return (v-.45)/.225
def main(root):
 root=Path(root);out=root/'result';out.mkdir(exist_ok=False);started=time.time()
 expected=json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
 for name,value in expected.items():
  if sha(root/name)!=value:raise ValueError('Input hash mismatch: '+name)
 # Use a non-.zip extension in the Kaggle dataset because Kaggle expands ZIP
 # assets at mount time and removes the original archive.
 with zipfile.ZipFile(root/'imperial.bin') as z:
  if z.testzip() is not None:raise ValueError('ZIP CRC failed')
  labels={x.split('\t')[0]:x.split('\t')[1] for x in z.read('labels.txt').decode().splitlines()}
  images={name:z.read(name) for name in sorted(labels)}
 if len(images)!=100:raise ValueError('Expected 100 images')
 sources=[('official_best',root/'official.pt'),('screen_mix_25',root/'screen.pt')];results=[]
 device=torch.device('cuda');torch.set_num_threads(2);cv2.setNumThreads(1)
 for model_name,path in sources:
  ck=torch.load(path,map_location='cpu',weights_only=True);model=mvit_v2_s(weights=None);model.head[1]=nn.Linear(model.head[1].in_features,2)
  model.load_state_dict(ck['model'],strict=True);model.to(device).eval()
  with torch.inference_mode():
   for name,data in images.items():
    value=clip(data).unsqueeze(0).to(device)
    with torch.autocast('cuda',dtype=torch.float16): probability=float(model(value).softmax(1)[0,1].float())
    label='ORIGINAL' if labels[name]=='Original Captured' else 'RERECORDED';answer='RERECORDED' if probability>=.5 else 'ORIGINAL'
    results.append(dict(model=model_name,image=name,label=label,probability=probability,answer=answer,correct=answer==label))
  del model;torch.cuda.empty_cache()
 summaries=[]
 for model_name,_ in sources:
  for label in ('ORIGINAL','RERECORDED'):
   p=[r for r in results if r['model']==model_name and r['label']==label]
   summaries.append(dict(model=model_name,label=label,images=len(p),accuracy=sum(r['correct'] for r in p)/len(p),mean_rerecorded_probability=float(np.mean([r['probability'] for r in p])),median_rerecorded_probability=float(np.median([r['probability'] for r in p]))))
 (out/'predictions.json').write_text(json.dumps(results,indent=2))
 report=dict(status='COMPLETE',summaries=summaries,input_sha256=expected,precision='CUDA_FP16',threshold=.5,torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),seconds=time.time()-started,limitations=['Static images repeated for 16 frames; no temporal cues.','No fitting or threshold selection.','Diagnostic use only.'])
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
