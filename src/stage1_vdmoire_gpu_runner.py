"""Evaluate frozen Stage1 models on recovered real VDMoire clips."""
import json,sys,time,zipfile
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torchvision.models.video import mvit_v2_s
def tensor(frames):
 values=[]
 for bgr in frames:
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]; scale=224/min(h,w); rgb=cv2.resize(rgb,(max(224,round(w*scale)),max(224,round(h*scale))),interpolation=cv2.INTER_AREA); h,w=rgb.shape[:2]; values.append(rgb[(h-224)//2:(h-224)//2+224,(w-224)//2:(w-224)//2+224])
 value=torch.from_numpy(np.stack(values)).permute(3,0,1,2).float()/255; return (value-.45)/.225
def model(path,device):
 checkpoint=torch.load(path,map_location='cpu',weights_only=True); value=mvit_v2_s(weights=None); value.head[1]=nn.Linear(value.head[1].in_features,2); value.load_state_dict(checkpoint['model']); return value.to(device).eval()
def main(root):
 root=Path(root); out=root/'result'; out.mkdir(); clips=root/'clips'; clips.mkdir()
 with zipfile.ZipFile(root/'clips.bin') as archive: archive.extractall(clips)
 device=torch.device('cuda'); rows=[]; started=time.time()
 for model_name,path in [('official_best',root/'official.pt'),('screen_mix_25',root/'screen.pt')]:
  net=model(path,device)
  with torch.inference_mode():
   for video in sorted(clips.glob('*.mp4')):
    capture=cv2.VideoCapture(str(video)); frames=[]
    while True:
     ok,frame=capture.read()
     if not ok: break
     frames.append(frame)
    capture.release(); probabilities=[]
    for slot in range(3):
     lo=round(slot*len(frames)/3); hi=max(lo+1,round((slot+1)*len(frames)/3)); indices=np.linspace(lo,hi-1,16).round().astype(int); value=tensor([frames[i] for i in indices]).unsqueeze(0).to(device)
     with torch.autocast('cuda',dtype=torch.float16): probabilities.append(float(net(value).softmax(1)[0,1]))
    probability=float(np.mean(probabilities)); rows.append({'model':model_name,'video':video.name,'probability':probability,'answer':'RERECORDED' if probability>=.5 else 'ORIGINAL','correct':probability>=.5,'slot_probabilities':probabilities})
  del net; torch.cuda.empty_cache()
 summaries=[]
 for name in ('official_best','screen_mix_25'):
  selected=[r for r in rows if r['model']==name]; summaries.append({'model':name,'clips':len(selected),'rerecorded_recall':sum(r['correct'] for r in selected)/len(selected),'mean_probability':float(np.mean([r['probability'] for r in selected]))})
 report={'status':'COMPLETE','summaries':summaries,'seconds':time.time()-started,'gpu':torch.cuda.get_device_name(),'limitations':['Eight RERECORDED clips only; no ORIGINAL specificity measurement.','Diagnostic use pending dataset license clarification.']}; (out/'report.json').write_text(json.dumps(report,indent=2)); (out/'predictions.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(report))
if __name__=='__main__': main(Path(sys.argv[1]))
