"""Build a test-submission ZIP with the incumbent Stage2 temporal model plus crop probes."""
from __future__ import annotations
import ast, hashlib, io, json, time, zipfile
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
import stage2_pipeline as p
from daily_submission import inspect_zip

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stage1-auto-release-20260910/candidate/submit.zip'
CV=ROOT/'artifacts/stage2-source-cv-20260910'
FEATURES=ROOT/'artifacts/stage2-controls-20260909/features.pt'
OUT=ROOT/'artifacts/stage2-semantic-crop-submit-candidate-20260916'
SEEDS=(20260825,20260826,20260827,20260916,20260917)
EPOCHS=80

def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def sha(path):return sha_bytes(path.read_bytes())
def frame_at(path,index):
 cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read();cap.release()
 if not ok:raise RuntimeError(f'Cannot decode {path} frame {index}')
 return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
def crop_tensor(frame,transform):
 h,w=frame.shape[:2];y=int(h*.25)
 parts=(frame[y:,:int(w*.48)],frame[y:,int(w*.52):],frame[y:,int(w*.25):int(w*.75)])
 return torch.stack([transform(Image.fromarray(x)) for x in parts])
def crop_feature(frame,backbone,transform):
 with torch.inference_mode():z=backbone(crop_tensor(frame,transform))
 return torch.cat((z[0]-z[1],z[2]))
def save(name,value):
 (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

STAGE2_REPLACEMENT='''class _Stage2Temporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.r=nn.GRU(512,192,2,batch_first=True,bidirectional=True,dropout=.15)
        self.tc=nn.Linear(384,1);self.te=nn.Linear(384,1)
        self.scene=nn.Sequential(nn.Linear(768,192),nn.ReLU(),nn.Dropout(.2),nn.Linear(192,4))
    def logits(self,x):
        h,_=self.r(x);return self.tc(h).squeeze(-1),self.te(h).squeeze(-1),h

def _stage2_crop_batch(path,transform):
    rgb=np.asarray(Image.open(path).convert('RGB'))
    h,w=rgb.shape[:2];y=int(h*.25)
    parts=(rgb[y:,:int(w*.48)],rgb[y:,int(w*.52):],rgb[y:,int(w*.25):int(w*.75)])
    return torch.stack([transform(Image.fromarray(x)) for x in parts])

def _stage2_crop_feature(backbone,path,transform,device):
    batch=_stage2_crop_batch(path,transform).to(device)
    with torch.autocast(device_type='cuda',dtype=torch.float16):z=backbone(batch).float()
    return torch.cat((z[0]-z[1],z[2]))

def _frame_number(path: Path):
    match=re.search(r'(\\d+)$',path.stem)
    return int(match.group(1)) if match else 0

def predict_stage2(data_dir,model_dir):
    device=_device();model_dir=Path(model_dir);transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    backbone=resnet18(weights=None);backbone.load_state_dict(torch.load(model_dir/'resnet18-f37072fd.pth',map_location='cpu',weights_only=True));backbone.fc=nn.Identity();backbone.to(device).eval()
    checkpoint=torch.load(model_dir/'best.pt',map_location='cpu',weights_only=True)
    if checkpoint.get('crop_format')!='stage2-semantic-crop-ensemble-v1' or checkpoint.get('crop_seeds')!=[20260825,20260826,20260827,20260916,20260917]:raise ValueError('Stage2 crop checkpoint contract mismatch')
    temporal=_Stage2Temporal();temporal.load_state_dict(checkpoint['model'],strict=True);temporal.to(device).eval()
    probes=[]
    for state in checkpoint['crop_probes']:
        probe=nn.Linear(2816,4);probe.load_state_dict(state,strict=True);probes.append(probe.to(device).eval())
    mean=checkpoint['crop_mean'].to(device);std=checkpoint['crop_std'].to(device)
    if mean.shape!=(2816,) or std.shape!=(2816,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std<=0):raise ValueError('Invalid Stage2 crop normalization')
    rows=[];image_root=Path(data_dir)/'images'
    with torch.inference_mode():
        for folder in sorted(p for p in image_root.iterdir() if p.is_dir()):
            paths=sorted((p for p in folder.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}),key=_frame_number)
            if not paths:continue
            loader=DataLoader(_Stage2Frames(paths,transform),batch_size=256,num_workers=6,pin_memory=True);features=[]
            for images in loader:
                with torch.autocast(device_type='cuda',dtype=torch.float16):features.append(backbone(images.to(device,non_blocking=True)).float().cpu())
            sequence=torch.cat(features)[None].to(device);cl,el,h=temporal.logits(sequence);ci,ei=int(cl.argmax(1)),int(el.argmax(1))
            point=torch.cat((h[0,ci],h[0,ei]));crop_c=_stage2_crop_feature(backbone,paths[ci],transform,device);crop_e=_stage2_crop_feature(backbone,paths[ei],transform,device)
            x=torch.clamp((torch.cat((point,crop_c,crop_e))-mean)/std,-10,10);scene=torch.stack([q(x) for q in probes]).mean(0)
            nums=[_frame_number(q) for q in paths]
            rows.append({'ID':folder.name,'collision_frame':nums[ci],'entry_frame':nums[ei],'evasion_space':int(scene[:2].argmax()),'entry_side':'RIGHT' if int(scene[2:].argmax()) else 'LEFT'})
    del backbone,temporal,probes;torch.cuda.empty_cache()
    return pd.DataFrame(rows,columns=['ID','collision_frame','entry_frame','evasion_space','entry_side'])

'''

def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir();started=time.monotonic();torch.set_num_threads(2)
 if sha(BASE)!='2c0090adb06f272c0833cfb151ed479ed9c78273830f724e0439fef220626197':raise RuntimeError('Base ZIP hash mismatch')
 rows=pd.concat([pd.read_csv(CV/'fold-0/train.csv',dtype={'ID':str,'source_id':str}),pd.read_csv(CV/'fold-0/validation.csv',dtype={'ID':str,'source_id':str})]).sort_values('ID').to_dict('records')
 if len(rows)!=66 or len({r['ID'] for r in rows})!=66:raise RuntimeError('Expected 66 labels')
 protocol=json.loads((CV/'protocol.json').read_text());features=torch.load(FEATURES,weights_only=True,map_location='cpu')
 if sha(FEATURES)!=protocol['features_sha256'] or set(features)!={r['ID'] for r in rows}:raise RuntimeError('Feature cache mismatch')
 with zipfile.ZipFile(BASE) as z:
  inference=z.read('inference.py').decode('utf-8-sig');temporal_bytes=z.read('model/stage2/best.pt');backbone_bytes=z.read('model/stage2/resnet18-f37072fd.pth')
 temporal=p.Stage2Temporal();temporal.load_state_dict(torch.load(io.BytesIO(temporal_bytes),weights_only=True,map_location='cpu')['model'],strict=True);temporal.eval()
 backbone=resnet18(weights=None);backbone.load_state_dict(torch.load(io.BytesIO(backbone_bytes),weights_only=True,map_location='cpu'),strict=True);backbone.fc=nn.Identity();backbone.eval();transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
 vectors=[]
 with torch.inference_mode():
  for i,r in enumerate(rows):
   cl,el,h=temporal.logits(features[r['ID']].unsqueeze(0));ci,ei=int(cl.argmax()),int(el.argmax());point=torch.cat((h[0,ci],h[0,ei]))
   cc=crop_feature(frame_at(Path(r['path']),ci),backbone,transform);ce=crop_feature(frame_at(Path(r['path']),ei),backbone,transform);vectors.append(torch.cat((point,cc,ce)))
   if (i+1)%10==0:save('status.json',{'status':'EXTRACTING','completed':i+1,'expected':66})
 x=torch.stack(vectors);mean=x.mean(0);std=x.std(0,unbiased=False).clamp_min(.1);xn=(x-mean)/std
 ev=torch.tensor([int(r['evasion_space']) for r in rows]);side=torch.tensor([int(r['entry_side']=='RIGHT') for r in rows]);states=[];train_predictions=[]
 for seed in SEEDS:
  torch.manual_seed(seed);probe=nn.Linear(2816,4);opt=torch.optim.AdamW(probe.parameters(),lr=1e-3,weight_decay=.1)
  for _ in range(EPOCHS):
   logits=probe(xn);loss=nn.functional.cross_entropy(logits[:,:2],ev)+nn.functional.cross_entropy(logits[:,2:],side);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  probe.eval();states.append(probe.state_dict());train_predictions.append(probe(xn).detach())
 scene=torch.stack(train_predictions).mean(0);training={'evasion_accuracy':float((scene[:,:2].argmax(1)==ev).float().mean()),'side_accuracy':float((scene[:,2:].argmax(1)==side).float().mean())}
 checkpoint={'model':temporal.state_dict(),'labels':p.OUTPUT_COLUMNS,'crop_format':'stage2-semantic-crop-ensemble-v1','crop_seeds':list(SEEDS),'crop_epochs':EPOCHS,'crop_mean':mean,'crop_std':std,'crop_probes':states}
 model_path=OUT/'best.pt';torch.save(checkpoint,model_path)
 start=inference.index('class _Stage2Temporal');end=inference.index('# BASELINE_INFERENCE_PART',start);new_inference=inference[:start]+STAGE2_REPLACEMENT+inference[end:];ast.parse(new_inference)
 target_path=OUT/'submit.zip'
 with zipfile.ZipFile(BASE) as src,zipfile.ZipFile(target_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as dst:
  for info in src.infolist():
   data=new_inference.encode('utf-8') if info.filename=='inference.py' else model_path.read_bytes() if info.filename=='model/stage2/best.pt' else src.read(info.filename);dst.writestr(info.filename,data)
 with zipfile.ZipFile(BASE) as src,zipfile.ZipFile(target_path) as dst:
  if dst.testzip() is not None or dst.namelist()!=src.namelist():raise RuntimeError('ZIP structure/CRC failure')
  changed=[n for n in dst.namelist() if dst.read(n)!=src.read(n)]
  if changed!=['model/stage2/best.pt','inference.py']:raise RuntimeError(f'Unexpected changes {changed}')
 digest=inspect_zip(target_path)
 validation={'status':'STATIC_CPU_PARITY_PASS_GPU_PENDING','sha256':digest,'bytes':target_path.stat().st_size,'base_zip_sha256':sha(BASE),'changed_entries':changed,'stage2_checkpoint_sha256':sha(model_path),'inference_sha256':sha_bytes(new_inference.encode()),'training_rows':66,'sources':len({r['source_id'] for r in rows}),'seeds':list(SEEDS),'epochs':EPOCHS,'training_fit_diagnostic':training,'zip_crc_pass':True,'submitted':False,'gpu_check':'Not run','seconds':time.monotonic()-started}
 save('validation.json',validation);save('status.json',{'status':'ZIP_READY_CPU_VALIDATED_GPU_PENDING','sha256':digest});print(json.dumps(validation,indent=2))

if __name__=='__main__':
 try:main()
 except Exception as exc:
  if OUT.exists():save('status.json',{'status':'FAILED','error':repr(exc)})
  raise
