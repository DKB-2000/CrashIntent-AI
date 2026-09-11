"""Inference-only paired Stage2 numerical and JPEG sensitivity diagnostic."""
import ast,contextlib,hashlib,json,sys,time
from pathlib import Path
import cv2,numpy as np,pandas as pd,torch
from torch import nn
from PIL import Image
from torchvision.models import resnet18,ResNet18_Weights


def main():
 root=Path(sys.argv[1]);out=root/'result';out.mkdir()
 started=time.monotonic();torch.set_num_threads(2);cv2.setNumThreads(1)
 assert torch.cuda.is_available()
 manifest=json.loads((root/'manifest.json').read_text())
 for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
 rows=pd.read_csv(root/'labels.csv',dtype={'ID':str,'source_id':str}).to_dict('records')
 tree=ast.parse((root/'inference.py').read_text(encoding='utf-8-sig'))
 nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_Stage2Temporal'];assert len(nodes)==1
 ns={'torch':torch,'nn':nn};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual submission Stage2>','exec'),ns)
 models={}
 for name in ['old','new']:
  model=ns['_Stage2Temporal']();model.load_state_dict(torch.load(root/f'{name}.pt',weights_only=True,map_location='cpu')['model'],strict=True);models[name]=model.eval()
 backbone=resnet18(weights=None);backbone.load_state_dict(torch.load(root/'resnet.pt',weights_only=True,map_location='cpu'));backbone.fc=nn.Identity();backbone.cuda().eval()
 transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
 cache=torch.load(root/'features.pt',map_location='cpu',weights_only=True)
 variants=['cached_cpu','cached_gpu','raw_gpu32','raw_gpu16','jpeg_gpu32','jpeg_gpu16']
 saved={v:{} for v in variants};pred=[];logits={}
 with torch.inference_mode():
  for r in rows:
   sid=r['ID'];cap=cv2.VideoCapture(str(root/'videos'/f'{sid}.mp4'));raw=[];jpeg=[]
   while True:
    ok,bgr=cap.read()
    if not ok:break
    raw.append(transform(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))))
    ok,encoded=cv2.imencode('.jpg',bgr);assert ok
    decoded=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    jpeg.append(transform(Image.fromarray(cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB))))
   cap.release();assert len(raw)==r['frames']
   seqs={'cached_cpu':cache[sid],'cached_gpu':cache[sid]}
   for input_name,frames in [('raw',raw),('jpeg',jpeg)]:
    batch=torch.stack(frames).cuda()
    for precision in [32,16]:
     with torch.autocast('cuda',dtype=torch.float16,enabled=precision==16):feat=backbone(batch).float().cpu()
     assert torch.isfinite(feat).all()
     seqs[f'{input_name}_gpu{precision}']=feat
   for v,seq in seqs.items():
    saved[v][sid]=seq
    for name,model in models.items():
     device='cpu' if v=='cached_cpu' else 'cuda'
     model.to(device)
     # Use actual packaged forward, keeping temporal module FP32 like serving.
     x=seq[None].to(device);ci,ei,scene=model(x)
     hidden,_=model.r(x)
     cl=model.tc(hidden).squeeze(-1);el=model.te(hidden).squeeze(-1)
     c,e=int(ci),int(ei)
     pred.append(dict(ID=sid,model=name,variant=v,collision_frame=c,entry_frame=e,evasion_space=int(scene[0,:2].argmax()),entry_side=['LEFT','RIGHT'][int(scene[0,2:].argmax())]))
     logits[f'{name}/{v}/{sid}/collision']=cl.cpu().numpy()
     logits[f'{name}/{v}/{sid}/entry']=el.cpu().numpy()
     logits[f'{name}/{v}/{sid}/scene']=scene.cpu().numpy()
   print('video',sid,flush=True)
 torch.save(saved,out/'features.pt');np.savez_compressed(out/'logits.npz',**logits)
 pd.DataFrame(pred).to_csv(out/'predictions.csv',index=False)
 (out/'report.json').write_text(json.dumps({'status':'PASS','gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'samples':len(rows),'variants':variants,'seconds':time.monotonic()-started,'manifest':manifest,'scope':'Inference sensitivity only; labels were used for final fitting, not independent accuracy test. JPEG is a reconstruction, not official evaluation images.'},indent=2))


if __name__=='__main__':main()
