"""Validate Stage2 precision evidence and summarize paired prediction changes."""
import argparse,ast,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--bundle-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
 a.output_dir.mkdir(exist_ok=False);torch.set_num_threads(2)
 with zipfile.ZipFile(a.bundle_dir/'dataset/precision.bin') as inp,zipfile.ZipFile(a.result) as z:
  assert inp.testzip() is None and z.testzip() is None
  manifest=json.loads(inp.read('manifest.json'));assert manifest==json.loads(z.read('manifest.json'))
  assert z.read('runner.py')==inp.read('runner.py')
  report=json.loads(z.read('result/report.json'));assert report['status']=='PASS' and report['manifest']==manifest and 'T4' in report['gpu']
  features=torch.load(io.BytesIO(z.read('result/features.pt')),weights_only=True,map_location='cpu')
  cache=torch.load(io.BytesIO(inp.read('features.pt')),weights_only=True,map_location='cpu')
  frame=pd.read_csv(io.BytesIO(z.read('result/predictions.csv')),dtype={'ID':str})
  labels=pd.read_csv(io.BytesIO(inp.read('labels.csv')),dtype={'ID':str}).set_index('ID')
  tree=ast.parse(inp.read('inference.py').decode('utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_Stage2Temporal'];assert len(nodes)==1
  ns={'torch':torch,'nn':nn};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<submitted temporal>','exec'),ns)
  assert len(frame)==66*6*2 and not frame.duplicated(['ID','model','variant']).any()
  log=np.load(io.BytesIO(z.read('result/logits.npz')))
  maximum=0.
  with torch.inference_mode():
   for modelname in ['old','new']:
    model=ns['_Stage2Temporal']();model.load_state_dict(torch.load(io.BytesIO(inp.read(modelname+'.pt')),weights_only=True,map_location='cpu')['model'],strict=True);model.eval()
    for variant,seqs in features.items():
     assert set(seqs)==set(labels.index)
     for sid,seq in seqs.items():
      assert tuple(seq.shape)==(int(labels.loc[sid,'frames']),512) and torch.isfinite(seq).all()
      if variant.startswith('cached'):assert torch.equal(seq,cache[sid])
      h,_=model.r(seq[None]);cl=model.tc(h).squeeze(-1);el=model.te(h).squeeze(-1)
      prefix=f'{modelname}/{variant}/{sid}'
      saved_c,saved_e=log[prefix+'/collision'],log[prefix+'/entry'];saved_s=log[prefix+'/scene']
      ci,ei=int(saved_c.argmax()),int(saved_e.argmax())
      scene=model.scene(torch.cat([h[:,ci],h[:,ei]],1))
      for actual,expected in [(cl.numpy(),saved_c),(el.numpy(),saved_e),(scene.numpy(),saved_s)]:
       maximum=max(maximum,float(np.max(np.abs(actual-expected))))
       np.testing.assert_allclose(actual,expected,rtol=1e-3,atol=1e-3)
      q=frame[(frame.ID==sid)&(frame.model==modelname)&(frame.variant==variant)].iloc[0]
      assert q.collision_frame==ci and q.entry_frame==ei
      assert q.evasion_space==int(saved_s[0,:2].argmax()) and q.entry_side==['LEFT','RIGHT'][int(saved_s[0,2:].argmax())]
  comparisons={};columns=['collision_frame','entry_frame','evasion_space','entry_side']
  pairs=[('cached_cpu','cached_gpu'),('cached_gpu','raw_gpu32'),('raw_gpu32','raw_gpu16'),('jpeg_gpu32','jpeg_gpu16'),('raw_gpu16','jpeg_gpu16'),('cached_cpu','jpeg_gpu16')]
  for modelname in ['old','new']:
   comparisons[modelname]={}
   for left,right in pairs:
    x=frame[(frame.model==modelname)&(frame.variant==left)].set_index('ID').loc[labels.index]
    y=frame[(frame.model==modelname)&(frame.variant==right)].set_index('ID').loc[labels.index]
    changes={c:int((x[c]!=y[c]).sum()) for c in columns}
    comparisons[modelname][left+' -> '+right]={'changed_predictions':changes,'videos_with_any_change':int((x[columns]!=y[columns]).any(axis=1).sum()),'max_collision_shift_frames':int((x.collision_frame-y.collision_frame).abs().max()),'max_entry_shift_frames':int((x.entry_frame-y.entry_frame).abs().max())}
  feature_differences={}
  for left,right in pairs:
   error=[float((features[left][sid]-features[right][sid]).abs().max()) for sid in labels.index]
   feature_differences[left+' -> '+right]={'max_absolute_difference':max(error)}
  result={'status':'PASS','gpu':report['gpu'],'samples':66,'prediction_rows':len(frame),'max_cpu_recomputed_logit_error':maximum,'comparisons':comparisons,'feature_differences':feature_differences,'note':'Inference sensitivity on reused training samples; JPEG recreated locally, not official images. CPU/GPU temporal logits checked with 1e-3 tolerance.'}
  (a.output_dir/'result-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
  frame.to_csv(a.output_dir/'predictions.csv',index=False)
  print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
