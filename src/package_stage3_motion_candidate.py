"""Build a separately preserved motion candidate and validate CPU deployment parity."""
import ast,hashlib,importlib.util,json,shutil,zipfile
from pathlib import Path
import numpy as np,pandas as pd,torch
from daily_submission import inspect_zip

OUT=Path('artifacts/stage3-motion-submit-candidate-20260910')
GPU=Path('artifacts/kaggle-stage3-motion-candidate-20260910')
SOURCE=Path('artifacts/stage3-representation-20260910')
BASE=Path('artifacts/stage2-submit-candidate-20260910/submit.zip')
MODULE=Path('src/stage3_motion_inference.py')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def main():
 torch.set_num_threads(2)
 import cv2
 cv2.setNumThreads(1)
 OUT.mkdir(exist_ok=True);GPU.mkdir(exist_ok=True)
 verification=json.loads((SOURCE/'independent-validation.json').read_text())
 assert verification['status']=='PASS' and sha(SOURCE/'results.json')==verification['results_sha256']
 source=SOURCE/'20260910-motion';report=json.loads((source/'report.json').read_text())
 assert sha(source/'model.pt')==report['checkpoint_sha256']
 assert sha(BASE)=='c0e8f2e3fc5847650e2e4a63f3e2b48e4058ad34edfbfe24a9a05333a4cf7047'
 ckpt=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
 ckpt.update(format='stage3-motion-control-v1',accel_classes=['ACCELERATING','DECELERATING','CONSTANT','STOPPED'],steer_classes=['LEFT','STRAIGHT','RIGHT'])
 torch.save(ckpt,OUT/'best.pt')
 with zipfile.ZipFile(BASE) as z:
  original=z.read('inference.py').decode('utf-8-sig');tree=ast.parse(original)
  old=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='predict_stage3'];assert len(old)==1
  lines=original.splitlines(keepends=True);fn=old[0]
  replacement=''.join(lines[:fn.lineno-1]+lines[fn.end_lineno:])+'\n'+MODULE.read_text(encoding='utf-8-sig')
  after=ast.parse(replacement)
  before_nodes={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name!='predict_stage3'}
  after_nodes={n.name:ast.dump(n,include_attributes=False) for n in after.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
  assert all(after_nodes[k]==v for k,v in before_nodes.items())
  assert len([n for n in after.body if isinstance(n,ast.FunctionDef) and n.name=='predict_stage3'])==1
  assert len(set(n.name for n in after.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))))==len([n for n in after.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))])
  path=OUT/'submit.zip'
  with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as target:
   for name in z.namelist():target.writestr(name,replacement.encode() if name=='inference.py' else (OUT/'best.pt').read_bytes() if name=='model/stage3/best.pt' else z.read(name))
  with zipfile.ZipFile(path) as target:
   assert z.namelist()==target.namelist()
   for name in z.namelist():
    if name not in ['inference.py','model/stage3/best.pt']:assert z.read(name)==target.read(name)
 (OUT/'inference.py').write_text(replacement,encoding='utf-8')
 digest=inspect_zip(path)
 spec=importlib.util.spec_from_file_location('motion_candidate_check',OUT/'inference.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 device=torch.device('cpu');model,mean,std=module.s3_motion_load(OUT/'best.pt',device)
 frame=pd.read_csv(SOURCE/'validation-samples.csv')
 ref=pd.read_csv(source/'validation-predictions.csv')
 reference_logits=np.load(source/'validation-logits.npz');predictions=[];max_error=0.
 for sid,g in frame.groupby('ID',sort=False):
  with np.load(SOURCE/'cache'/(sid+'.npz')) as features:
   assert features['endpoints'].tolist()==g.endpoint.tolist()
   motion=features['features'][:,1280:]
  a,s=module.s3_motion_logits(model,motion,mean,std,device)
  for actual,head in [(a,'accel'),(s,'steer')]:
   expected=reference_logits[head][g.index]
   max_error=max(max_error,float(np.max(np.abs(actual-expected))))
   np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
  predictions.extend(zip([sid]*len(g),g.endpoint,np.array(module._S3M_ACCEL)[a.argmax(1)],np.array(module._S3M_STEER)[s.argmax(1)]))
 pred=pd.DataFrame(predictions,columns=module._S3M_COLUMNS)
 pd.testing.assert_frame_equal(pred,ref,check_dtype=False)
 # Decode real validation video in each of two routes, including all startup frames.
 replay=[]
 for sid in ['COMMA2K19_C1_0022','COMMA2K19_C1_0075']:
  g=frame[frame.ID==sid];video=(Path('artifacts/stage3-comma-chunk1-calibrated-v1')/g.video.iloc[0]).resolve()
  actual=module.s3_motion_features(video)
  with np.load(SOURCE/'cache'/(sid+'.npz')) as z:np.testing.assert_array_equal(actual,z['features'][:,1280:])
  replay.append(dict(ID=sid,rows=len(actual)));print('deployment feature parity',sid,len(actual),flush=True)
 # Existing already-used public GPU fixtures. Add expected CPU output inside fixtures only.
 dataset=GPU/'dataset';dataset.mkdir(exist_ok=True)
 fixtures=Path('artifacts/kaggle-stage3-candidate-20260910/dataset/public-fixtures.bin')
 public=OUT/'public-fixture';public.mkdir(exist_ok=True)
 with zipfile.ZipFile(fixtures) as z:
  video=public/'OPEN_001.mp4';video.write_bytes(z.read('stage3/videos/OPEN_001.mp4'))
  expected=module.s3_motion_predict_video(model,video,mean,std,device)
  expected.to_csv(OUT/'expected-stage3.csv',index=False)
  with zipfile.ZipFile(dataset/'public-fixtures.bin','w',zipfile.ZIP_DEFLATED) as target:
   for name in z.namelist():target.writestr(name,z.read(name))
   target.writestr('expected-stage3.csv',expected.to_csv(index=False).encode())
 shutil.copyfile(path,dataset/'candidate.bin')
 save(dataset/'candidate-assets.json',{'candidate.bin':digest,'public-fixtures.bin':sha(dataset/'public-fixtures.bin')})
 shutil.copyfile('artifacts/kaggle-stage3-candidate-20260910/dataset/dataset-metadata.json',dataset/'dataset-metadata.json')
 result=dict(status='CPU_PASS_GPU_PENDING',candidate_sha256=digest,bytes=path.stat().st_size,
  baseline_sha256=sha(BASE),source_checkpoint_sha256=sha(source/'model.pt'),deploy_checkpoint_sha256=sha(OUT/'best.pt'),
  selection='first fixed seed 20260910, not validation-best seed',validation_f1=report['metrics']['validation']['steer']['macro_f1'],
  cached_rows=len(pred),classification_matches=True,max_logits_error=max_error,real_video_replay=replay,public_expected_rows=len(expected),
  preserved_ast_nodes=len(before_nodes),stage1_stage2_model_bytes_unchanged=True,
  code_sha256=sha(MODULE),submitted=False)
 save(OUT/'validation.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
