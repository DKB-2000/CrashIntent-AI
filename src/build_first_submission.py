"""Build a first submission candidate using baseline Stage1/2 and local Stage3 weights."""
import argparse,ast,hashlib,json,zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--stage3-checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    stage3=Path(__file__).with_name('stage3_pipeline.py').read_text(encoding='utf-8');tree=ast.parse(stage3)
    functions={'Stage3MViT','device_for','prepare_frame','clip_tensor','video_frames','autocast','load_model','predict_video','predict_paths','predict_stage3'}
    constants={'ACCEL','STEER','OUTPUT_COLUMNS','FORMAT'}
    pieces=['from collections import deque','from contextlib import nullcontext']
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in functions:pieces.append(ast.get_source_segment(stage3,node))
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in node.targets):pieces.append(ast.get_source_segment(stage3,node))
    with zipfile.ZipFile(args.baseline) as original:
        baseline=original.read('inference.py').decode('utf-8');base_tree=ast.parse(baseline)
        first=next(n for n in base_tree.body if isinstance(n,ast.ClassDef) and n.name=='_Stage3MViT')
        inference='\n'.join(baseline.splitlines()[:first.lineno-1])+'\n'+'\n\n'.join(pieces)+'\n'
        result=ast.parse(inference)
        for name in ('predict_stage1','predict_stage2','predict_stage3'):
            matches=[n for n in result.body if isinstance(n,ast.FunctionDef) and n.name==name]
            assert len(matches)==1 and [a.arg for a in matches[0].args.args]==['data_dir','model_dir']
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.output,'x',zipfile.ZIP_DEFLATED) as z:
            for name in ['requirements.txt','model/stage1/best.pt','model/stage2/best.pt','model/stage2/resnet18-f37072fd.pth']:z.writestr(name,original.read(name))
            z.writestr('inference.py',inference.encode());z.write(args.stage3_checkpoint,'model/stage3/best.pt')
    with zipfile.ZipFile(args.output) as z:
        assert z.testzip() is None and len(z.namelist())==6
        ast.parse(z.read('inference.py'))
    summary=dict(status='STATIC_PASS_GPU_INTEGRATION_PENDING',bytes=args.output.stat().st_size,sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),stage3_sha256=hashlib.sha256(args.stage3_checkpoint.read_bytes()).hexdigest(),limitations=['Stage1/2 baseline smoke weights; Stage2 entry/scene outputs not trained','Stage3 predicts majority classes on external validation','Not submitted; integrated GPU test pending'])
    args.output.with_suffix('.manifest.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
