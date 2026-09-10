"""Replace compatible Stage1/2 weights in a validated base ZIP for GPU review."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import torch
import stage1_pipeline as s1
import stage2_pipeline as s2
from daily_submission import inspect_zip

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True);parser.add_argument('--stage1',type=Path)
    parser.add_argument('--stage2',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base_sha=inspect_zip(args.base);replacements={}
    for stage,path in [(1,args.stage1),(2,args.stage2)]:
        if path is None:continue
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        if stage==1:
            if ckpt.get('size')!=224 or ckpt.get('frames')!=16:raise ValueError('Stage1 preprocessing mismatch')
            model=s1.Stage1MViT().net
        else:model=s2.Stage2Temporal()
        model.load_state_dict(ckpt['model'],strict=True)
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite checkpoint')
        replacements[f'model/stage{stage}/best.pt']=path
        del model,ckpt
    if not replacements:raise ValueError('No model replacements supplied')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.base) as src,zipfile.ZipFile(args.output,'x',zipfile.ZIP_DEFLATED) as dst:
        for name in src.namelist():
            if name in replacements:dst.write(replacements[name],name)
            else:dst.writestr(name,src.read(name))
    digest=inspect_zip(args.output)
    report=dict(status='STATIC_PASS_GPU_PENDING',sha256=digest,base_sha256=base_sha,replacements={k:dict(path=str(v),sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in replacements.items()},submitted=False)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
