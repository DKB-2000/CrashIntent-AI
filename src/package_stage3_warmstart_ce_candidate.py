"""Package the predeclared warmstart_ce seed into the incumbent submission ZIP."""
import ast
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from daily_submission import inspect_zip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/stage3-warmstart-ce-submit-candidate-20260917'
BASE = ROOT / 'artifacts/stage1-auto-release-20260910/candidate/submit.zip'
SOURCE = ROOT / 'artifacts/stage3-prediction-preservation-20260914/warmstart_ce/20260910/model.pt'
MODULE = ROOT / 'src/stage3_motion_inference.py'
EVAL = ROOT / 'artifacts/stage3-civic-535-balanced-evaluation-20260917'
EXPECTED_BASE_SHA = '2c0090adb06f272c0833cfb151ed479ed9c78273830f724e0439fef220626197'
EXPECTED_SOURCE_SHA = '22aa2f8caf0c5a419a0a5b79f68f204345da1fbb38bfd99c1ec6e66938702af1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    assert sha(BASE) == EXPECTED_BASE_SHA
    assert sha(SOURCE) == EXPECTED_SOURCE_SHA
    review = json.loads((EVAL/'final-review.json').read_text(encoding='utf-8'))
    assert review['status'] == 'COMPLETE_VALIDATED' and review['gates']['warmstart_ce']['passed']

    source_ckpt = torch.load(SOURCE, map_location='cpu', weights_only=True)
    deploy_ckpt = dict(source_ckpt)
    deploy_ckpt.update(
        format='stage3-motion-control-v1',
        mode='motion',
        accel_classes=['ACCELERATING','DECELERATING','CONSTANT','STOPPED'],
        steer_classes=['LEFT','STRAIGHT','RIGHT'],
    )
    deploy_path = OUT/'best.pt'
    torch.save(deploy_ckpt, deploy_path)

    with zipfile.ZipFile(BASE) as base:
        original = base.read('inference.py').decode('utf-8-sig')
        tree = ast.parse(original)
        old = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'predict_stage3']
        assert len(old) == 1
        fn = old[0]
        lines = original.splitlines(keepends=True)
        inference = ''.join(lines[:fn.lineno-1] + lines[fn.end_lineno:]) + '\n' + MODULE.read_text(encoding='utf-8-sig')
        before = {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name!='predict_stage3'}
        after_tree = ast.parse(inference)
        after = {n.name:ast.dump(n,include_attributes=False) for n in after_tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        assert all(after[k] == v for k,v in before.items())
        assert len([n for n in after_tree.body if isinstance(n,ast.FunctionDef) and n.name=='predict_stage3']) == 1
        target_path = OUT/'submit.zip'
        with zipfile.ZipFile(target_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as target:
            for info in base.infolist():
                data = inference.encode('utf-8') if info.filename=='inference.py' else deploy_path.read_bytes() if info.filename=='model/stage3/best.pt' else base.read(info.filename)
                target.writestr(info.filename,data)
    with zipfile.ZipFile(BASE) as base, zipfile.ZipFile(target_path) as target:
        assert base.namelist() == target.namelist()
        changed=[]
        for name in base.namelist():
            if base.read(name) != target.read(name): changed.append(name)
        assert changed == ['inference.py','model/stage3/best.pt']

    (OUT/'inference.py').write_text(inference,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('warmstart_ce_candidate',OUT/'inference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model,mean,std=module.s3_motion_load(deploy_path,torch.device('cpu'))
    source_model=module.S3MotionModel();source_model.load_state_dict(source_ckpt['model'],strict=True);source_model.eval()
    checked=0;max_logits_error=0.0
    for cache in sorted(EVAL.glob('*-features.npz')):
        with np.load(cache,allow_pickle=False) as data: features=data['features']
        a,s=module.s3_motion_logits(model,features,mean,std,torch.device('cpu'))
        x=np.zeros((len(features),1682),np.float32);x[:,1280:]=np.clip((features-source_ckpt['mean'].numpy()[1280:])/source_ckpt['std'].numpy()[1280:],-10,10)
        with torch.inference_mode(): expected_a,expected_s=source_model(torch.from_numpy(x))
        for actual,expected in ((a,expected_a.numpy()),(s,expected_s.numpy())):
            max_logits_error=max(max_logits_error,float(np.max(np.abs(actual-expected))))
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
            np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))
        checked+=len(features)

    digest=inspect_zip(target_path)
    report=dict(status='STATIC_AND_CPU_PARITY_PASS_GPU_PENDING',candidate_sha256=digest,bytes=target_path.stat().st_size,
        base_sha256=sha(BASE),source_checkpoint_sha256=sha(SOURCE),deploy_checkpoint_sha256=sha(deploy_path),
        changed_entries=changed,selection='warmstart_ce fixed seed 20260910; selected by family-level frozen proxy gate, not best seed',
        proxy_macro_f1=review['averages']['warmstart_ce'],proxy_rows=review['scored'],cpu_parity_rows=checked,max_logits_error=max_logits_error,classification_matches=True,
        stage1_stage2_requirements_unchanged=True,zip_crc_pass=True,gpu_check='Not run',submitted=False)
    (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
