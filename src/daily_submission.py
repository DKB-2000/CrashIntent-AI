"""Publish a dated copy of the latest GPU-validated submission; never upload."""
import argparse, ast, hashlib, json, shutil, tempfile, zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REGISTRY=ROOT/'artifacts/submission-release/selected.json'
REQUIRED={'inference.py','requirements.txt','model/stage1/best.pt','model/stage2/best.pt','model/stage2/resnet18-f37072fd.pth','model/stage3/best.pt'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def inspect_zip(path):
    if path.stat().st_size>10_000_000_000:raise ValueError('ZIP exceeds 10GB')
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=6 or set(z.namelist())!=REQUIRED:raise ValueError('Unexpected ZIP contents')
        if sum(i.file_size for i in z.infolist())>32_000_000_000:raise ValueError('Uncompressed size exceeds 32GB')
        if z.testzip() is not None:raise ValueError('ZIP CRC failed')
        tree=ast.parse(z.read('inference.py').decode('utf-8-sig'))
        for stage in (1,2,3):
            nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==f'predict_stage{stage}']
            if len(nodes)!=1 or [v.arg for v in nodes[0].args.args]!=['data_dir','model_dir']:raise ValueError('Invalid entrypoint')
    return sha(path)

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.temporary.json');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(path)

def register(candidate,evidence,notes):
    candidate=candidate.resolve();evidence=evidence.resolve()
    for path in (candidate,evidence):
        if not path.is_relative_to(ROOT):raise ValueError('Release inputs must be inside project')
    digest=inspect_zip(candidate)
    with zipfile.ZipFile(evidence) as z:
        if z.testzip() is not None:raise ValueError('Evidence CRC failed')
        report=json.loads(z.read('integration.json'));assets=json.loads(z.read('candidate-assets.json'));install=json.loads(z.read('install.json'))
        if report.get('status')!='PASS' or assets.get('candidate.bin',assets.get('candidate.zip'))!=digest:raise ValueError('GPU evidence does not match candidate')
        if install['exit_code']!=0 or install['seconds']>600:raise ValueError('Installation contract failed')
        if sum(v['seconds'] for v in report['results'].values())>3600:raise ValueError('Inference sample time exceeds limit')
        if not report.get('python_internet_socket_blocked'):raise ValueError('Offline GPU check missing')
        if set(report['results'])!={'stage1','stage2','stage3'}:raise ValueError('Missing Stage checks')
    selected=dict(status='GPU_VALIDATED',candidate=str(candidate.relative_to(ROOT)),sha256=digest,evidence=str(evidence.relative_to(ROOT)),evidence_sha256=sha(evidence),notes=notes,registered_at=datetime.now(timezone.utc).isoformat(),scope='Public-example execution validated; hidden-data runtime and model accuracy not certified')
    write_json(REGISTRY,selected)
    return selected

def publish():
    selected=json.loads(REGISTRY.read_text(encoding='utf-8'))
    if selected['status']!='GPU_VALIDATED':raise ValueError('Candidate is not validated')
    candidate=(ROOT/selected['candidate']).resolve();evidence=(ROOT/selected['evidence']).resolve()
    if not candidate.is_relative_to(ROOT) or not evidence.is_relative_to(ROOT):raise ValueError('Release path escapes project')
    if inspect_zip(candidate)!=selected['sha256'] or sha(evidence)!=selected['evidence_sha256']:raise ValueError('Selected release changed after validation')
    today=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d')
    out=ROOT/'artifacts/daily-submissions'/today;out.mkdir(parents=True,exist_ok=True)
    dest=out/'submit.zip'
    if dest.exists() and sha(dest)!=selected['sha256']:
        out=out/selected['sha256'][:12];out.mkdir(parents=True,exist_ok=True);dest=out/'submit.zip'
    if not dest.exists():
        temp=dest.with_suffix('.partial');shutil.copyfile(candidate,temp)
        if inspect_zip(temp)!=selected['sha256']:raise ValueError('Copied ZIP hash mismatch')
        temp.replace(dest)
    report=dict(selected,status='READY_FOR_MANUAL_UPLOAD',zip=str(dest.relative_to(ROOT)),prepared_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),submitted=False)
    write_json(out/'report.json',report)
    write_json(ROOT/'artifacts/daily-submissions/latest.json',report)
    (out/'README.txt').write_text('Prepared for manual website upload. Not submitted.\nSHA256: '+selected['sha256']+'\n'+selected['scope']+'\n'+selected['notes']+'\n',encoding='utf-8')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    register_parser=sub.add_parser('register');register_parser.add_argument('--candidate',type=Path,required=True);register_parser.add_argument('--evidence',type=Path,required=True);register_parser.add_argument('--notes',required=True)
    sub.add_parser('publish');args=parser.parse_args()
    result=register(args.candidate,args.evidence,args.notes) if args.command=='register' else publish()
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
