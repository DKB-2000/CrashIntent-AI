"""Independently audit GPU pilot results; never register a model."""
import argparse,hashlib,io,json,math,zipfile
from pathlib import Path
import numpy as np
import torch
from stage1_quality_trial_runner import ARMS,DESIGNS,STEPS,LR,schedule,audit_rows,rates,training_rows
from prepare_stage1_robustness import digest,write_json


def verify_predictions(rows,predictions):
    if [p['path'] for p in predictions]!=[r['path'] for r in rows]:raise ValueError('Prediction coverage/order mismatch')
    for row,p in zip(rows,predictions):
        for key in ('source_id','upload_group','condition','suite','sha256'):
            if row[key]!=p[key]:raise ValueError('Prediction provenance mismatch')
        if p['target']!=row['label']:raise ValueError('Prediction target mismatch')
        logits=np.asarray(p['slot_logits'],dtype=float);probs=np.asarray(p['slot_probabilities'],dtype=float)
        if logits.shape!=(3,2) or probs.shape!=(3,) or not np.isfinite(logits).all() or not np.isfinite(probs).all() or not ((probs>=0)&(probs<=1)).all():raise ValueError('Invalid logits/probabilities')
        e=np.exp(logits-logits.max(axis=1,keepdims=True));expected=e[:,1]/e.sum(axis=1)
        if not np.allclose(expected,probs,atol=5e-4,rtol=0) or not math.isclose(float(probs.mean()),p['probability'],abs_tol=1e-12,rel_tol=0):raise ValueError('Probability recomputation mismatch')
        if p['answer']!=('RERECORDED' if p['probability']>=.5 else 'ORIGINAL'):raise ValueError('Threshold mismatch')


def macro_f1(rows):
    values=[]
    for label in ('ORIGINAL','RERECORDED'):
        tp=sum(r['target']==r['answer']==label for r in rows)
        denominator=sum(r['target']==label for r in rows)+sum(r['answer']==label for r in rows)
        values.append(2*tp/denominator if denominator else 0.)
    return sum(values)/2


def verify_summary(expected,saved):
    if expected.keys()!=saved.keys():raise ValueError('Summary conditions mismatch')
    for condition,values in expected.items():
        actual=saved[condition]
        if values.keys()!=actual.keys():raise ValueError('Summary fields mismatch')
        for key,value in values.items():
            # Python versions can sum these floats with different rounding.
            # Counts and classification rates must still match exactly.
            if key=='mean_probability':
                equal=math.isclose(value,actual[key],abs_tol=1e-12,rel_tol=0)
            else:equal=value==actual[key]
            if not equal:raise ValueError(f'Summary mismatch: {condition}/{key}')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--bundle-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    a.output_dir.mkdir(exist_ok=False);torch.set_num_threads(1)
    bundle=json.loads((a.bundle_dir/'bundle.json').read_text());asset=a.bundle_dir/'dataset/quality.bin'
    if digest(asset)!=bundle['sha256']:raise ValueError('Local bundle changed')
    with zipfile.ZipFile(asset) as inp,zipfile.ZipFile(a.result) as z:
        if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Corrupt/duplicate result archive')
        manifest=json.loads(inp.read('manifest.json'))
        if json.loads(z.read('manifest.json'))!=manifest or z.read('runner.py')!=inp.read('runner.py'):raise ValueError('Remote code/input identity mismatch')
        plan=json.loads(inp.read('plan.json'));train,val=audit_rows(plan['rows'],plan['excluded'])
        arms=plan.get('arms',list(ARMS))
        if arms not in DESIGNS:raise ValueError('Unsupported arm design')
        report=json.loads(z.read('result/report.json'))
        if report['status']!='PASS' or report['completed_arms']!=arms or report['manifest']!=manifest or report['training_steps_per_arm']!=STEPS or report['lr']!=LR or report['threshold']!=.5 or report['precision']!='FP32 training / CUDA FP16 evaluation' or 'T4' not in report['gpu']:raise ValueError('Incomplete/wrong GPU experiment')
        schedule_bytes=z.read('result/schedule.json')
        if hashlib.sha256(schedule_bytes).hexdigest()!=report['schedule_sha256'] or json.loads(schedule_bytes)!=schedule(sorted({r['source_id'] for r in train})):raise ValueError('Training schedule mismatch')
        initial=torch.load(io.BytesIO(inp.read('initial.pt')),map_location='cpu',weights_only=True)['model']
        summaries={};predictions={};normal={}
        for arm in arms:
            saved=[json.loads(line) for line in z.read(f'result/{arm}-predictions.jsonl').decode().splitlines()]
            verify_predictions(val,saved);summary=rates(saved)
            verify_summary(summary,json.loads(z.read(f'result/{arm}-summary.json')))
            predictions[arm]=saved;summaries[arm]=summary;normal[arm]=macro_f1([p for p in saved if p['suite']=='normal'])
            (a.output_dir/(arm+'-predictions.jsonl')).write_bytes(z.read(f'result/{arm}-predictions.jsonl'))
            if arm=='baseline':continue
            trace=[json.loads(line) for line in z.read(f'result/{arm}-steps.jsonl').decode().splitlines()]
            if [p['step'] for p in trace]!=list(range(1,STEPS+1)) or any(not math.isfinite(p['loss']) or not math.isfinite(p['gradient_norm']) for p in trace):raise ValueError('Training trace incomplete')
            if any(a in arms for a in ('mixed_50','mixed_25','screen_mix_25')):
                for step,entries in enumerate(json.loads(schedule_bytes)):
                    expected=[dict(path=r['path'],slot=s) for r,s in training_rows(train,arm,step,entries)]
                    if trace[step].get('samples')!=expected:raise ValueError('Actual training sample schedule mismatch')
            data=z.read(f'result/{arm}.pt')
            if hashlib.sha256(data).hexdigest()!=report['checkpoint_sha256'][arm]:raise ValueError('Checkpoint checksum mismatch')
            ckpt=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
            if ckpt['arm']!=arm or ckpt['optimizer_steps']!=STEPS or ckpt['initial_sha256']!=manifest['initial.pt'] or ckpt['size']!=224 or ckpt['frames']!=16:raise ValueError('Checkpoint provenance mismatch')
            weights=ckpt['model']
            if weights.keys()!=initial.keys() or any(v.shape!=initial[k].shape or not torch.isfinite(v).all() for k,v in weights.items()) or not any(not torch.equal(v,initial[k]) for k,v in weights.items()):raise ValueError('Invalid or unchanged trained weights')
            (a.output_dir/(arm+'.pt')).write_bytes(data);del data,ckpt,weights
        del initial
        decisions={}
        for arm in arms[1:]:
            base=summaries['baseline'];candidate=summaries[arm]
            noise=all(candidate[c]['errors']<base[c]['errors'] for c in ('original_noise','original_quality_mix'))
            rr=all(candidate[c]['errors']<=base[c]['errors']+1 for c in candidate if c.startswith('rerecorded_'))
            keep=normal[arm]>=normal['baseline']-.02
            decisions[arm]=dict(noise_and_mix_improved=noise,rr_each_condition_preserved=rr,normal_f1_preserved=keep,promising_pilot=bool(noise and rr and keep))
        result=dict(status='PASS',scope='Single-seed development comparison, not independent phone validation',prediction_rows=len(val)*len(arms),optimizer_updates=STEPS*(len(arms)-1),normal_macro_f1=normal,condition_summaries=summaries,decisions=decisions,checkpoint_sha256=report['checkpoint_sha256'],gpu=report['gpu'],result_sha256=digest(a.result),bundle_sha256=bundle['sha256'],automatically_promoted=False)
        write_json(a.output_dir/'result-validation.json',result)
        print(json.dumps(dict(status='PASS',normal_macro_f1=normal,decisions=decisions)))

if __name__=='__main__':main()
