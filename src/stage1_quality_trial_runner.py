"""Fixed three-arm Stage1 quality pilot. CUDA only; no model selection."""
import hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.models.video import mvit_v2_s

ARMS=('baseline','existing_recipe','matched_quality')
DESIGNS=(list(ARMS),[*ARMS,'mixed_50'],['baseline','mixed_50','screen_mix_50'],['baseline','mixed_50','mixed_25','screen_mix_25'])
SEED=20260911
STEPS=128
LR=1e-5


def training_rows(train,arm,step,entries):
    """Keep each source/slot class pair intact; mix exactly two pairs per recipe."""
    if arm not in (*ARMS[1:],'mixed_50','screen_mix_50','mixed_25','screen_mix_25'):raise ValueError('Unknown training arm')
    old={(r['source_id'],int(r['variant'])):r for r in train if r['role']=='existing_recipe'}
    quality={(r['source_id'],r['profile'],r['label']):r for r in train if r['role']=='matched_quality'}
    cues={(r['source_id'],r['profile'],r['cue']):r for r in train if r['role']=='screen_cue'}
    selected=[]
    for pair,entry in enumerate(entries):
        if arm in ('mixed_25','screen_mix_25'):
            recipe='matched_quality' if (pair+step)%4==0 else 'existing_recipe'
        else:
            recipe=('matched_quality' if (pair+step)%2==0 else 'existing_recipe') if arm in ('mixed_50','screen_mix_50') else arm
        for label in ('ORIGINAL','RERECORDED'):
            sid=entry['source_id'];profile=entry['profile']
            row=quality[sid,('clean','noise','quality_mix')[profile],label] if recipe=='matched_quality' else old[sid,0 if label=='ORIGINAL' else 1+profile%2]
            if arm in ('screen_mix_50','screen_mix_25') and recipe=='matched_quality' and label=='RERECORDED':
                row=cues[sid,('clean','noise','quality_mix')[profile],('weak_screen','no_geometry','no_flicker')[(step+pair//2)%3]]
            selected.append((row,entry['slot']))
    return selected


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def schedule(sources):
    rng=random.Random(SEED)
    return [[dict(source_id=rng.choice(sources),profile=rng.randrange(3),slot=rng.randrange(3)) for _ in range(4)] for _ in range(STEPS)]


def audit_rows(rows,excluded):
    if not rows or len({r['path'] for r in rows})!=len(rows):raise ValueError('Duplicate/empty rows')
    if any(r['role'] not in ('existing_recipe','matched_quality','screen_cue','evaluation') for r in rows):raise ValueError('Unknown data role')
    train=[r for r in rows if r['role'] in ('existing_recipe','matched_quality','screen_cue')]
    val=[r for r in rows if r['role']=='evaluation']
    for key in ('source_id','upload_group'):
        if {r[key] for r in train}&{r[key] for r in val}:raise ValueError('Train/validation leakage')
    if {r['source_id'] for r in rows}&set(excluded):raise ValueError('Phone holdout leakage')
    if any(r['label'] not in ('ORIGINAL','RERECORDED') for r in rows):raise ValueError('Label mismatch')
    for arm in ARMS[1:]:
        part=[r for r in train if r['role']==arm]
        expected=6 if arm=='matched_quality' else 3
        sources={r['source_id'] for r in part}
        if len(sources)!=85 or len(part)!=85*expected:raise ValueError('Training arm coverage')
        for sid in sources:
            p=[r for r in part if r['source_id']==sid]
            if arm=='matched_quality':
                if {(r['profile'],r['label']) for r in p}!={(c,l) for c in ('clean','noise','quality_mix') for l in ('ORIGINAL','RERECORDED')}:raise ValueError('Quality pair missing')
            elif {(int(r['variant']),r['label']) for r in p}!={(0,'ORIGINAL'),(1,'RERECORDED'),(2,'RERECORDED')}:raise ValueError('Existing recipe missing')
    if {r['source_id'] for r in train if r['role']=='existing_recipe'}!={r['source_id'] for r in train if r['role']=='matched_quality'}:raise ValueError('Unmatched training sources')
    cues=[r for r in train if r['role']=='screen_cue']
    if cues:
        sources={r['source_id'] for r in train if r['role']=='matched_quality'}
        if len(cues)!=765 or {r['source_id'] for r in cues}!=sources or any(r['label']!='RERECORDED' for r in cues):raise ValueError('Cue coverage/label mismatch')
        for sid in sources:
            if {(r['profile'],r['cue']) for r in cues if r['source_id']==sid}!={(p,c) for p in ('clean','noise','quality_mix') for c in ('weak_screen','no_geometry','no_flicker')}:raise ValueError('Cue profile coverage mismatch')
        groups={r['source_id']:r['upload_group'] for r in train if r['role']=='matched_quality'}
        if any(r['upload_group']!=groups[r['source_id']] for r in cues):raise ValueError('Cue source group mismatch')
    if len(val)!=420 or sum(r['suite']=='stress' for r in val)!=357:raise ValueError('Evaluation coverage')
    return train,val


def rates(predictions):
    groups={}
    for row in predictions:groups.setdefault(row['condition'],[]).append(row)
    return {c:dict(videos=len(p),errors=sum(r['answer']!=r['target'] for r in p),
                   positive_rate=sum(r['answer']=='RERECORDED' for r in p)/len(p),
                   mean_probability=sum(r['probability'] for r in p)/len(p)) for c,p in sorted(groups.items())}


def main(root):
    started=time.monotonic();out=root/'result';out.mkdir(exist_ok=False)
    requested=json.loads((root/'plan.json').read_text()).get('arms',list(ARMS))
    if requested not in DESIGNS:raise ValueError('Unsupported arm design')
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2,allow_nan=False))
    state=dict(status='RUNNING',completed_arms=[],scope='Development pilot; not blind/phone validation',training_steps_per_arm=STEPS,lr=LR)
    write('report.json',state)
    try:
        manifest=json.loads((root/'manifest.json').read_text())
        for name,digest in manifest.items():
            path=(root/name).resolve()
            if not path.is_relative_to(root.resolve()) or sha(path)!=digest:raise ValueError('Bundle input mismatch: '+name)
        plan=json.loads((root/'plan.json').read_text());rows=plan['rows'];train,val=audit_rows(rows,plan['excluded'])
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(2)
        import cv2
        cv2.setNumThreads(1)
        import serving_inference as serving
        device=torch.device('cuda');sources=sorted({r['source_id'] for r in train});batches=schedule(sources)
        write('schedule.json',batches)
        def budget():
            if time.monotonic()-started>10000:raise TimeoutError('Pilot time budget reached')
        def decode(row,slot):
            path=root/row['path']
            return serving._decode_stage1_clip(path,224,serving._clip_ids(path,16,slot,3))
        for arm in requested:
            budget();random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
            model=mvit_v2_s(weights=None);model.head[1]=nn.Linear(model.head[1].in_features,2)
            checkpoint=torch.load(root/'initial.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
            model.to(device)
            losses=[]
            if arm!='baseline':
                for module in model.modules():
                    if isinstance(module,nn.Dropout):module.p=0.
                    if module.__class__.__name__=='StochasticDepth':module.p=0.
                optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.)
                model.train()
                with (out/(arm+'-steps.jsonl')).open('x') as trace:
                    for step,entries in enumerate(batches):
                        budget();optimizer.zero_grad(set_to_none=True);loss_sum=0.
                        selected=training_rows(train,arm,step,entries)
                        for row,slot in selected:
                            clip=decode(row,slot).unsqueeze(0).to(device)
                            logits=model(clip)
                            loss=nn.functional.cross_entropy(logits,torch.tensor([int(row['label']=='RERECORDED')],device=device))/8
                            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                            loss.backward();loss_sum+=float(loss.detach());del clip,logits,loss
                        norm=float(nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                        optimizer.step();losses.append(loss_sum)
                        trace.write(json.dumps(dict(step=step+1,loss=loss_sum,gradient_norm=norm,samples=[dict(path=r['path'],slot=s) for r,s in selected]))+'\n');trace.flush()
                        if (step+1)%16==0:print(arm,step+1,loss_sum,flush=True)
                del optimizer
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},size=224,frames=16,arm=arm,optimizer_steps=STEPS,initial_sha256=manifest['initial.pt']),out/(arm+'.pt'))
            model.eval();predictions=[]
            with torch.inference_mode(),(out/(arm+'-predictions.jsonl')).open('x') as stream:
                for row in val:
                    budget();logits=[];probs=[]
                    for slot in range(3):
                        clip=decode(row,slot).unsqueeze(0).to(device)
                        with torch.autocast('cuda',dtype=torch.float16):values=model(clip)
                        # Match submitted FP16 softmax then convert to float for averaging.
                        probs.append(float(values.softmax(1)[0,1].float()));logits.append(values[0].float().cpu().tolist())
                    probability=float(np.mean(probs))
                    record=dict(path=row['path'],source_id=row['source_id'],upload_group=row['upload_group'],condition=row['condition'],suite=row['suite'],sha256=row['sha256'],target=row['label'],probability=probability,answer='RERECORDED' if probability>=.5 else 'ORIGINAL',slot_logits=logits,slot_probabilities=probs)
                    stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();predictions.append(record)
            write(arm+'-summary.json',rates(predictions))
            state['completed_arms'].append(arm);state.update(arm=arm,seconds=time.monotonic()-started)
            write('report.json',state);del model;torch.cuda.empty_cache()
        state.update(status='PASS',manifest=manifest,torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name(),precision='FP32 training / CUDA FP16 evaluation',threshold=.5,seconds=time.monotonic()-started,
                     checkpoint_sha256={a:sha(out/(a+'.pt')) for a in requested[1:]},schedule_sha256=sha(out/'schedule.json'))
        write('report.json',state)
    except Exception as error:
        state.update(status='FAILED',error=repr(error));write('report.json',state);raise

if __name__=='__main__':main(Path(sys.argv[1]).resolve())
