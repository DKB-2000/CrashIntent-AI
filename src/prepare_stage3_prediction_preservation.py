"""Freeze warm-start CE versus teacher-preserving fine-tuning; no new data."""
import json
from pathlib import Path
from prepare_stage3_straight_sample_control import sha

P=Path(__file__).resolve().parents[1]
PREV=P/'artifacts/stage3-right-recovery-20260914'
ROOT=P/'artifacts/stage3-prediction-preservation-20260914'
REF=P/'artifacts/stage3-representation-20260910'

def main():
    ROOT.mkdir(exist_ok=False)
    prior=json.loads((PREV/'plan.json').read_text());review=json.loads((PREV/'final-review.json').read_text())
    assert review['status']=='COMPLETE_VALIDATED'
    features=json.loads((PREV/'feature-provenance.json').read_text())
    assert sha(PREV/'training-features.npz')==features['training_features_sha256']
    for name in ['repeat-samples.csv','transition-samples.csv','schedule.csv']:
        assert sha(PREV/name)==prior['outputs_sha256'][name]
        (ROOT/name).write_bytes((PREV/name).read_bytes())
    source=(P/'src/train_stage3_right_recovery.py').read_text(encoding='utf-8')
    def change(a,b):
        nonlocal source
        assert a in source,a;source=source.replace(a,b)
    change("ROOT=P/'artifacts/stage3-right-recovery-20260914'","ROOT=P/'artifacts/stage3-prediction-preservation-20260914'\nFEATURES=P/'artifacts/stage3-right-recovery-20260914'")
    start=source.index('    xnew=np.empty(');end=source.index("    ck=torch.load(REF/'20260910-motion/model.pt'",start)
    source=source[:start]+'''    feature_record=json.loads((FEATURES/'feature-provenance.json').read_text())
    assert base.sha(FEATURES/'training-features.npz')==feature_record['training_features_sha256']
    with np.load(FEATURES/'training-features.npz') as z:
        np.testing.assert_array_equal(z['baseline'],xold);xnew=z['transition'].copy()
    assert xnew.shape==(len(new),xold.shape[1]) and np.isfinite(xnew).all()
    np.testing.assert_array_equal(xnew[:1000],xold)
    print('Frozen features and original1000 exact; no video extraction',flush=True)
'''+source[end:]
    change("xs={'repeat_original':norm(xold),'add_transition':norm(xnew)}","xs={'repeat_original':norm(xnew),'add_transition':norm(xnew)}")
    change("frames={'repeat_original':old,'add_transition':new}","frames={'repeat_original':new,'add_transition':new}")
    change("torch.manual_seed(seed);model=base.Control();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)",'''torch.manual_seed(seed);model=base.Control()
            initial=torch.load(REF/f'{seed}-motion/model.pt',weights_only=True,map_location='cpu')
            model.load_state_dict(initial['model'],strict=True)
            teacher=base.Control();teacher.load_state_dict(initial['model'],strict=True);teacher.eval()
            for param in teacher.parameters():param.requires_grad_(False)
            with torch.inference_mode():
                _,teacher_logits=teacher(norm(xold))
                targets=torch.softmax(teacher_logits/plan['temperature'],dim=1).clone()
            initial_a,initial_s=predict(model,xv)
            with np.load(REF/f'{seed}-motion/validation-logits.npz') as reference:
                np.testing.assert_array_equal(initial_a,reference['accel']);np.testing.assert_array_equal(initial_s,reference['steer'])
            opt=torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=0.)''')
    change("col='repeat_index' if arm=='repeat_original' else 'transition_index'","col='transition_index'")
    change('losses=[];model.train()','losses=[];penalties=[];model.train()')
    change("loss=F.cross_entropy(a,ta[ids])+F.cross_entropy(s[moving],ts[ids][moving])",'''loss=F.cross_entropy(a,ta[ids])+F.cross_entropy(s[moving],ts[ids][moving])
                    anchor=moving & torch.from_numpy(ids<1000)
                    penalty=torch.zeros(())
                    if arm=='add_transition' and anchor.any():
                        temperature=plan['temperature']
                        penalty=F.kl_div(F.log_softmax(s[anchor]/temperature,dim=1),targets[ids[anchor.numpy()]],reduction='batchmean')*temperature**2
                        loss=loss+plan['preservation_weight']*penalty
                    penalties.append(float(penalty.detach()))''')
    change("updates=len(losses)))","updates=len(losses),preservation_penalty=float(np.mean(penalties))))")
    block="""            if arm=='repeat_original':
                with np.load(PREV/arm/str(seed)/'validation-logits.npz') as prior_logits:
                    np.testing.assert_array_equal(aa,prior_logits['accel']);np.testing.assert_array_equal(ss,prior_logits['steer'])
"""
    change(block,'')
    # Rename local experimental arms clearly; previous straight-only reference path stays fixed.
    change("'repeat_original'","'warmstart_ce'")
    # Replace new arm names except path to the earlier straight-only checkpoint.
    change("'add_transition'","'warmstart_preserve'")
    change("PREV/'warmstart_preserve'/str(seed)","PREV/'add_transition'/str(seed)")
    # Per-arm adoption against original; teacher effect versus CE is reported separately.
    begin=source.index('    pooled=[r for r in rows');finish=source.index('    for path,h in provenance.items()',begin)
    source=source[:begin]+'''    pooled=[r for r in rows if r['route']=='ALL'];gates={}
    for candidate in plan['arms']:
        delta=[next(r['f1'] for r in pooled if r['seed']==seed and r['arm']==candidate)-next(r['f1'] for r in pooled if r['seed']==seed and r['arm']=='original') for seed in plan['seeds']]
        rd=[float(np.mean([r['f1'] for r in rows if r['route']==route and r['arm']==candidate])-np.mean([r['f1'] for r in rows if r['route']==route and r['arm']=='original'])) for route in sorted(valid.route.unique())]
        recall=np.mean([r['recall'] for r in pooled if r['arm']==candidate],0)-np.mean([r['recall'] for r in pooled if r['arm']=='original'],0)
        latency=True
        for r in [r for r in transition if r['arm']==candidate]:
            b=next(b for b in transition if b['seed']==r['seed'] and b['kind']==r['kind'] and b['arm']=='original')
            latency &= r['missed']<=b['missed'] and r['mean_delay_s'] is not None and b['mean_delay_s'] is not None and r['mean_delay_s']<=b['mean_delay_s']+.1
        gates[candidate]=dict(seed_f1_deltas=delta,route_f1_deltas=rd,recall_delta=recall.tolist(),latency_pass=bool(latency),passed=bool(all(v>0 for v in delta) and sum(v>0 for v in rd)>=3 and min(recall)>=-.02 and latency))
    preservation_effect=dict(seed_f1_deltas=[next(r['f1'] for r in pooled if r['seed']==seed and r['arm']=='warmstart_preserve')-next(r['f1'] for r in pooled if r['seed']==seed and r['arm']=='warmstart_ce') for seed in plan['seeds']],
        recall_delta=(np.mean([r['recall'] for r in pooled if r['arm']=='warmstart_preserve'],0)-np.mean([r['recall'] for r in pooled if r['arm']=='warmstart_ce'],0)).tolist())
'''+source[finish:]
    change("all(g['passed'] for g in gates.values())","any(g['passed'] for g in gates.values())")
    change('gates=gates,metrics=rows','gates=gates,preservation_effect=preservation_effect,metrics=rows')
    change('Input/manifest hashes; new feature anchors exact; frozen normalization exact;', 'Input/manifest/features hashes; all6 initial checkpoints exact; frozen normalization exact;')
    (P/'src/train_stage3_prediction_preservation.py').write_text(source,encoding='utf-8')
    inputs=[Path(__file__),P/'src/train_stage3_prediction_preservation.py',PREV/'plan.json',PREV/'final-review.json',PREV/'feature-provenance.json',PREV/'training-features.npz']
    inputs += [REF/f'{seed}-motion/model.pt' for seed in prior['seeds']]
    plan=dict(status='READY_NO_TRAINING',arms=['warmstart_ce','warmstart_preserve'],seeds=prior['seeds'],added_right=90,epochs=12,updates_per_arm=1560,
        learning_rate=.0001,temperature=2.,preservation_weight=1.,
        policy='Paired original-checkpoint warm starts, fresh AdamW, same1180samples/schedule/1560updates/frozen normalization. CE on both heads; preserve adds T^2 KL teacher||student steer on original moving samples only, all directions equally. Teacher uses original1000 only. Fixed T2/weight1; no sweep or validation selection.',
        gate='For each arm versus original: all3seed F1 improve,>=3/4route means improve,no mean class recall loss>2pp,perseed turn-start/end misses cannot rise and detected mean delay rise<=0.1s. Candidate followup only; report preservation vs CE separately.',
        limits='Reused development routes, sensor proxy labels; teacher errors may be preserved. No guarantee of right recovery or official improvement. Final checkpoints only; no automatic deployment.',
        inputs_sha256={str(p.relative_to(P)):sha(p) for p in inputs},outputs_sha256={p.name:sha(p) for p in ROOT.glob('*.csv')})
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8');print(json.dumps({k:plan[k] for k in ['status','arms','learning_rate','temperature','preservation_weight','updates_per_arm']},indent=2))

if __name__=='__main__':main()
