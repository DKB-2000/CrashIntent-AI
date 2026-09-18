"""Keep straight augmentation; add matched right diversity at existing extra slots."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_stage3_straight_sample_control import sha

P=Path(__file__).resolve().parents[1]
PREV=P/'artifacts/stage3-transition-matched-control-20260914'
OUT=P/'artifacts/stage3-right-recovery-20260914'
DATA=P/'artifacts/stage3-comma-chunk1-calibrated-v1'
OLD=P/'artifacts/stage3-straight-sample-control-20260914/baseline-samples.csv'

def main():
    OUT.mkdir(exist_ok=False)
    prior=json.loads((PREV/'plan.json').read_text())
    for name,h in prior['outputs_sha256'].items():assert sha(PREV/name)==h
    old=pd.read_csv(PREV/'repeat-samples.csv');new=pd.read_csv(PREV/'transition-samples.csv');schedule=pd.read_csv(PREV/'schedule.csv')
    controls=pd.read_csv(OLD);labels=pd.read_csv(DATA/'labels_train_candidate.csv');manifest=pd.read_csv(DATA/'split_manifest.csv')
    signals=pd.read_csv(DATA/'signals_and_candidates.csv',usecols=['ID','frame_index','speed_mps'])
    train=labels.merge(manifest[['ID','route','split','video']],on='ID',validate='many_to_one').merge(signals,on=['ID','frame_index'],validate='one_to_one')
    assert train.split.eq('train').all()
    train['accel']=train.accel_label.map({'ACCELERATING':0,'DECELERATING':1,'CONSTANT':2,'STOPPED':3})
    train['speed_bin']=pd.cut(train.speed_mps,[-np.inf,5,15,25,np.inf],right=False,labels=['under5','5to15','15to25','25plus']).astype(str)
    train['stable11']=False
    for sid,g in train.groupby('ID'):
        g=g.sort_values('frame_index');assert np.all(np.diff(g.frame_index)==1)
        train.loc[g.index,'stable11']=((g.steer_label=='RIGHT')&(g.accel!=3)).astype(int).rolling(11,center=True,min_periods=11).sum().eq(11).to_numpy()
    pool=train[(train.steer_label=='RIGHT')&(train.accel!=3)&(train.frame_index>=15)&train.stable11]
    used={sid:g.endpoint.tolist() for sid,g in new.groupby('ID')};rng=np.random.default_rng(20260920);matched=[]
    for accel in range(3):
        candidates=controls[(controls.steer==2)&(controls.accel==accel)].copy()
        candidates['order']=rng.random(len(candidates));candidates=candidates.sort_values('order')
        routes={r:g.to_dict('records') for r,g in candidates.groupby('route')};order=list(routes);rng.shuffle(order);count=0
        while any(routes.values()) and count<30:
            for route in order:
                if not routes[route] or count>=30:continue
                b=routes[route].pop()
                g=pool[(pool.route==route)&(pool.accel==accel)&(pool.speed_bin==b['speed_bin'])]
                for idx in rng.permutation(g.index):
                    r=g.loc[idx]
                    if any(abs(int(r.frame_index)-t)<16 for t in used.get(r.ID,[])):continue
                    used.setdefault(r.ID,[]).append(int(r.frame_index))
                    matched.append(dict(ID=r.ID,route=route,endpoint=int(r.frame_index),accel=accel,steer=2,group=int(b['group']),video=r.video,
                        control_slot=int(b['slot']),speed_bin=r.speed_bin,speed_mps=float(r.speed_mps),stable11=True))
                    count+=1;break
    pairs=pd.DataFrame(matched);assert len(pairs)>0 and not pairs.control_slot.duplicated().any()
    pairs['new_index']=np.arange(len(new),len(new)+len(pairs))
    added=pairs.copy();added['record']=added.new_index
    combined=pd.concat([new,added[old.columns]],ignore_index=True)
    original_schedule=schedule.copy();changed=0
    for r in pairs.itertuples():
        mask=(schedule.batch>=100)&(schedule.transition_index==r.control_slot)
        changed+=int(mask.sum());schedule.loc[mask,'transition_index']=r.new_index
    pd.testing.assert_frame_equal(combined.iloc[:1090].reset_index(drop=True),new)
    assert not combined[['ID','endpoint']].duplicated().any()
    assert np.array_equal(schedule.repeat_index,original_schedule.repeat_index)
    assert (schedule.loc[schedule.batch<100,'transition_index']==original_schedule.loc[schedule.batch<100,'transition_index']).all()
    a=new.iloc[original_schedule.transition_index];b=combined.iloc[schedule.transition_index]
    for col in ['route','accel','steer','group']:assert np.array_equal(a[col],b[col])
    # Keep prior straight samples, schedule, and every original sample untouched.
    old.to_csv(OUT/'repeat-samples.csv',index=False);combined.to_csv(OUT/'transition-samples.csv',index=False)
    schedule.to_csv(OUT/'schedule.csv',index=False);pairs.to_csv(OUT/'right-pairs.csv',index=False)
    plan={k:v for k,v in prior.items() if k not in ['inputs_sha256','outputs_sha256']}
    plan.update(status='RIGHT_RECOVERY_READY_NO_TRAINING',added_right=len(pairs),right_per_accel=pairs.accel.value_counts().sort_index().to_dict(),right_routes=int(pairs.route.nunique()),changed_right_exposures=changed,
        policy='Prior straight90 and original1000 preserved. Up to30new right samples per accel group,stable11,rightfixedlabels,matchedroute/accel/speed-bin,16frame nonoverlap. Replace only existing extra-right repeat slots; no extra updates/class counts. Seed20260920.',
        comparison='Original,repeat-original and previous straight-only augmentation are all replayed and reported. Candidate must meet previous gates against original/repeat; previous straight-only tradeoff also reported. No automatic deployment.',
        inputs_sha256={str(p.relative_to(P)):sha(p) for p in [Path(__file__),PREV/'plan.json',PREV/'final-review.json',PREV/'schedule.csv',PREV/'transition-samples.csv',OLD,DATA/'labels_train_candidate.csv',DATA/'split_manifest.csv',DATA/'signals_and_candidates.csv']},
        outputs_sha256={p.name:sha(p) for p in OUT.glob('*.csv')})
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    source=(P/'src/train_stage3_transition_matched_control.py').read_text(encoding='utf-8')
    def change(oldtext,newtext):
        nonlocal source
        assert oldtext in source,oldtext;source=source.replace(oldtext,newtext)
    change("ROOT=P/'artifacts/stage3-transition-matched-control-20260914'","ROOT=P/'artifacts/stage3-right-recovery-20260914'\nPREV=P/'artifacts/stage3-transition-matched-control-20260914'")
    change('assert len(new)==1090','assert len(new)==1090+plan[\'added_right\']')
    change('np.empty((1090,xold.shape[1]),np.float32)','np.empty((len(new),xold.shape[1]),np.float32)')
    change("for arm in ['original']+plan['arms']:","for arm in ['original','straight_only']+plan['arms']:")
    change("folder=REF/f'{seed}-motion' if arm=='original' else ROOT/arm/str(seed)","folder=REF/f'{seed}-motion' if arm=='original' else PREV/'add_transition'/str(seed) if arm=='straight_only' else ROOT/arm/str(seed)")
    change("for a in ['original']+plan['arms']}","for a in ['original','straight_only']+plan['arms']}")
    change("all9 checkpoint/hash/logit replays exact","all12 checkpoint/hash/logit replays exact")
    # Equal-repeat arm is deliberately retrained; require exact previous replay before evaluating candidate.
    change("pred=ss.argmax(1)\n            if arm!='original':", "pred=ss.argmax(1)\n            if arm=='repeat_original':\n                with np.load(PREV/arm/str(seed)/'validation-logits.npz') as prior_logits:\n                    np.testing.assert_array_equal(aa,prior_logits['accel']);np.testing.assert_array_equal(ss,prior_logits['steer'])\n            if arm!='original':")
    (P/'src/train_stage3_right_recovery.py').write_text(source,encoding='utf-8')
    print(json.dumps({k:plan[k] for k in ['status','added_right','right_per_accel','right_routes','changed_right_exposures']},indent=2))

if __name__=='__main__':main()
