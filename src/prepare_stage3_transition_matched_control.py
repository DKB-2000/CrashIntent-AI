"""Freeze matched extra-exposure manifests and schedules; no model training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_stage3_straight_sample_control import sha

P=Path(__file__).resolve().parents[1]
REF=P/'artifacts/stage3-representation-20260910'
INV=P/'artifacts/stage3-transition-inventory-20260914'
OLD=P/'artifacts/stage3-straight-sample-control-20260914'
OUT=P/'artifacts/stage3-transition-matched-control-20260914'
ACCEL={'ACCELERATING':0,'DECELERATING':1,'CONSTANT':2,'STOPPED':3}

def main():
    OUT.mkdir(exist_ok=False)
    report=json.loads((INV/'report.json').read_text())
    assert report['status']=='INVENTORY_VALIDATED_NO_TRAINING'
    for name,h in report['outputs_sha256'].items():assert sha(INV/name)==h
    original=pd.read_csv(REF/'train-samples.csv');base=pd.read_csv(OLD/'baseline-samples.csv')
    pd.testing.assert_frame_equal(original[base.columns.intersection(original.columns)],base[base.columns.intersection(original.columns)])
    pool=pd.read_csv(INV/'nonoverlap-reservoir.csv');pool['accel']=pool.accel_label.map(ACCEL)
    pool['steer']=1
    # Match without replacement to original straight slots within each composition cell.
    rng=np.random.default_rng(20260919);matches=[];availability=[]
    straight=base[(base.steer==1)&(base.accel!=3)]
    for (route,accel,speed),g in pool.groupby(['route','accel','speed_bin'],sort=True):
        controls=straight[(straight.route==route)&(straight.accel==accel)&(straight.speed_bin==speed)]
        n=min(len(g),len(controls));availability.append(dict(route=route,accel=int(accel),speed_bin=speed,candidates=len(g),original_controls=len(controls),matchable=n))
        new=g.iloc[rng.permutation(len(g))[:n]];old=controls.iloc[rng.permutation(len(controls))[:n]]
        for (_,r),(_,b) in zip(new.iterrows(),old.iterrows()):
            matches.append(dict(**r.to_dict(),control_slot=int(b.slot),group=int(b.group),control_ID=b.ID,control_endpoint=int(b.endpoint)))
    pairs=pd.DataFrame(matches)
    counts=pairs.groupby('accel').size();assert set(counts.index)=={0,1,2}
    per_group=min(30,int(counts.min()))
    assert per_group>0
    chosen=[]
    # Round-robin routes within acceleration group limits domination by a single route.
    for accel,g in pairs.groupby('accel',sort=True):
        route_groups={r:h.iloc[rng.permutation(len(h))].to_dict('records') for r,h in g.groupby('route')}
        routes=list(route_groups);rng.shuffle(routes);picked=0
        while picked<per_group:
            for route in routes:
                if route_groups[route] and picked<per_group:
                    chosen.append(route_groups[route].pop());picked+=1
    chosen=pd.DataFrame(chosen).reset_index(drop=True);chosen['pair']=np.arange(len(chosen))
    assert not chosen[['ID','endpoint']].duplicated().any() and not chosen.control_slot.duplicated().any()
    for r in chosen.itertuples():
        b=base.iloc[r.control_slot]
        assert (r.route,r.accel,r.speed_bin,r.group)==(b.route,b.accel,b.speed_bin,b.group)
        assert r.steer==b.steer==1
    candidate=original.copy();extra=chosen[original.columns.intersection(chosen.columns)].copy()
    extra['record']=np.arange(1000,1000+len(extra));extra=extra[original.columns]
    candidate=pd.concat([candidate,extra],ignore_index=True)
    pd.testing.assert_frame_equal(candidate.iloc[:1000].reset_index(drop=True),original)
    assert not candidate[['ID','endpoint']].duplicated().any()
    # All original rows are seen once per epoch. An additional balanced10-group batch
    # repeats seven shared groups, with three straight slots paired old versus new.
    schedules=[]
    for epoch in range(1,13):
        erng=np.random.default_rng(20260919+epoch)
        ids={int(group):erng.permutation(g.index).tolist() for group,g in original.groupby('group')}
        pair_groups={int(group):erng.permutation(g.index).tolist() for group,g in chosen.groupby('group')}
        for batch in range(100+per_group):
            order=erng.permutation(10)
            for position,group in enumerate(order):
                if batch<100:a=b=int(ids[int(group)][batch]);pair=-1
                elif int(group) in pair_groups:
                    pair=int(pair_groups[int(group)][batch-100]);a=int(chosen.iloc[pair].control_slot);b=1000+pair
                else:a=b=int(ids[int(group)][batch-100]);pair=-1
                schedules.append(dict(epoch=epoch,batch=batch,position=position,group=int(group),repeat_index=a,transition_index=b,pair=pair))
    schedule=pd.DataFrame(schedules)
    for epoch,g in schedule.groupby('epoch'):
        assert sorted(g[g.batch<100].repeat_index)==list(range(1000))
        assert np.array_equal(g[g.batch<100].repeat_index,g[g.batch<100].transition_index)
        a=original.iloc[g.repeat_index];b=candidate.iloc[g.transition_index]
        for col in ['accel','steer','group','route']:assert np.array_equal(a[col],b[col])
        for _,batch in g.groupby('batch'):assert sorted(batch.group)==list(range(10))
    original.to_csv(OUT/'repeat-samples.csv',index=False);candidate.to_csv(OUT/'transition-samples.csv',index=False)
    chosen.to_csv(OUT/'matched-pairs.csv',index=False);pairs.to_csv(OUT/'all-matchable-pairs.csv',index=False)
    pd.DataFrame(availability).to_csv(OUT/'availability.csv',index=False);schedule.to_csv(OUT/'schedule.csv',index=False)
    plan=dict(status='MATCHED_CONTROL_READY_NO_TRAINING',original_preserved=1000,matchable=len(pairs),matchable_per_accel={str(k):int(v) for k,v in counts.items()},
        added_straight=len(chosen),extra_batches_per_epoch=per_group,epochs=12,updates_per_arm=12*(100+per_group),seeds=[20260910,20260911,20260912],
        selected_routes=int(chosen.route.nunique()),selected_kinds=chosen.kind.value_counts().to_dict(),selected_speed=chosen.speed_bin.value_counts().to_dict(),
        arms=['repeat_original','add_transition'],normalization='Freeze original mean/std/motion mask from verified reference checkpoint for both arms. Do not refit.',
        model='Same Control MLP,AdamW lr0.001 weight_decay0,clip1,batch10,fixed final checkpoint. Initialization paired per seed.',
        schedule='Every epoch original1000 once in shared order, then matched balanced10-group extra batches. Extra3straight slots repeat original versus new transition; other7slots identical. Same route/accel/steer/speedbin at each paired slot.',
        baseline='Also report original1200-update model; extra-training control has equal extra updates, do not compare candidate only against original.',
        gate='Candidate must improve moving F1 on all3seeds against both controls; improve>=3/4route means; no class recall loss>2pp; perseed turn-start/end misses must not increase and detected mean delay increase<=0.1s. Reused development data, not official improvement.',
        limitations='Compares targeted new transition data versus repeated original data; does not isolate transition targeting from generic new-data diversity. Exact speed/video not matched. Added route exposure differs from original-only model but is identical between extra-training arms.',
        inputs_sha256={str(p.relative_to(P)):sha(p) for p in [Path(__file__),REF/'train-samples.csv',OLD/'baseline-samples.csv',INV/'report.json',INV/'nonoverlap-reservoir.csv',REF/'20260910-motion/model.pt']},
        outputs_sha256={p.name:sha(p) for p in OUT.glob('*.csv')})
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8');print(json.dumps(plan,indent=2))

if __name__=='__main__':main()
