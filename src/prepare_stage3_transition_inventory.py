"""Inventory train-only transition straight clips with explicit overlap accounting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_stage3_straight_sample_control import sha

P=Path(__file__).resolve().parents[1]
REF=P/'artifacts/stage3-representation-20260910'
DATA=P/'artifacts/stage3-comma-chunk1-calibrated-v1'
OUT=P/'artifacts/stage3-transition-inventory-20260914'

def main():
    OUT.mkdir(exist_ok=False)
    paths=[REF/'train-samples.csv',DATA/'labels_train_candidate.csv',DATA/'signals_and_candidates.csv',DATA/'split_manifest.csv',Path(__file__)]
    hashes={str(p.relative_to(P)):sha(p) for p in paths}
    original=pd.read_csv(paths[0]);labels=pd.read_csv(paths[1]);signals=pd.read_csv(paths[2]);manifest=pd.read_csv(paths[3])
    train=labels.merge(manifest[['ID','route','split','video']],on='ID',validate='many_to_one')
    train=train.merge(signals[['ID','frame_index','speed_mps']],on=['ID','frame_index'],validate='one_to_one')
    assert len(train)==93602 and train.split.eq('train').all()
    assert set(train.route).isdisjoint(manifest.loc[manifest.split!='train','route'])
    oldsets={sid:g.endpoint.to_numpy() for sid,g in original.groupby('ID')}
    events=[];candidates=[]
    for sid,g in train.groupby('ID',sort=True):
        g=g.sort_values('frame_index').reset_index(drop=True)
        frame=g.frame_index.to_numpy();assert np.all(np.diff(frame)==1)
        truth=g.steer_label.to_numpy();moving=g.accel_label.to_numpy()!='STOPPED';old=oldsets.get(sid,np.array([],dtype=int))
        for t in range(1,len(g)):
            before,after=truth[t-1],truth[t]
            if before==after or (before!='STRAIGHT' and after!='STRAIGHT'):continue
            if not (moving[t-1] and moving[t]):continue
            kind='turn_end' if after=='STRAIGHT' else 'turn_start'
            stable=t>=10 and t+10<=len(g) and moving[t-10:t+10].all() and np.all(truth[t-10:t]==before) and np.all(truth[t:t+10]==after)
            event=f'{sid}:{int(frame[t])}:{kind}'
            indices=range(t,min(t+10,len(g))) if kind=='turn_end' else range(max(0,t-10),t)
            count=0;nonoverlap=0;selected=0
            for j in indices:
                if truth[j]!='STRAIGHT' or not moving[j] or frame[j]<15:continue
                # Stop at another intervening transition; only this adjacent straight run.
                lo,hi=(t,j+1) if kind=='turn_end' else (j,t)
                if not np.all(truth[lo:hi]=='STRAIGHT'):continue
                endpoint=int(frame[j]);distance=int(np.abs(old-endpoint).min()) if len(old) else 100000
                overlap=distance<16
                count+=1;nonoverlap+=not overlap;selected+=distance==0
                candidates.append(dict(event=event,ID=sid,route=g.route.iloc[0],kind=kind,transition_endpoint=int(frame[t]),endpoint=endpoint,
                    stable_event=bool(stable),accel_label=g.accel_label.iloc[j],steer_label='STRAIGHT',speed_mps=float(g.speed_mps.iloc[j]),
                    video=g.video.iloc[0],nearest_original_endpoint_distance=distance,overlaps_original_clip=overlap,already_selected=distance==0))
            events.append(dict(event=event,ID=sid,route=g.route.iloc[0],kind=kind,stable_event=bool(stable),candidate_frames=count,
                nonoverlap_frames=nonoverlap,original_selected_frames=selected))
    e=pd.DataFrame(events);c=pd.DataFrame(candidates)
    # Primary reservoir: sustained transitions, one frame per event and no shared raw frames
    # with any existing16-frame clip or another newly selected clip.
    pool=c[c.stable_event&~c.overlaps_original_clip].copy()
    pool['speed_bin']=pd.cut(pool.speed_mps,[-np.inf,5,15,25,np.inf],right=False,labels=['under5','5to15','15to25','25plus']).astype(str)
    rng=np.random.default_rng(20260918);pool['order']=rng.random(len(pool))
    selected=[];used_events=set();used={sid:list(v) for sid,v in oldsets.items()}
    for r in pool.sort_values(['order','event','endpoint']).to_dict('records'):
        if r['event'] in used_events:continue
        if any(abs(r['endpoint']-end)<16 for end in used.get(r['ID'],[])):continue
        selected.append(r);used_events.add(r['event']);used.setdefault(r['ID'],[]).append(r['endpoint'])
    s=pd.DataFrame(selected,columns=pool.columns).drop(columns=['order'])
    assert not s[['ID','endpoint']].duplicated().any() and not s.event.duplicated().any()
    for sid,g in s.groupby('ID'):
        points=sorted(g.endpoint)
        assert all(b-a>=16 for a,b in zip(points,points[1:]))
        assert all(abs(x-y)>=16 for x in points for y in oldsets.get(sid,[]))
    original.to_csv(OUT/'preserved-original-1000.csv',index=False)
    pd.testing.assert_frame_equal(pd.read_csv(OUT/'preserved-original-1000.csv'),original)
    e.to_csv(OUT/'events.csv',index=False);c.to_csv(OUT/'candidate-frames.csv',index=False);s.to_csv(OUT/'nonoverlap-reservoir.csv',index=False)
    coverage=[]
    for route in sorted(train.route.unique()):
        for kind in ['turn_end','turn_start']:
            g=e[(e.route==route)&(e.kind==kind)&e.stable_event];chosen=s[(s.route==route)&(s.kind==kind)]
            coverage.append(dict(route=route,kind=kind,stable_events=len(g),events_with_original_point=int((g.original_selected_frames>0).sum()),
                events_with_nonoverlap_candidate=int((g.nonoverlap_frames>0).sum()),reservoir=len(chosen)))
    pd.DataFrame(coverage).to_csv(OUT/'route-coverage.csv',index=False)
    report=dict(status='INVENTORY_VALIDATED_NO_TRAINING',original_preserved=1000,train_routes=int(train.route.nunique()),
        events=len(e),stable_events=int(e.stable_event.sum()),candidate_rows=len(c),unique_candidate_frames=len(c.drop_duplicates(['ID','endpoint'])),
        stable_unique_candidate_frames=len(c[c.stable_event].drop_duplicates(['ID','endpoint'])),
        stable_unique_nonoverlap_frames=len(pool.drop_duplicates(['ID','endpoint'])),reservoir=len(s),reservoir_routes=int(s.route.nunique()),
        reservoir_kind=s.kind.value_counts().to_dict(),reservoir_accel=s.accel_label.value_counts().to_dict(),reservoir_speed=s.speed_bin.value_counts().to_dict(),
        stable_events_with_original_point=int((e[e.stable_event].original_selected_frames>0).sum()),
        policy='Existing1000 unchanged. Train-only fixed labels. Straight/turn boundary,10moving constant labels each side; candidates within1s on straight side, endpoint>=15. One per event;16frame input windows share no frames with original or added clips; random seed20260918. No road/object/human interpretation.',
        caveat='A reservoir, not a balanced training manifest. Sensor label persistence is not human truth. Same route/video and temporal context remain correlated despite disjoint raw frames. Availability does not prove learning deficiency or benefit.',
        next='Design equal-update/equal-class additional exposure controls with original normalization frozen, preserving all original samples. Check route/accel/speed matching before training.',
        inputs_sha256=hashes,outputs_sha256={p.name:sha(p) for p in OUT.glob('*.csv')})
    for path,h in hashes.items():assert sha(P/path)==h
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
