"""Sensor-only frozen 5-35m/s class-balance scout on unused Chunk3 routes."""
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from download_stage3_civic_pilot import Remote, save, sha
from prepare_stage3_comma2k19 import _deduplicate_times, _smooth

P=Path(__file__).resolve().parents[1]
OUT=P/'artifacts/stage3-civic-5-35-class-scout-20260916'
OLD=P/'artifacts/stage3-civic-low-speed-scout-20260916'
ACQ=P/'artifacts/stage3-civic-low-speed-acquisition-20260916'
CAL=P/'artifacts/stage3-civic-low-speed-proxy-20260916'


def archive():
    source=json.loads((OLD/'plan.json').read_text(encoding='utf-8'))
    remote=Remote(source['source'],source['archive_size'])
    return remote,zipfile.ZipFile(remote)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'plan.json').exists()
    scout=json.loads((OLD/'plan.json').read_text(encoding='utf-8'))
    acquired=json.loads((ACQ/'plan.json').read_text(encoding='utf-8'))
    review=json.loads((CAL/'calibration-review.json').read_text(encoding='utf-8'))
    assert review['status']=='CALIBRATION_VALIDATED' and not review['gate_passed']
    routes=sorted(set(scout['routes'])-set(acquired['routes']))
    selected=[s['segment'] for s in scout['segments'] if s['segment'].rsplit('/',1)[0] in routes]
    assert len(routes)==6 and len(selected)==12
    plan=dict(status='RESERVED_SENSOR_ONLY',routes=routes,segments=selected,
              source=scout['source'],archive_size=scout['archive_size'],revision=scout['revision'],
              offset_deg=review['offset_deg'],
              rules='Frozen existing proxy: speed5-35, abs(yaw)<.12; straight abs(angle)<=.5 and abs(yaw)<.003; turns signed angle>1.5 and yaw>.012; centered11 persistence',
              source_plan_sha256=sha((OLD/'plan.json').read_bytes()),acquisition_plan_sha256=sha((ACQ/'plan.json').read_bytes()),
              calibration_review_sha256=sha((CAL/'calibration-review.json').read_bytes()),script_sha256=sha(Path(__file__).read_bytes()),
              limits='12segments,1GiB transfer,30min watcher,3 finite range attempts,OS lock,no retry',
              scope='Only6 unused disjoint Chunk3 recording routes. Sensor-stratified route selection; no video or model predictions.')
    save(OUT/'plan.json',plan);save(OUT/'status.json',dict(status='PREPARED',completed=0,total=12))
    print(json.dumps(dict(routes=6,segments=12)),flush=True)


def read(archive,name):
    return np.asarray(np.load(io.BytesIO(archive.read(name)),allow_pickle=False))


def signals(archive,seg,offset):
    ft=read(archive,seg+'/global_pose/frame_times').reshape(-1);t=ft[::2]
    names=set(archive.namelist())
    speedkey='speed' if seg+'/processed_log/CAN/speed/t' in names else 'car_speed'
    st,sv=_deduplicate_times('speed',read(archive,seg+f'/processed_log/CAN/{speedkey}/t').reshape(-1),
                             read(archive,seg+f'/processed_log/CAN/{speedkey}/value').reshape(-1))
    at,av=_deduplicate_times('steering',read(archive,seg+'/processed_log/CAN/steering_angle/t').reshape(-1),
                             read(archive,seg+'/processed_log/CAN/steering_angle/value').reshape(-1))
    pos=read(archive,seg+'/global_pose/frame_positions');vel=read(archive,seg+'/global_pose/frame_velocities')
    assert pos.shape==vel.shape==(len(ft),3) and np.isfinite(pos).all() and np.isfinite(vel).all()
    up=pos/np.linalg.norm(pos,axis=1,keepdims=True)
    h=vel-(vel*up).sum(1,keepdims=True)*up
    h=np.column_stack([_smooth(h[:,j],21) for j in range(3)])
    yaw=np.sum(np.cross(h,np.gradient(h,ft,axis=0))*up,axis=1)/np.maximum(np.sum(h*h,axis=1),1.)
    yaw=np.interp(t,ft,_smooth(yaw,21))
    gt=read(archive,seg+'/processed_log/IMU/gyro/t').reshape(-1)
    gv=read(archive,seg+'/processed_log/IMU/gyro/value')
    assert gv.shape==(len(gt),3) and np.isfinite(gv).all()
    mask=np.r_[True,np.diff(gt)>0];gt=gt[mask];gv=gv[mask]
    gy=_smooth(np.interp(t,gt,-gv[:,2]),11)
    speed=np.interp(t,st,sv);angle=np.interp(t,at,av)-offset
    valid=(t>=max(st[0],at[0],gt[0]))&(t<=min(st[-1],at[-1],gt[-1]))&(t>=ft[0]+1)&(t<=ft[-1]-1)
    assert np.isfinite(np.column_stack([speed,angle,yaw,gy])).all()
    return dict(speed=speed,angle=angle,yaw=yaw,gyro=gy,valid=valid)


def cache_path(index):
    return OUT/'signal-cache'/f'{index:02d}.npz'


def load_or_fetch(archive,seg,offset,index):
    """Persist each completed segment so an interrupted remote scan can resume."""
    path=cache_path(index)
    if path.exists():
        with np.load(path,allow_pickle=False) as cached:
            assert str(cached['segment'].item())==seg
            return {key:cached[key] for key in ('speed','angle','yaw','gyro','valid')}
    result=signals(archive,seg,offset)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,segment=np.asarray(seg),**result)
    os.replace(tmp,path)
    return result


def stable(label):
    keep=np.zeros(len(label),bool)
    for i in range(5,len(label)-5):keep[i]=label[i]!='UNKNOWN' and np.all(label[i-5:i+6]==label[i])
    return keep


def run():
    plan=json.loads((OUT/'plan.json').read_text(encoding='utf-8'))
    assert sha(Path(__file__).read_bytes())==plan['script_sha256']
    assert sha((OLD/'plan.json').read_bytes())==plan['source_plan_sha256']
    previous=None
    if (OUT/'execution.json').exists():
        previous=json.loads((OUT/'execution.json').read_text(encoding='utf-8'))
    save(OUT/'execution.json',dict(status='STARTED',pid=os.getpid(),started_at=time.time(),previous=previous))
    remote,z=archive()
    try:
        assert json.loads((OLD/'plan.json').read_text(encoding='utf-8'))['revision']==plan['revision']
        data={}
        for i,seg in enumerate(plan['segments']):
            data[seg]=load_or_fetch(z,seg,plan['offset_deg'],i)
            save(OUT/'status.json',dict(status='SCOUTING',completed=i+1,total=12,pid=os.getpid()))
            print(f'signals {i+1}/12',flush=True)
        rows=[]
        for route in plan['routes']:
            segs=sorted(s for s in data if s.rsplit('/',1)[0]==route)
            assert len(segs)==2
            for target in segs:
                source=next(s for s in segs if s!=target)
                a=data[source];b=data[target]
                mask=a['valid']&(a['speed']>=5)&(a['speed']<=35)
                assert mask.sum()>=30
                high=b['valid']&(b['speed']>=5)&(b['speed']<=35)
                speed=b['speed'];angle=b['angle'];yaw=b['yaw']
                domain=high&(np.abs(yaw)<.12)
                label=np.full(len(speed),'UNKNOWN',dtype='<U8')
                label[domain&(np.abs(angle)<=.5)&(np.abs(yaw)<.003)]='STRAIGHT'
                label[domain&(angle>1.5)&(yaw>.012)]='LEFT'
                label[domain&(angle< -1.5)&(yaw< -.012)]='RIGHT'
                keep=stable(label)
                count={k:int(np.sum(keep&(label==k))) for k in ('LEFT','STRAIGHT','RIGHT')}
                rows.append(dict(route=route,segment=target,domain_rows=int(domain.sum()),
                                 retained=int(keep.sum()),classes=count))
        summary=[]
        for route in plan['routes']:
            group=[r for r in rows if r['route']==route]
            counts={k:sum(r['classes'][k] for r in group) for k in ('LEFT','STRAIGHT','RIGHT')}
            summary.append(dict(route=route,segments=2,domain_rows=sum(r['domain_rows'] for r in group),classes=counts))
        summary.sort(key=lambda r:(-sum(v>=20 for v in r['classes'].values()),-r['domain_rows'],r['route']))
        result=dict(status='COMPLETE_VALIDATED',segments=rows,routes=summary,transfer_bytes=remote.received,
                    class_balanced_routes=[r for r in summary if all(v>=20 for v in r['classes'].values())],
                    validation='Fixed existing5-35 rules and scout reservation; finite source sensors; no video or model predictions used.')
        save(OUT/'report.json',result)
        save(OUT/'status.json',dict(status='COMPLETE_VALIDATED',completed=12,total=12,class_balanced_routes=len(result['class_balanced_routes'])))
        print(json.dumps(dict(status=result['status'],class_balanced_routes=len(result['class_balanced_routes']),transfer_bytes=remote.received)),flush=True)
    finally:z.close()


def watch():
    import msvcrt
    with (OUT/'watch.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            with (OUT/'run.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=P,stdout=log,stderr=subprocess.STDOUT)
                save(OUT/'watch-status.json',dict(status='RUNNING',pid=os.getpid(),child_pid=child.pid,timeout_seconds=1800))
                try:rc=child.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,check=True,timeout=30)
                    raise RuntimeError('Class scout timed out')
                if rc:raise RuntimeError(f'Class scout exit {rc}; see run.log')
            assert json.loads((OUT/'report.json').read_text(encoding='utf-8'))['status']=='COMPLETE_VALIDATED'
            save(OUT/'watch-status.json',dict(status='COMPLETE'))
        except Exception as exc:
            save(OUT/'status.json',dict(status='FAILED',error=str(exc)))
            save(OUT/'watch-status.json',dict(status='FAILED',error=str(exc)))
            raise


if __name__=='__main__':{'prepare':prepare,'run':run,'watch':watch}[sys.argv[1]]()

