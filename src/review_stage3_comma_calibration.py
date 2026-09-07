"""Estimate a training-only steering offset and produce a local review bundle.

Sensor agreement is a diagnostic, not competition ground truth. Existing labels
and videos are never overwritten. IMU axes: forward/right/down (comma2k19 README).
"""
from __future__ import annotations
import argparse
import io
import json
import html
import os
import zipfile
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from prepare_stage3_comma2k19 import _smooth, _ffmpeg

SEED = 20260825
CLASSES = ['LEFT', 'STRAIGHT', 'RIGHT']


def classify(angle, offset):
    return np.where(angle-offset > .5, 'LEFT', np.where(angle-offset < -.5, 'RIGHT', 'STRAIGHT'))


def robust_fit(frame, signal):
    speed = frame.speed_mps.to_numpy()
    yaw = frame[signal].to_numpy()
    # Bicycle-model terms: curvature and speed-dependent understeer.
    design = np.column_stack([np.ones(len(frame)), yaw/speed*1000, yaw*speed])
    target = frame.steering_angle_deg.to_numpy()
    if len(frame)<30 or not np.isfinite(design).all() or not np.isfinite(target).all() or np.linalg.matrix_rank(design)<3:
        raise ValueError('Insufficient finite, independent motion samples for offset fit')
    weights = np.ones(len(frame))
    for _ in range(15):
        beta = np.linalg.lstsq(design*np.sqrt(weights[:, None]), target*np.sqrt(weights), rcond=None)[0]
        residual = target-design@beta
        scale = max(1.4826*np.median(np.abs(residual-np.median(residual))), .05)
        weights = np.minimum(1., 1.5*scale/np.maximum(np.abs(residual), 1e-9))
    return beta, dict(offset_deg=float(beta[0]), curvature_gain=float(beta[1]), speed_gain=float(beta[2]), median_abs_error_deg=float(np.median(np.abs(residual))), rows=len(frame))


def read_signals(archive, manifest, root, splits):
    frames=[]
    with zipfile.ZipFile(archive) as source:
        for entry in manifest:
            sid=entry['ID']; prefix=entry['source_segment'].rstrip('/')+'/'
            def load(key):
                return np.load(io.BytesIO(source.read(prefix+key)), allow_pickle=False)
            frame=pd.read_csv(root/'converted'/sid/'labels_debug.csv')
            target=frame.timestamp.to_numpy()
            times=np.asarray(load('processed_log/IMU/gyro/t')).reshape(-1)
            gyro=np.asarray(load('processed_log/IMU/gyro/value'))
            if gyro.shape != (len(times),3) or not np.isfinite(gyro).all() or not np.isfinite(times).all() or np.any(np.diff(times)<0):
                raise ValueError(f'{sid}: invalid gyro data')
            keep=np.r_[True,np.diff(times)>0]; times=times[keep]; gyro=gyro[keep]
            yaw=_smooth(np.interp(target,times,-gyro[:,2]),11)
            idx=np.clip(np.searchsorted(times,target),1,len(times)-1)
            valid=(target>=times[0])&(target<=times[-1])&((times[idx]-times[idx-1])<.2)
            ft=load('global_pose/frame_times').reshape(-1)
            vel=load('global_pose/frame_velocities'); pos=load('global_pose/frame_positions')
            if vel.shape != (len(ft),3) or pos.shape != vel.shape or not np.isfinite(vel).all() or not np.isfinite(pos).all():
                raise ValueError(f'{sid}: invalid pose')
            up=pos/np.linalg.norm(pos,axis=1,keepdims=True)
            horizontal=vel-(vel*up).sum(axis=1,keepdims=True)*up
            horizontal=np.column_stack([_smooth(horizontal[:,j],21) for j in range(3)])
            derivative=np.gradient(horizontal,ft,axis=0)
            pose_yaw=np.sum(np.cross(horizontal,derivative)*up,axis=1)/np.maximum(np.sum(horizontal**2,axis=1),1.)
            frame['gyro_yaw_left_rad_s']=yaw
            frame['pose_yaw_left_rad_s']=np.interp(target,ft,_smooth(pose_yaw,21))
            frame['gyro_valid']=valid
            frame['route']=entry['source_route']; frame['split']=splits[entry['source_route']]
            frame['segment_offset']=entry['steer_offset']
            frames.append(frame)
    return pd.concat(frames,ignore_index=True)


def build_review(data, root, out, offset, deadzone=.5):
    chosen=[]; used=set()
    rules={
        'straight': lambda f: (f.speed_mps>10)&(f.pose_yaw_left_rad_s.abs()<.003),
        'left_curve': lambda f: (f.speed_mps>5)&(f.pose_yaw_left_rad_s>.012),
        'right_curve': lambda f: (f.speed_mps>5)&(f.pose_yaw_left_rad_s<-.012),
        'stopped': lambda f: f.speed_mps<.5,
        'accelerating': lambda f: (f.speed_mps>1)&(f.long_accel_mps2>.5),
        'decelerating': lambda f: (f.speed_mps>1)&(f.long_accel_mps2<-.5),
        'offset_disagreement': lambda f: (f.speed_mps>5)&(f.steer_label!=f.candidate_steer),
    }
    train=data[data.split=='train']
    for kind,rule in rules.items():
        candidates=[]
        for sid,group in train.groupby('ID',sort=True):
            group=group.reset_index(drop=True)
            mask=rule(group).to_numpy().astype(float)
            for start in range(10,len(group)-89,40):
                score=float(mask[start:start+80].mean())
                if score>=.5:
                    candidates.append((score,sid,start,group.iloc[start:start+80].copy()))
        routes=set(); count=0
        for score,sid,start,clip in sorted(candidates,key=lambda x:(-x[0],x[1],x[2])):
            route=str(clip.route.iloc[0])
            if route in routes or any(sid==s and abs(start-t)<80 for s,t in used):
                continue
            chosen.append((kind,score,sid,start,clip)); used.add((sid,start)); routes.add(route); count+=1
            if count==2: break
    clips=out/'clips'; clips.mkdir()
    records=[]; cards=[]; sheets=[]
    for number,(kind,score,sid,start,clip) in enumerate(chosen):
        name=f'{number+1:02d}_{kind}_{sid}_{start:04d}'
        video=root/'converted'/sid/'videos'/f'{sid}.mp4'
        cap=cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        temporary=clips/f'{name}.temporary.mp4'; destination=clips/f'{name}.mp4'
        writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,480))
        snapshots=[]
        for local,(_,row) in enumerate(clip.iterrows()):
            ok,frame=cap.read()
            if not ok: raise ValueError(f'{name}: decode failed at {local}')
            frame=cv2.resize(frame,(640,480))
            cv2.rectangle(frame,(0,0),(640,98),(0,0,0),-1)
            lines=[f'{kind}  {sid}  t={(start+local)/10:.1f}s',f'{row.speed_mps*3.6:.1f} km/h  {row.accel_label}  angle={row.steering_angle_deg:+.2f}',f'OLD {row.steer_label} | V1 {row.candidate_steer} | RAW {row.raw_steer}',f'gyro={row.gyro_yaw_left_rad_s:+.3f} pose={row.pose_yaw_left_rad_s:+.3f} rad/s']
            for j,line in enumerate(lines): cv2.putText(frame,line,(8,19+j*24),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
            writer.write(frame)
            if local in (0,25,50,79): snapshots.append(frame.copy())
        cap.release(); writer.release()
        subprocess.run([_ffmpeg(),'-y','-loglevel','error','-i',str(temporary),'-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p',str(destination)],check=True)
        temporary.unlink()
        sheet=np.vstack([np.hstack(snapshots[:2]),np.hstack(snapshots[2:])]); cv2.imwrite(str(clips/f'{name}.jpg'),sheet)
        records.append(dict(clip=name,category=kind,ID=sid,route=str(clip.route.iloc[0]),split='train',start_frame=start,end_frame=start+79,selection_fraction=score,visual_steer='',visual_accel='',decision='',notes=''))
        cards.append(f'<article><h2>{html.escape(kind)} · {sid} · {start/10:.1f}s</h2><video controls preload="metadata" src="clips/{name}.mp4"></video><p>후보 조건 충족 {score:.0%}. 실제 장면과 OLD/FIXED/RAW를 비교하세요.</p><a href="clips/{name}.jpg">4프레임 요약</a></article>')
        print(f'Review {number+1}/{len(chosen)}: {name}',flush=True)
    pd.DataFrame(records).to_csv(out/'visual_review.csv',index=False)
    document='<!doctype html><meta charset="utf-8"><title>Stage3 조향 검수</title><style>body{font-family:system-ui;max-width:1400px;margin:30px auto;padding:20px;background:#f4f6f8;color:#17212d}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:20px}article{background:white;padding:18px;border-radius:12px}video{width:100%}h2{font-size:18px}</style><h1>Stage3 표본 검수</h1><p>모든 표본은 학습 route에서만 선정했습니다. OLD=구간별 중앙값, V1=학습 데이터로 추정한 고정 영점, RAW=영점 0도. 각 방식의 경계값은 아래에 표시합니다. 모두 임시값입니다. IMU·pose는 정답이 아닌 비교 신호입니다.</p>'+f'<p>V1 offset: {offset:+.4f}°, dead-zone: +/-{deadzone:g}°. OLD/RAW는 초기 +/-0.5도입니다. 판정은 visual_review.csv에 기록합니다. 원본 라벨은 보존했습니다.</p><main>'+''.join(cards)+'</main>'
    (out/'review.html').write_text(document,encoding='utf-8')
    return records


def export_candidates(data, out, offset):
    for part in ('train','validation'):
        table=data.loc[data.split==part,['ID','frame_index','accel_label','candidate_steer']].rename(columns={'candidate_steer':'steer_label'})
        table.to_csv(out/f'labels_{part}_candidate.csv',index=False)
    train=data[(data.split=='train')&(data.speed_mps>=5)&data.gyro_valid]
    angle=train.steering_angle_deg.to_numpy()-offset
    straight=(train.pose_yaw_left_rad_s.abs()<.003).to_numpy()
    turn=(train.pose_yaw_left_rad_s.abs()>.012).to_numpy()
    target=np.where(train.pose_yaw_left_rad_s>0,'LEFT','RIGHT')
    rows=[]
    for width in (.5,1.,1.5,2.):
        prediction=np.where(angle>width,'LEFT',np.where(angle<-width,'RIGHT','STRAIGHT'))
        rows.append(dict(deadzone_deg=width,samples=len(train),near_straight_pose_samples=int(straight.sum()),near_straight_pose_labeled_straight=float((prediction[straight]=='STRAIGHT').mean()),strong_pose_turn_samples=int(turn.sum()),strong_pose_turn_direction_agreement=float((prediction[turn]==target[turn]).mean()),**{k:int((prediction==k).sum()) for k in CLASSES}))
    pd.DataFrame(rows).to_csv(out/'threshold_sensitivity_train.csv',index=False)
    (out/'candidate_label_metadata.json').write_text(json.dumps(dict(status='PROVISIONAL_NOT_TRAINING_VALIDATED',offset_deg=offset,deadzone_deg=.5,selection='Offset fitted on training routes only. Threshold unchanged; sensitivity diagnostics are not ground truth.',source='signals_and_candidates.csv',next='Review steering threshold semantics and temporal stability before model training.'),indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--skip-clips',action='store_true')
    args=parser.parse_args(); root=args.dataset_dir.resolve(); out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    routes=sorted({m['source_route'] for m in manifest}); shuffled=np.random.default_rng(SEED).permutation(routes)
    val=set(shuffled[:max(1,round(len(routes)*.2))]); splits={r:('validation' if r in val else 'train') for r in routes}
    split=pd.DataFrame([dict(ID=m['ID'],route=m['source_route'],split=splits[m['source_route']],video=Path(os.path.relpath(root/"converted"/m["ID"] / "videos" / (m["ID"]+".mp4"),out)).as_posix()) for m in manifest])
    split.to_csv(out/'split_manifest.csv',index=False)
    data=read_signals(args.archive,manifest,root,splits)
    mask=(data.speed_mps>=5)&(data.speed_mps<=35)&data.gyro_valid&(data.gyro_yaw_left_rad_s.abs()<.12)&(data.pose_yaw_left_rad_s.abs()<.12)&(data.frame_index>=10)&(data.frame_index<590)
    fit=data[mask & (data.split=='train') & (data.frame_index%10==0)]
    estimates={}; route_stats=[]
    for signal in ['gyro_yaw_left_rad_s','pose_yaw_left_rad_s']:
        beta,stats=robust_fit(fit,signal); estimates[signal]=stats
        for route,group in fit.groupby('route'):
            if len(group)>=30:
                _,info=robust_fit(group,signal); route_stats.append(dict(route=route,signal=signal,**info))
    offset=estimates['gyro_yaw_left_rad_s']['offset_deg']
    rng=np.random.default_rng(SEED); train_routes=sorted(fit.route.unique()); boot=[]
    for _ in range(100):
        sample=pd.concat([fit[fit.route==r] for r in rng.choice(train_routes,len(train_routes),replace=True)],ignore_index=True)
        beta,_=robust_fit(sample,'gyro_yaw_left_rad_s'); boot.append(float(beta[0]))
    data['raw_steer']=classify(data.steering_angle_deg.to_numpy(),0.)
    data['candidate_steer']=classify(data.steering_angle_deg.to_numpy(),offset)
    data.to_csv(out/'signals_and_candidates.csv',index=False)
    export_candidates(data,out,offset)
    pd.DataFrame(route_stats).to_csv(out/'route_diagnostics.csv',index=False)
    metrics={}
    for part,group in data.groupby('split'):
        moving=group[group.speed_mps>=5]
        strong=moving[moving.pose_yaw_left_rad_s.abs()>.012]
        truth=np.where(strong.pose_yaw_left_rad_s>0,'LEFT','RIGHT')
        metrics[part]=dict(segments=int(group.ID.nunique()),routes=int(group.route.nunique()),samples=len(group),changed_vs_old=int((group.steer_label!=group.candidate_steer).sum()),accel_distribution=group.accel_label.value_counts().to_dict(),candidate_steer_distribution=group.candidate_steer.value_counts().to_dict(),strong_pose_turn_samples=len(strong),direction_agreement={k:float((strong[k].to_numpy()==truth).mean()) for k in ['steer_label','raw_steer','candidate_steer']})
    summary=dict(status='CANDIDATE_REQUIRES_VISUAL_REVIEW',seed=SEED,fit_rows=len(fit),fit_routes=len(train_routes),estimates=estimates,selected_offset_deg=offset,route_bootstrap_offset_interval_95=np.quantile(boot,[.025,.975]).tolist(),split_metrics=metrics,notes=['Fit and bootstrap use training routes only; validation is diagnostic only.','Gyro yaw is negative down-axis angular rate; pose yaw uses horizontal ECEF velocity cross derivative projected onto local up.','No official steering thresholds are inferred. Dead-zone remains provisional +/-0.5 degree.','No original labels overwritten. Offset candidate and steering threshold are not yet validated for training.','Route bootstrap measures route sensitivity, not physical calibration certification.'],source='https://github.com/commaai/comma2k19#log-format')
    (out/'calibration_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    if not args.skip_clips:
        records=build_review(data,root,out,offset)
        print(f'Review bundle: {len(records)} clips, {out / "review.html"}',flush=True)


if __name__=='__main__': main()