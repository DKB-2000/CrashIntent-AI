"""Validate candidate label splits and decode every frame of review clips."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def require(condition, message):
    if not condition: raise ValueError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir',type=Path,required=True)
    args=parser.parse_args(); out=args.review_dir.resolve()
    split=pd.read_csv(out/'split_manifest.csv'); data=pd.read_csv(out/'signals_and_candidates.csv')
    require(split.ID.is_unique,'Duplicate segment')
    require(split.groupby('route').split.nunique().max()==1,'Route leakage')
    require(set(split.split)=={'train','validation'},'Missing split')
    total=0
    for path in split.video: require((out/path).is_file(),f'Missing video: {path}')
    for part in ('train','validation'):
        labels=pd.read_csv(out/f'labels_{part}_candidate.csv')
        require(list(labels.columns)==['ID','frame_index','accel_label','steer_label'],'Label schema mismatch')
        require(not labels.isna().any().any(),'Missing labels')
        require(not labels.duplicated(['ID','frame_index']).any(),'Duplicate frame')
        require(set(labels.ID)==set(split.loc[split.split==part,'ID']),'Split IDs mismatch')
        expected=data.loc[data.split==part,['ID','frame_index','accel_label','candidate_steer']].rename(columns={'candidate_steer':'steer_label'}).reset_index(drop=True)
        pd.testing.assert_frame_equal(labels,expected)
        for _,group in labels.groupby('ID'): require(np.array_equal(group.frame_index,np.arange(len(group))),'Noncontiguous frame indices')
        total+=len(labels)
    require(total==len(data),'Missing candidate samples')
    review=pd.read_csv(out/'visual_review.csv')
    require(set(review.ID)<=set(split.loc[split.split=='train','ID']),'Review contains validation samples')
    version_path=out/'label_version.json'
    if version_path.is_file():
        version=json.loads(version_path.read_text(encoding='utf-8'))
        angle=data.steering_angle_deg.to_numpy()-version['steer_offset_deg']
        width=version['steer_deadzone_deg']
        expected=np.where(angle>width,'LEFT',np.where(angle < -width,'RIGHT','STRAIGHT'))
        require(np.array_equal(expected,data.candidate_steer),'Version parameters do not match candidate labels')
        combined=pd.read_csv(out/'labels.csv')
        pd.testing.assert_frame_equal(combined,data[['ID','frame_index','accel_label','candidate_steer']].rename(columns={'candidate_steer':'steer_label'}))
    count=0
    for name in review['clip']:
        cap=cv2.VideoCapture(str(out/'clips'/f'{name}.mp4')); n=0
        while True:
            ok,_=cap.read()
            if not ok: break
            n+=1
        fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        require(n==80 and np.isclose(fps,10),f'Invalid review clip: {name}, frames={n}, fps={fps}')
        count+=n
    summary=dict(status='PASS',routes_disjoint=True,train_segments=int((split.split=='train').sum()),validation_segments=int((split.split=='validation').sum()),candidate_label_rows=total,review_clips=len(review),review_frames_decoded=count,training_status='NOT_STARTED; provisional labels' if version_path.is_file() else 'NOT_STARTED; steering threshold unresolved')
    (out/'bundle_validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()