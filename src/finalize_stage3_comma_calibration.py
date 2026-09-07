"""Freeze a provisional steering label version using training-only diagnostics."""
import argparse,json,os,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from review_stage3_comma_calibration import build_review


def choose_threshold(sensitivity):
    eligible=sensitivity[(sensitivity.near_straight_pose_labeled_straight>=.90)&(sensitivity.strong_pose_turn_direction_agreement>=.97)]
    if eligible.empty: raise ValueError('No threshold satisfies the provisional training-only criteria')
    return float(eligible.sort_values('deadzone_deg').iloc[0].deadzone_deg)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration-dir',type=Path,required=True)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(); source=a.calibration_dir.resolve(); root=a.dataset_dir.resolve(); out=a.output_dir.resolve()
    sensitivity=pd.read_csv(source/'threshold_sensitivity_train.csv')
    width=choose_threshold(sensitivity)
    report=json.loads((source/'calibration_report.json').read_text())
    offset=float(report['selected_offset_deg'])
    data=pd.read_csv(source/'signals_and_candidates.csv')
    angle=data.steering_angle_deg.to_numpy()-offset
    data['candidate_steer']=np.where(angle>width,'LEFT',np.where(angle<-width,'RIGHT','STRAIGHT'))
    out.mkdir(parents=True,exist_ok=False)
    split=pd.read_csv(source/'split_manifest.csv')
    split['video']=[Path(os.path.relpath(root/'converted'/sid/'videos'/f'{sid}.mp4',out)).as_posix() for sid in split.ID]
    split.to_csv(out/'split_manifest.csv',index=False)
    data.to_csv(out/'signals_and_candidates.csv',index=False)
    columns=['ID','frame_index','accel_label','candidate_steer']
    for part in ['train','validation']:
        data.loc[data.split==part,columns].rename(columns={'candidate_steer':'steer_label'}).to_csv(out/f'labels_{part}_candidate.csv',index=False)
    data[columns].rename(columns={'candidate_steer':'steer_label'}).to_csv(out/'labels.csv',index=False)
    metrics={}
    for part,group in data.groupby('split'):
        metrics[part]=dict(samples=len(group),segments=int(group.ID.nunique()),routes=int(group.route.nunique()),accel_distribution=group.accel_label.value_counts().to_dict(),steer_distribution=group.candidate_steer.value_counts().to_dict())
    metadata=dict(status='PROVISIONAL_V1_NOT_OFFICIAL_GROUND_TRUTH',steer_offset_deg=offset,steer_deadzone_deg=width,positive_steer_label='LEFT',threshold_selection='Smallest tested training threshold with >=90% near-straight pose agreement and >=97% strong-turn pose direction agreement. Engineering criteria, not competition metrics.',calibration_source=os.path.relpath(source,out),fit_split='train',split_metrics=metrics,limitations=['Same RAV4 and highway corridor; route split does not prove geographic or vehicle generalization.','Sensor proxies and sparse visual review cannot establish official steering semantics.','Acceleration labels retain previous speed derivative thresholds.','Videos referenced by relative paths in split_manifest.csv; no duplicate video encoding.'])
    (out/'label_version.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    shutil.copyfile(source/'threshold_sensitivity_train.csv',out/'threshold_sensitivity_train.csv')
    build_review(data,root,out,offset,deadzone=width)
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__': main()