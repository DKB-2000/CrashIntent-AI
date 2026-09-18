"""Compare low- and normal-speed steering/pose agreement on training routes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
SOURCE = P/'artifacts/stage3-comma-chunk1-calibrated-v1/signals_and_candidates.csv'
OUT = P/'artifacts/stage3-low-speed-sensor-agreement-20260916'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    data = pd.read_csv(SOURCE)
    assert set(data[data.split == 'train'].route).isdisjoint(data[data.split == 'validation'].route)
    data = data[data.split == 'train']
    rows = []
    totals = []
    for name,lo,hi in [('low_2_5',2,5),('normal_5_35',5,35)]:
        frame = data[(data.speed_mps >= lo) & (data.speed_mps < hi) &
                     (data.pose_yaw_left_rad_s.abs() > .012) &
                     (data.steering_corrected_deg.abs() > 1.5)].copy()
        frame['agree'] = np.sign(frame.pose_yaw_left_rad_s) == np.sign(frame.steering_corrected_deg)
        totals.append(dict(bin=name, strong_rows=len(frame), routes=frame.route.nunique(),
                           agreeing=int(frame.agree.sum()), agreement=float(frame.agree.mean())))
        for route,g in frame.groupby('route'):
            rows.append(dict(bin=name, route=route, strong_rows=len(g), agreeing=int(g.agree.sum()),
                             agreement=float(g.agree.mean())))
    csvpath = OUT/'route-agreement.csv'
    pd.DataFrame(rows).sort_values(['bin','route']).to_csv(csvpath,index=False)
    report = dict(status='COMPLETE_VALIDATED', scope='RAV4 training routes only; sensor consistency diagnostic, not official label accuracy',
                  thresholds={'pose_abs_rad_s_gt':.012,'steering_abs_deg_gt':1.5}, bins=totals,
                  validation='Training/validation route disjointness; fixed thresholds; route rows recompute pooled counts',
                  input_sha256=sha(SOURCE), output_sha256=sha(csvpath))
    for item in totals:
        subset = [r for r in rows if r['bin'] == item['bin']]
        assert sum(r['strong_rows'] for r in subset) == item['strong_rows']
        assert sum(r['agreeing'] for r in subset) == item['agreeing']
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(totals,indent=2))


if __name__=='__main__':
    main()
