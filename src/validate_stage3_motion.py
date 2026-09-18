"""Verify saved motion probes against decoded inputs and independently recompute tables."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch

import stage3_pipeline as p

OUT = Path('artifacts/stage3-motion-route-20260910')
DATA = Path('artifacts/stage3-comma-chunk1-calibrated-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    p.cv2.setNumThreads(1)
    provenance = json.loads((OUT/'provenance.json').read_text())
    raw = json.loads((OUT/'probe-logits.json').read_text())
    summary = json.loads((OUT/'summary.json').read_text())
    samples = pd.read_csv(OUT/'samples.csv')
    assert len(samples) == 48 and not samples[['ID', 'sample_index']].duplicated().any()
    assert samples.groupby(['route', 'steer_label', 'steer_stable']).size().eq(2).all()
    assert samples.accel_label.ne('STOPPED').all()
    assert sha(OUT/'samples.csv') == provenance['samples_sha256'] == raw['signature']['samples_sha256']
    assert sha(OUT/'diagnostic-model.pt') == provenance['checkpoint_sha256'] == raw['signature']['checkpoint_sha256']
    assert sha(Path(__file__).with_name('diagnose_stage3_motion.py')) == raw['signature']['script_sha256']
    assert sha(p.__file__) == raw['signature']['pipeline_sha256']
    for name, key in [('labels_validation_candidate.csv', 'labels_sha256'),
                      ('signals_and_candidates.csv', 'signals_sha256'), ('split_manifest.csv', 'manifest_sha256')]:
        assert sha(DATA/name) == provenance[key]
    rows = raw['rows']
    assert len(rows) == 192
    lookup = {(r['probe_id'], r['mode']): r for r in rows}
    assert len(lookup) == len(rows)
    modes = ['normal', 'shuffle_history', 'reverse_history', 'repeat_last']
    checked_inputs = 0
    for sid, group in samples.groupby('ID'):
        video = (DATA/group.video.iloc[0]).resolve()
        frames = list(p.video_frames(video))
        video_sha = sha(video)
        for sample in group.itertuples():
            end = int(sample.sample_index)
            assert end >= 15
            original = frames[end-15:end+1]
            for mode in modes:
                row = lookup[(int(sample.probe_id), mode)]
                assert row['ID'] == sid and row['sample_index'] == end
                assert row['video_sha256'] == video_sha
                order = row['temporal_indices']
                assert len(order) == 16 and order[-1] == 15
                if mode == 'repeat_last': assert order == [15]*16
                else: assert sorted(order) == list(range(16))
                if mode == 'normal': assert order == list(range(16))
                if mode == 'reverse_history': assert order == list(range(14, -1, -1))+[15]
                if mode == 'shuffle_history': assert order != list(range(16))
                tensor = p.clip_tensor([original[i] for i in order]).unsqueeze(0)
                assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == row['input_sha256']
                for head, size in [('accel', 4), ('steer', 3)]:
                    assert len(row[head+'_logits']) == size and np.isfinite(row[head+'_logits']).all()
                checked_inputs += 1
        del frames
    # Independent per-record prediction/probability reductions, including all strata.
    table = pd.read_csv(OUT/'motion-summary.csv')
    max_error = 0.
    for record in table.itertuples():
        subset = samples if record.dimension == 'all' else samples[samples[record.dimension].astype(str) == record.value]
        assert len(subset) == record.rows
        names = p.ACCEL if record.head == 'accel' else p.STEER
        correct, changed, tv, true_delta = [], [], [], []
        for sample in subset.to_dict('records'):
            now = np.array(lookup[(sample['probe_id'], record.mode)][record.head+'_logits'])
            before = np.array(lookup[(sample['probe_id'], 'normal')][record.head+'_logits'])
            target = names.index(sample[record.head+'_label'])
            now_exp = np.exp(now-now.max()); before_exp = np.exp(before-before.max())
            npb = now_exp/now_exp.sum(); bpb = before_exp/before_exp.sum()
            correct.append(now.argmax() == target)
            changed.append(now.argmax() != before.argmax())
            tv.append(sum(abs(npb-bpb))*.5)
            true_delta.append(npb[target]-bpb[target])
        for actual, expected in [(np.mean(correct), record.accuracy), (np.mean(changed), record.changed_rate),
                                  (np.mean(tv), record.mean_probability_tv), (np.mean(true_delta), record.mean_true_probability_delta)]:
            max_error = max(max_error, abs(float(actual)-expected))
    assert max_error < 1e-12
    # Rebuild full route/stratum data from original labels, sensors and archived predictions.
    truth = pd.read_csv(DATA/'labels_validation_candidate.csv').rename(columns={'frame_index': 'sample_index'})
    manifest = pd.read_csv(DATA/'split_manifest.csv')
    assert manifest.groupby('route').split.nunique().eq(1).all()
    truth = truth.merge(manifest[['ID', 'route']], on='ID', validate='many_to_one')
    signals = pd.read_csv(DATA/'signals_and_candidates.csv')
    truth = truth.merge(signals[['ID', 'sample_index', 'speed_mps', 'steering_angle_deg', 'pose_yaw_left_rad_s']],
                        on=['ID', 'sample_index'], validate='one_to_one')
    version = json.loads((DATA/'label_version.json').read_text())
    truth['angle_band'] = pd.cut((truth.steering_angle_deg-version['steer_offset_deg']).abs(),
        [0,1,2,5,np.inf], labels=['0-1deg','1-2deg_boundary','2-5deg','>5deg'], include_lowest=True).astype(str)
    truth['speed_band'] = pd.cut(truth.speed_mps, [-np.inf,5,15,25,np.inf],
        labels=['<=5mps','5-15mps','15-25mps','>25mps']).astype(str)
    truth['pose_band'] = pd.cut(truth.pose_yaw_left_rad_s.abs(), [0,.003,.012,np.inf],
        labels=['near_straight','weak_turn','strong_turn'], include_lowest=True).astype(str)
    truth['steer_stable'] = False
    for _, g in truth.groupby('ID'):
        g = g.sort_values('sample_index')
        values = g.steer_label.tolist()
        truth.loc[g.index, 'steer_stable'] = [i >= 15 and len(set(values[i-15:i+1])) == 1 for i in range(len(g))]
    strata = pd.read_csv(OUT/'route-strata.csv')
    classes = pd.read_csv(OUT/'route-strata-classes.csv')
    agreement = {}
    for tag, source in provenance['sources'].items():
        assert sha(source['archive']) == source['sha256']
        with zipfile.ZipFile(source['archive']) as z:
            pred = pd.read_csv(z.open('trial-run/validation_predictions.csv')).rename(columns={'accel_label':'accel_pred','steer_label':'steer_pred'})
            if tag == '1000': assert hashlib.sha256(z.read('trial-run/best.pt')).hexdigest() == provenance['checkpoint_sha256']
        f = truth.merge(pred, on=['ID','sample_index'], validate='one_to_one')
        assert len(f) == len(truth) == len(pred) == 18603
        if tag == '1000':
            check = samples.merge(f, on=['ID','sample_index'], validate='one_to_one', suffixes=('', '_source'))
            assert len(check) == len(samples)
            for head, names in [('accel',p.ACCEL), ('steer',p.STEER)]:
                assert check[head+'_label'].equals(check[head+'_label_source'])
                agreement[head] = float(np.mean([names[int(np.argmax(lookup[(int(r.probe_id),'normal')][head+'_logits']))] == getattr(r, head+'_pred_source') for r in check.itertuples()]))
        if tag == '1000':
            fast = f[(f.accel_label != 'STOPPED') & (f.speed_mps > 25)]
            highspeed = []
            for (route, label), g in fast.groupby(['route', 'steer_label']):
                highspeed.append(dict(route=route, steer_label=label, size=len(g),
                                      mean=float((g.steer_pred == label).mean())))
            pd.DataFrame(highspeed).to_csv(OUT/'highspeed-class-recall.csv', index=False)
            straight = f[(f.accel_label != 'STOPPED') & (f.steer_label == 'STRAIGHT')]
            bad = straight[straight.route.str.contains('08-02')]
            other = straight[~straight.route.str.contains('08-02')]
            concentration = dict(problem_route_straight_rows=len(bad),
                problem_route_straight_errors=int((bad.steer_pred != 'STRAIGHT').sum()),
                other_routes_straight_rows=len(other), other_routes_straight_errors=int((other.steer_pred != 'STRAIGHT').sum()))
            (OUT/'route-concentration.json').write_text(json.dumps(concentration,indent=2),encoding='utf-8')
        f = f[f.accel_label != 'STOPPED']
        for row in strata[strata.model == int(tag)].itertuples():
            g = f[f.route == row.route]
            if row.dimension != 'all': g = g[g[row.dimension].astype(str) == row.value]
            cm = np.zeros((3,3),dtype=int)
            for a,b in zip(g.steer_label,g.steer_pred): cm[p.STEER.index(a),p.STEER.index(b)] += 1
            support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
            den = support+predicted
            f1 = np.divide(2*tp,den,out=np.zeros(3),where=den!=0)
            assert len(g) == row.rows
            if len(g):
                assert abs(tp.sum()/len(g)-row.accuracy) < 1e-12
                assert abs(f1.mean()-row.macro_f1) < 1e-12
            cs = classes[(classes.model==int(tag))&(classes.route==row.route)&(classes.dimension==row.dimension)&(classes.value==row.value)].set_index('label')
            for i,name in enumerate(p.STEER):
                assert int(cs.loc[name,'support']) == support[i] and int(cs.loc[name,'predicted']) == predicted[i]
                assert abs(cs.loc[name,'recall']-(tp[i]/support[i] if support[i] else 0)) < 1e-12
    assert agreement == summary['baseline_saved_gpu_class_agreement']
    result = dict(status='PASS', input_hashes_recomputed=checked_inputs, video_count=int(samples.ID.nunique()),
        motion_summary_rows=len(table), route_summary_rows=len(strata), max_motion_metric_error=max_error,
        normal_gpu_class_agreement=agreement, summary_sha256=sha(OUT/'summary.json'),
        logits_sha256=sha(OUT/'probe-logits.json'), validator_sha256=sha(__file__))
    (OUT/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
