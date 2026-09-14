"""Paired diagnostics from saved OOF predictions; no fitting or selection."""
import json
import pandas as pd
from compare_stage2_spatial_motion import ROOT, OUT, CV, SEEDS, MODES, save

assert json.loads((OUT/'validation.json').read_text())['status'] == 'PASS'
original = pd.read_csv(CV/'soft_mixed_15_oof.csv', dtype={'ID': str, 'source_id': str}).set_index('ID')
result = {}
for mode in MODES:
    entries = []
    for seed in SEEDS:
        frame = pd.read_csv(OUT/f'{mode}_{seed}_oof.csv', dtype={'ID': str, 'source_id': str}).set_index('ID')
        control = pd.read_csv(OUT/f'point_{seed}_oof.csv', dtype={'ID': str, 'source_id': str}).set_index('ID').loc[frame.index]
        ref = original.loc[frame.index]
        for col in ['pred_collision_frame', 'pred_entry_frame']:
            assert (frame[col] == ref[col]).all()
        keys = ['collision_frame_ok', 'entry_frame_ok', 'evasion_space_ok', 'entry_side_ok']
        score = frame[keys].astype(float).mean(axis=1)
        base = control[keys].astype(float).mean(axis=1)
        before = ref[keys].astype(float).mean(axis=1)
        fold_delta = (score-base).groupby(frame.fold).mean()
        source_delta = (score-base).groupby(frame.source_id).mean()
        entries.append(dict(seed=seed, development_mean=float(score.mean()), delta_vs_point=float((score-base).mean()),
                            delta_vs_original=float((score-before).mean()),
                            fold_deltas={str(k): float(v) for k, v in fold_delta.items()},
                            source_improved=int((source_delta>0).sum()), source_worse=int((source_delta<0).sum()),
                            evasion_prediction_counts={str(k): int(v) for k, v in frame.pred_evasion_space.value_counts().items()},
                            entry_side_gained=int((frame.entry_side_ok & ~control.entry_side_ok).sum()),
                            entry_side_lost=int((~frame.entry_side_ok & control.entry_side_ok).sum())))
    result[mode] = entries
# The expanded zero-input control must reproduce the earlier point adaptation predictions.
parity = True
for seed in SEEDS:
    old = pd.read_csv(ROOT/f'artifacts/stage2-scene-context-20260910/point_{seed}_oof.csv', dtype={'ID': str}).set_index('ID')
    new = pd.read_csv(OUT/f'point_{seed}_oof.csv', dtype={'ID': str}).set_index('ID').loc[old.index]
    columns = ['pred_collision_frame', 'pred_entry_frame', 'pred_evasion_space', 'pred_entry_side']
    parity = parity and old[columns].equals(new[columns])
assert parity
save('paired-analysis.json', dict(status='PASS', previous_point_predictions_equal=True, results=result,
                                 conclusion='No deployment promotion: motion helps vs adapted point but mean remains below original source-CV model; evasion unchanged'))
print(json.dumps(result, indent=2))
