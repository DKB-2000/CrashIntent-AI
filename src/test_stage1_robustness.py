"""Check paired controls, leakage rejection and real video round-trip."""
from dataclasses import asdict
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
import prepare_stage1_robustness as robust
import prepare_stage1_rerecorded as synth

class RobustnessTests(unittest.TestCase):
    def test_full_matches_training_and_jpeg_bypass(self):
        base = synth._effect_parameters(random.Random(73))
        frame = np.random.default_rng(2).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        fields = (synth._perspective_matrix(64, 48, base), *synth._spatial_fields(64, 48, base))
        expected = synth._synthetic_frame(frame, 7, 10., base, *fields, np.random.default_rng(5))
        actual = robust.transform(frame, 7, 10., asdict(base), fields, np.random.default_rng(5))
        np.testing.assert_array_equal(actual, expected)
        values = robust.parameters(base, 'rerecorded_no_jpeg')
        with patch.object(cv2, 'imencode', side_effect=AssertionError('JPEG must be bypassed')):
            robust.transform(frame, 7, 10., values, fields, np.random.default_rng(5))

    def test_quality_controls_have_no_screen_cues(self):
        base = synth._effect_parameters(random.Random(2))
        for name in robust.ORIGINAL_CONDITIONS:
            values = robust.parameters(base, name)
            for group, neutral in robust.GROUPS.items():
                if group not in ('jpeg', 'blur', 'noise'):
                    for key, value in neutral.items():
                        self.assertEqual(values[key], value)
        for group, neutral in robust.GROUPS.items():
            values = robust.parameters(base, 'rerecorded_no_' + group)
            for key in values:
                self.assertEqual(values[key], neutral.get(key, getattr(base, key)))
        weak = robust.parameters(base, 'rerecorded_weak_screen')
        for key in ('jpeg_quality', 'blur_sigma', 'noise_sigma'):
            self.assertEqual(weak[key], getattr(base, key))

    def test_group_selection_and_leakage(self):
        table = pd.DataFrame([
            dict(source_id='001', upload_group='train', split='train', variant=0, source_sha256='a'),
            dict(source_id='002', upload_group='val1', split='val', variant=0, source_sha256='b'),
            dict(source_id='003', upload_group='val1', split='val', variant=0, source_sha256='c'),
            dict(source_id='004', upload_group='val2', split='val', variant=0, source_sha256='d')])
        selected = robust.select_sources(table, {'005'}, 1)
        self.assertEqual({r['upload_group'] for r in selected}, {'val1', 'val2'})
        self.assertEqual(selected, robust.select_sources(table.sample(frac=1), {'005'}, 1))
        with self.assertRaisesRegex(ValueError, 'Phone holdout'):
            robust.select_sources(table, {'002'}, 1)
        table.loc[1, 'upload_group'] = 'train'
        with self.assertRaisesRegex(ValueError, 'leakage'):
            robust.select_sources(table, set(), 1)

    def test_real_video_roundtrip_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'out'
            (out / 'records').mkdir(parents=True)
            video = root / '000001.mp4'
            writer = synth.H264Writer(video, 64, 48, 10., 24)
            for _ in range(3):
                writer.write(np.random.default_rng(9).integers(0, 256, (48, 64, 3), dtype=np.uint8))
            writer.close()
            row = dict(source_id='000001', source_sha256=robust.digest(video), upload_group='test')
            task = (row, str(root), str(out), 'test-code')
            rows = robust.render_source(task)
            self.assertEqual(len(rows), 17)
            self.assertEqual(sum(r['frames'] for r in rows), 51)
            self.assertEqual(robust.render_source(task), rows)
            (out / rows[0]['output_path']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                robust.render_source(task)

if __name__ == '__main__':
    unittest.main()
