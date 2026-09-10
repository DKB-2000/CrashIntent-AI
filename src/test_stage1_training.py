"""Leakage, integrity and scoring regressions for full Stage1 training."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_stage1_gpu

import pandas as pd
import torch
import copy

from prepare_stage1_training import plan
from train_stage1_full import audit, metrics, BalancedBatches, accumulated_step, planned_epochs


class Stage1TrainingTests(unittest.TestCase):
    def test_full_bundle_rejects_preview_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame([dict(output_path='preview.mp4')]).to_csv(root / 'generation_manifest.csv', index=False)
            (root / 'generation_report.json').write_text(json.dumps(dict(status='PASS', samples=1, sources=1)))
            output = root / 'bundle'
            with patch('sys.argv', ['prepare_stage1_gpu.py', '--dataset-dir', str(root), '--output-dir', str(output), '--mode', 'full']):
                with self.assertRaisesRegex(ValueError, 'Full training requires'):
                    prepare_stage1_gpu.main()
            self.assertFalse(output.exists())

    def test_balanced_batches_cover_both_classes_and_reproduce_epochs(self):
        labels = ['ORIGINAL'] * 3 + ['RERECORDED'] * 9
        sampler = BalancedBatches(labels, 8)
        batches = list(sampler)
        self.assertEqual(len(batches), 3)
        for batch in batches:
            self.assertEqual([labels[i] for i in batch].count('ORIGINAL'), 4)
        self.assertEqual(set(sum(batches, [])), set(range(len(labels))))
        self.assertEqual(batches, list(sampler))
        sampler.epoch = 1
        self.assertNotEqual(batches, list(sampler))
        with self.assertRaises(ValueError):
            BalancedBatches(labels, 7)
        with self.assertRaises(ValueError):
            BalancedBatches(['ORIGINAL'], 8)

    def test_accumulated_update_matches_full_batch_with_partial_microbatch(self):
        torch.manual_seed(7)
        x = torch.randn(7, 3)
        y = torch.tensor([0, 0, 1, 1, 1, 0, 1])
        for weights in (None, torch.tensor([1.5, .75])):
            full = torch.nn.Linear(3, 2)
            split = copy.deepcopy(full)
            opt = torch.optim.SGD(full.parameters(), lr=.02)
            other = torch.optim.SGD(split.parameters(), lr=.02)
            expected = torch.nn.functional.cross_entropy(full(x), y, weight=weights)
            opt.zero_grad(); expected.backward()
            torch.nn.utils.clip_grad_norm_(full.parameters(), 1.)
            opt.step()
            loss, norm = accumulated_step(split, other, x, y, 2, torch.device('cpu'), weights)
            self.assertAlmostEqual(loss, float(expected.detach()), places=6)
            for left, right in zip(full.parameters(), split.parameters()):
                torch.testing.assert_close(left, right)

    def test_small_data_cannot_finish_with_too_few_updates(self):
        self.assertEqual(planned_epochs(6, 120, 5), 24)
        self.assertEqual(planned_epochs(6, 120, 440), 6)
        self.assertGreaterEqual(planned_epochs(1, 121, 5) * 5, 121)

    def test_macro_f1_penalizes_collapse(self):
        rows = [dict(target='ORIGINAL', answer='RERECORDED'),
                dict(target='RERECORDED', answer='RERECORDED')]
        self.assertAlmostEqual(metrics(rows)['macro_f1'], 1 / 3)
        rows[0]['answer'] = 'ORIGINAL'
        self.assertEqual(metrics(rows)['macro_f1'], 1.)

    def test_phone_siblings_are_excluded_and_groups_do_not_overlap(self):
        root = Path(__file__).resolve().parents[1]
        rows, phone, excluded = plan(root / 'data_raw/ccd/videos/Crash-1500', root / 'data_raw/ccd/Crash-1500.txt')
        self.assertEqual(len(phone), 30)
        self.assertGreater(len(excluded), 30)
        self.assertEqual(len(rows) + len(excluded), 1500)
        self.assertFalse({r['source_id'] for r in rows} & set(excluded))
        self.assertFalse({r['upload_group'] for r in rows if r['split'] == 'train'} &
                         {r['upload_group'] for r in rows if r['split'] == 'val'})

    def test_audit_rejects_group_leakage_holdout_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index, (split, label) in enumerate([('train', 'ORIGINAL'), ('train', 'RERECORDED'),
                                                   ('val', 'ORIGINAL'), ('val', 'RERECORDED')]):
                path = root / f'{index}.mp4'
                path.write_bytes(b'test')
                rows.append(dict(source_id=f'{index // 2:06d}', upload_group=f'group{index // 2}',
                                 split=split, label=label, output_path=path.name,
                                 sha256=hashlib.sha256(b'test').hexdigest()))
            def write_manifest():
                pd.DataFrame(rows).to_csv(root / 'generation_manifest.csv', index=False)
                (root / 'generation_report.json').write_text(json.dumps(dict(manifest_sha256=hashlib.sha256((root / 'generation_manifest.csv').read_bytes()).hexdigest())))
            pd.DataFrame({'source_id': ['000099']}).to_csv(root / 'excluded_sources.csv', index=False)
            write_manifest()
            self.assertEqual(len(audit(root)), 4)
            rows[2]['upload_group'] = 'group0'
            write_manifest()
            with self.assertRaisesRegex(ValueError, 'upload_group leakage'):
                audit(root)
            rows[2]['upload_group'] = 'group1'
            write_manifest()
            pd.DataFrame({'source_id': ['000000']}).to_csv(root / 'excluded_sources.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'Phone holdout leakage'):
                audit(root)
            pd.DataFrame({'source_id': ['000099']}).to_csv(root / 'excluded_sources.csv', index=False)
            (root / '0.mp4').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum/path mismatch'):
                audit(root)


if __name__ == '__main__':
    unittest.main()
