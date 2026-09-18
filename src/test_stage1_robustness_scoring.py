"""Deterministic paired statistics and saved-logit integrity tests."""
import unittest
import numpy as np
import pandas as pd
from evaluate_stage1_robustness import summarize, verify_predictions
from prepare_stage1_robustness import CONDITIONS, ORIGINAL_CONDITIONS

def fixtures():
    rows, predictions = [], []
    for sid in ['001', '002']:
        for condition in CONDITIONS:
            target = 'ORIGINAL' if condition in ORIGINAL_CONDITIONS else 'RERECORDED'
            probability = .2 if target == 'ORIGINAL' else .8
            if sid == '001' and condition == 'original_jpeg':
                probability = .8
            if sid == '002' and condition == 'rerecorded_no_moire':
                probability = .2
            row = dict(source_id=sid, upload_group=sid, condition=condition,
                       output_path=f'{sid}_{condition}.mp4', sha256='dummy', label=target)
            rows.append(row)
            logit = float(np.log(probability / (1-probability)))
            predictions.append(dict(**{k:v for k,v in row.items() if k != 'label'},
                                    target=target, probability=probability,
                                    answer='RERECORDED' if probability >= .5 else 'ORIGINAL',
                                    slot_logits=[[0., logit]]*3, slot_probabilities=[probability]*3))
    return pd.DataFrame(rows), predictions

class ScoreTests(unittest.TestCase):
    def test_paired_false_positive_and_recall(self):
        table, predictions = fixtures()
        verify_predictions(table, predictions)
        summaries, pairs = summarize(predictions)
        lookup = {r['condition']:r for r in summaries}
        self.assertEqual(lookup['original_clean']['false_positive_rate'], 0.)
        self.assertEqual(lookup['original_jpeg']['false_positive_rate'], .5)
        self.assertEqual(lookup['original_jpeg']['original_to_rerecorded'], 1)
        self.assertEqual(lookup['rerecorded_no_moire']['recall'], .5)
        self.assertEqual(lookup['rerecorded_no_moire']['rerecorded_to_original'], 1)
        self.assertEqual(len(pairs), 34)

    def test_incomplete_or_corrupt_saved_results_rejected(self):
        table, predictions = fixtures()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            verify_predictions(table, predictions[:-1])
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            summarize(predictions[:-1])
        predictions[0]['slot_logits'][0] = [0., 12.]
        with self.assertRaisesRegex(ValueError, 'softmax'):
            verify_predictions(table, predictions)

if __name__ == '__main__':
    unittest.main()
