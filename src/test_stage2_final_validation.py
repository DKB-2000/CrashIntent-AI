import unittest
from fractions import Fraction
from prepare_stage2_final_validation import parse_frames
from score_stage2_final_validation import score_rows, label_gate, integer


class FinalValidationTests(unittest.TestCase):
    def setUp(self):
        self.times = {'TEST_ONLY': [Fraction(0), Fraction(1, 10), Fraction(3, 10), Fraction(301, 1000), Fraction(1)]}
        self.row = dict(ID='TEST_ONLY', review_status='ACCEPT', reviewer='synthetic unit fixture', collision_frame='0', entry_frame='0', evasion_space='0', entry_side='LEFT', recording_source='UNKNOWN')
        self.pred = dict(ID='TEST_ONLY', collision_frame='2', entry_frame='3', evasion_space='0', entry_side='LEFT')

    def test_exact_tolerance_and_variable_frame_time(self):
        result = score_rows([self.row], [self.pred], self.times)
        self.assertTrue(result['details'][0]['collision_frame_ok'])
        self.assertFalse(result['details'][0]['entry_frame_ok'])
        self.assertEqual(result['metrics']['four_item_mean'], .75)

    def test_missing_prediction_counts_as_four_errors(self):
        result = score_rows([self.row], [], self.times)
        self.assertEqual(result['metrics']['four_item_mean'], 0)
        self.assertEqual(result['missing_prediction_count'], 1)

    def test_invalid_prediction_values_are_wrong(self):
        for bad in ['-1', '5', 'NaN', 'inf', '1.5', '']:
            result = score_rows([self.row], [dict(self.pred, collision_frame=bad)], self.times)
            self.assertFalse(result['details'][0]['collision_frame_valid'])
        self.assertIsNone(integer('NaN'))

    def test_pending_never_scored(self):
        pending = dict(self.row, review_status='PENDING')
        self.assertEqual(label_gate([pending], self.times)['status'], 'WAITING_FOR_HUMAN_LABELS')
        with self.assertRaises(ValueError):
            score_rows([pending], [self.pred], self.times)

    def test_invalid_truth_and_duplicates_rejected(self):
        for row in [dict(self.row, collision_frame='5'), dict(self.row, reviewer=''), dict(self.row, entry_side='UNKNOWN')]:
            with self.assertRaises(ValueError):
                score_rows([row], [self.pred], self.times)
        with self.assertRaises(ValueError):
            score_rows([self.row], [self.pred, self.pred], self.times)

    def test_exclusion_reason_and_unknown_source(self):
        with self.assertRaises(ValueError):
            label_gate([dict(self.row, review_status='EXCLUDE')], self.times)
        result = score_rows([self.row], [self.pred], self.times)
        self.assertIsNone(result['source_equal_mean'])
        self.assertEqual(result['known_source_groups'], 0)

    def test_pts_parser_contiguity_and_nonzero_origin(self):
        log = 'config in time_base: 1/1000\nn: 0 pts: 20 pts_time:0.02\nn: 1 pts: 55 pts_time:0.055'
        self.assertEqual(parse_frames(log), (1, 1000, [20, 55]))
        for bad in [log.replace('n: 1', 'n: 2'), log.replace('pts: 55', 'pts: 20')]:
            with self.assertRaises(ValueError):
                parse_frames(bad)


if __name__ == '__main__':
    unittest.main()
