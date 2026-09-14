import unittest
from fractions import Fraction as F
from compare_stage2_unlabeled import compare


class ComparisonTests(unittest.TestCase):
    def test_rational_tolerance_and_no_accuracy(self):
        a = dict(ID='x', collision_frame=0, entry_frame=0, evasion_space=0, entry_side='LEFT')
        b = dict(a, collision_frame=1, entry_frame=2, evasion_space=1)
        summary, pairs = compare(dict(incumbent=[a], ccd_pretrained=[b]), {'x':[F(0),F(3,10),F(301,1000)]})
        self.assertFalse(pairs[0]['collision_frame_abs_delta_gt_0_3s'])
        self.assertTrue(pairs[0]['entry_frame_abs_delta_gt_0_3s'])
        self.assertEqual(summary['evasion_space']['paired_counts'], {'0 -> 1':1})
        self.assertIsNone(summary['accuracy'])
        self.assertIsNone(summary['preferred_model'])

    def test_identical_predictions_do_not_establish_accuracy(self):
        a = dict(ID='x', collision_frame=0, entry_frame=0, evasion_space=0, entry_side='RIGHT')
        summary, _ = compare(dict(incumbent=[a], ccd_pretrained=[a]), {'x':[F(0)]})
        self.assertEqual(summary['all_four_exactly_agree'], 1)
        self.assertIsNone(summary['accuracy'])

    def test_invalid_rows_rejected(self):
        a = dict(ID='x', collision_frame=0, entry_frame=0, evasion_space=0, entry_side='LEFT')
        for bad in ([a,a], [dict(a, collision_frame=2)], [dict(a, entry_side='BAD')], []):
            with self.assertRaises(ValueError):
                compare(dict(incumbent=[a], ccd_pretrained=bad), {'x':[F(0)]})


if __name__ == '__main__':
    unittest.main()
