import unittest
import cv2
import numpy as np
from compare_stage3_fullframe_cv import full_frame, flow_stats, summary


class FullFrameTests(unittest.TestCase):
    def test_landscape_preserves_edges_and_aspect(self):
        frame = np.full((90, 160, 3), 150, np.uint8)
        frame[:, :20] = [0, 0, 255]
        frame[:, -20:] = [255, 0, 0]
        out = full_frame(frame)
        self.assertEqual(out.shape, (96, 96, 3))
        self.assertTrue(np.all(out[:21] == 0) and np.all(out[75:] == 0))
        np.testing.assert_array_equal(out[21:75, :12], np.broadcast_to([255,0,0], (54,12,3)))
        np.testing.assert_array_equal(out[21:75, -12:], np.broadcast_to([0,0,255], (54,12,3)))

    def test_portrait_and_square(self):
        out = full_frame(np.full((160,90,3), 200, np.uint8))
        self.assertTrue(np.all(out[:, :21] == 0) and np.all(out[:, 75:] == 0))
        self.assertTrue(np.all(out[:, 21:75] == 200))
        rng = np.random.default_rng(42)
        frame = rng.integers(0,256,(96,96,3), dtype=np.uint8)
        np.testing.assert_array_equal(full_frame(frame), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def test_static_flow_and_history_windows(self):
        frame = np.arange(96*96, dtype=np.uint8).reshape(96,96)
        np.testing.assert_array_equal(flow_stats(frame, frame), np.zeros(134))
        seq = np.broadcast_to(np.arange(15, dtype=np.float32)[None,:,None], (1,15,134)).copy()
        value = summary(seq)
        np.testing.assert_array_equal(value[0,:134], np.full(134,14))
        np.testing.assert_array_equal(value[0,134:268], np.full(134,12))
        np.testing.assert_array_equal(value[0,268:], np.full(134,7))

    def test_bad_history_rejected(self):
        with self.assertRaises(AssertionError):
            summary(np.zeros((2,14,134)))
        with self.assertRaises(AssertionError):
            summary(np.full((2,15,134), np.nan))


if __name__ == '__main__':
    unittest.main()
