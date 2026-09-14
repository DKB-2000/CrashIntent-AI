import unittest
import numpy as np
import torch
import compare_stage2_spatial_motion as m


class FeatureTests(unittest.TestCase):
    def test_constant_frame_and_no_motion(self):
        x = m.regions(np.full((96, 160), 255, np.uint8), np.zeros((96, 160, 2), np.float32)).reshape(16, 5)
        self.assertTrue(torch.equal(x[:, 0], torch.ones(16)))
        self.assertEqual(float(x[:, 1:].abs().sum()), 0.)

    def test_motion_direction_and_scale(self):
        flow = np.zeros((96, 160, 2), np.float32)
        flow[..., 0], flow[..., 1] = -3, 4
        x = m.regions(np.zeros((96, 160), np.uint8), flow).reshape(16, 5)
        self.assertTrue(torch.allclose(x[0], torch.tensor([0., 0., -3/160, 4/96, 5/160])))

    def test_masks_disjoint_and_complete(self):
        a, b = m.mode_mask('motion'), m.mode_mask('spatial')
        self.assertEqual(int(a.sum()), 96)
        self.assertEqual(int(b.sum()), 64)
        self.assertTrue(torch.equal(a+b, m.mode_mask('combined')))
        self.assertEqual(float((a*b).sum()), 0.)
        self.assertEqual(float(m.mode_mask('point').sum()), 0.)

    def test_initial_predictions_preserved_with_auxiliary(self):
        model = m.p.Stage2Temporal().eval()
        head = m.head_from(model.scene).eval()
        x = torch.randn(3, 768)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model.scene(x), head(torch.cat([x, torch.randn(3, 160)], 1)), atol=1e-6))

    def test_predicted_event_order_and_boundary(self):
        seq = torch.arange(240).reshape(3, 80)
        self.assertTrue(torch.equal(m.auxiliary(seq, 2, 0), torch.cat([seq[2], seq[0]])))


if __name__ == '__main__':
    unittest.main()
