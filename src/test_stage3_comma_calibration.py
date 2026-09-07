import unittest
import numpy as np
import pandas as pd
from review_stage3_comma_calibration import robust_fit, classify
from finalize_stage3_comma_calibration import choose_threshold

class CalibrationTests(unittest.TestCase):
    def test_recovers_offset_with_outliers(self):
        rng=np.random.default_rng(9)
        speed=rng.uniform(5,35,1200); yaw=rng.uniform(-.08,.08,1200)
        angle=-.25+2.8*yaw/speed*1000+2.1*yaw*speed+rng.normal(0,.08,1200)
        angle[::13]+=rng.normal(0,20,len(angle[::13]))
        frame=pd.DataFrame(dict(speed_mps=speed,yaw=yaw,steering_angle_deg=angle))
        before=frame.copy(deep=True)
        beta,_=robust_fit(frame,'yaw')
        self.assertAlmostEqual(beta[0],-.25,delta=.03)
        self.assertAlmostEqual(beta[1],2.8,delta=.04)
        self.assertAlmostEqual(beta[2],2.1,delta=.04)
        pd.testing.assert_frame_equal(frame,before)

    def test_rejects_unidentifiable_straight_only_fit(self):
        frame=pd.DataFrame(dict(speed_mps=np.ones(100)*20,yaw=np.zeros(100),steering_angle_deg=np.ones(100)))
        with self.assertRaises(ValueError): robust_fit(frame,'yaw')

    def test_threshold_requires_both_diagnostic_constraints(self):
        rows=pd.DataFrame(dict(deadzone_deg=[.5,1.,1.5,2.],near_straight_pose_labeled_straight=[.58,.85,.94,.97],strong_pose_turn_direction_agreement=[.996,.99,.978,.95]))
        self.assertEqual(choose_threshold(rows),1.5)
        with self.assertRaises(ValueError): choose_threshold(rows.iloc[[0,1,3]])

    def test_signed_threshold_boundaries(self):
        labels=classify(np.array([-1.,-.75,-.25,.25,.5]),-.25)
        self.assertEqual(labels.tolist(),['RIGHT','STRAIGHT','STRAIGHT','STRAIGHT','LEFT'])

if __name__=='__main__': unittest.main()