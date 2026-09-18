"""Window tests independent of dataset labels."""
import unittest
import numpy as np
from stage3_temporal_motion_experiment import motion_at
class Windows(unittest.TestCase):
 def test_no_future_and_boundary(self):
  f=np.repeat(np.arange(50,dtype=np.float32)[:,None],134,axis=1)
  for t in (0,1,4,20,49):
   a=motion_at(f,t,15,30);changed=f.copy();changed[t+1:]=99999
   np.testing.assert_array_equal(a,motion_at(changed,t,15,30))
   if not t:np.testing.assert_array_equal(a,np.zeros(402,np.float32))
   else:
    self.assertEqual(a[134],np.mean(np.arange(max(1,t-14),t+1,dtype=np.float32)))
    self.assertEqual(a[268],np.mean(np.arange(max(1,t-29),t+1,dtype=np.float32)))
 def test_zero_motion(self):
  for t in range(35):np.testing.assert_array_equal(motion_at(np.zeros((35,134),np.float32),t,3,5),np.zeros(402,np.float32))
if __name__=='__main__':unittest.main()
