import unittest
from review_stage3_human import add_segment,dense_rows
class ReviewTests(unittest.TestCase):
 def test_inclusive_unknown_and_stop_mask(self):
  s=add_segment([],1,2,'LEFT','MOVING',5);s=add_segment(s,3,3,'RIGHT','STOPPED',5)
  rows=list(dense_rows('A',5,s));self.assertEqual([r['valid_steer'] for r in rows],[0,1,1,0,0]);self.assertEqual(rows[0]['reviewed'],0);self.assertEqual(rows[3]['steer_label'],'RIGHT')
 def test_reject_overlap_and_bad_interval(self):
  s=add_segment([],0,2,'LEFT','MOVING',5)
  for begin,end in [(2,3),(-1,0),(4,5),(4,3)]:
   with self.assertRaises(ValueError):add_segment(s,begin,end,'RIGHT','MOVING',5)
if __name__=='__main__':unittest.main()
