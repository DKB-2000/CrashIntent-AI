"""Small invariants for temporal order perturbation and model IO."""
import sys
import numpy as np,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compare_stage3_sequence_cv import alter,SequenceModel
x=np.arange(3*15*134,dtype=np.float32).reshape(3,15,134);original=x.copy()
shuffled=alter(x,'shuffle_past');np.testing.assert_array_equal(shuffled[:,-1],x[:,-1]);np.testing.assert_array_equal(x,original)
for i in range(3):np.testing.assert_array_equal(np.sort(shuffled[i,:14,0]),x[i,:14,0])
assert not np.array_equal(shuffled[:,:14],x[:,:14])
repeated=alter(x,'repeat_current');np.testing.assert_array_equal(repeated,np.repeat(x[:,-1:,:],15,axis=1));np.testing.assert_array_equal(alter(x,'ordered'),x)
torch.set_num_threads(2);model=SequenceModel()
a,s=model(torch.zeros(3,15,134));assert a.shape==(3,4) and s.shape==(3,3) and torch.isfinite(a).all() and torch.isfinite(s).all()
print('PASS order perturbation preserves current and past multiset; no cross-sample mixing; model IO finite')
