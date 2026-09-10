import copy,unittest
import torch
from torch import nn
from torch.nn import functional as F
from diagnose_stage3_optimization import accumulated_backward

class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.a=nn.Linear(5,4);self.s=nn.Linear(5,3)
    def forward(self,x):return self.a(x),self.s(x)

class AccumulationTest(unittest.TestCase):
    def check_targets(self,targets):
        torch.manual_seed(17)
        model=Tiny().double();reference=copy.deepcopy(model)
        x=torch.randn(len(targets),5,dtype=torch.float64)
        ta=torch.tensor(targets);ts=torch.arange(len(targets))%3
        actual=accumulated_backward(model,x,ta,ts,2)
        a,s=reference(x);valid=ta!=3
        loss=F.cross_entropy(a,ta)+(F.cross_entropy(s[valid],ts[valid]) if valid.any() else s.sum()*0)
        loss.backward()
        self.assertAlmostEqual(actual,float(loss.detach()),places=12)
        for p,q in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=1e-12,rtol=1e-12)
    def test_uneven_moving_chunks(self):self.check_targets([0,3,1,2,3,1,0])
    def test_all_stopped(self):self.check_targets([3,3,3])
if __name__=="__main__":unittest.main()
