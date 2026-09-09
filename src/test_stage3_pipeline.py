"""Behavior tests for Stage3 temporal alignment, masking and inference isolation."""
import tempfile
import unittest
from unittest.mock import patch
from stage3_pipeline import mixed_clip_groups, train_batches
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from stage3_pipeline import (ACCEL,STEER,causal_indices,clip_tensor,video_frames,
    predict_video,predict_paths,multitask_loss,classification_metrics,load_records,evaluate)


class Recorder(nn.Module):
    def __init__(self):
        super().__init__();self.seen=[]
    def forward(self,x):
        self.seen.extend(t.detach().cpu().clone() for t in x)
        value=x[:,:,-1].mean((1,2,3))
        a=torch.stack([value,-value,value*0,value*0],1)
        s=torch.stack([value,-value,value*0],1)
        return a,s


def make_video(path,values,fps=10):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(64,48))
    if not writer.isOpened(): raise RuntimeError('Video writer unavailable')
    for value in values: writer.write(np.full((48,64,3),value,dtype=np.uint8))
    writer.release()


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2);cv2.setNumThreads(1)

    def test_mixed_plan_coverage_reproducibility_and_epoch_phase(self):
        records=[dict(accel=np.zeros(19+i,dtype=int)) for i in range(9)]
        first=list(mixed_clip_groups(records,3,1,4))
        self.assertEqual(first,list(mixed_clip_groups(records,3,1,4)))
        actual=[sample for group,samples in first for sample in samples]
        expected=[(i,t) for i,r in enumerate(records) for t in range(1,len(r['accel']),3)]
        self.assertCountEqual(actual,expected)
        self.assertEqual(len(actual),len(set(actual)))
        self.assertTrue(all(len(group)<=4 for group,_ in first))
        self.assertGreater(sum(a[0]!=b[0] for a,b in zip(actual,actual[1:])),9)
        self.assertNotEqual(first,list(mixed_clip_groups(records,3,2,4)))

    def test_mixed_batches_preserve_clip_and_label_alignment(self):
        records=[dict(path=i,accel=np.arange(19)%4,steer=np.arange(19)%3) for i in range(3)]
        frames={i:[np.full((3,2,2),i*50+t,dtype=np.uint8) for t in range(19)] for i in range(3)}
        seen=[]
        with patch('stage3_pipeline.video_frames',side_effect=lambda path: iter(frames[path])):
            for x,a,s in train_batches(records,2,3,1,3):
                for clip,ai,si in zip(x,a,s):
                    value=int(round(float((clip[0,-1,0,0]*.225+.45)*255)))
                    i,end=divmod(value,50);seen.append((i,end))
                    self.assertEqual(int(ai),end%4);self.assertEqual(int(si),end%3)
                    torch.testing.assert_close(clip,clip_tensor([frames[i][t] for t in causal_indices(end)]))
        self.assertCountEqual(seen,[(i,t) for i in range(3) for t in range(1,19,3)])

    def test_causal_padding_and_no_future(self):
        self.assertEqual(causal_indices(0).tolist(),[0]*16)
        self.assertEqual(causal_indices(2).tolist(),[0]*14+[1,2])
        self.assertEqual(causal_indices(20).tolist(),list(range(5,21)))
        with self.assertRaises(ValueError): causal_indices(-1)

    def test_streaming_dense_alignment_partial_batch_and_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            first=Path(directory)/'a.mp4';second=Path(directory)/'b.mp4'
            make_video(first,[10,40,100,180,230]);make_video(second,[240,220,200])
            frames=list(video_frames(first));model=Recorder().eval()
            result=predict_video(model,first,'a',torch.device('cpu'),3)
            self.assertEqual(result.sample_index.tolist(),list(range(5)))
            for endpoint,seen in enumerate(model.seen):
                expected=clip_tensor([frames[i] for i in causal_indices(endpoint)])
                torch.testing.assert_close(seen,expected)
            many=predict_paths(Recorder().eval(),[second,first],torch.device('cpu'),1)
            pd.testing.assert_frame_equal(result,many[many.ID=='a'].reset_index(drop=True))

    def test_stop_mask_has_zero_steer_gradient(self):
        a=torch.randn(2,4,requires_grad=True);s=torch.randn(2,3,requires_grad=True)
        loss=multitask_loss((a,s),torch.tensor([3,0]),torch.tensor([1,2]));loss.backward()
        torch.testing.assert_close(s.grad[0],torch.zeros(3))
        self.assertGreater(s.grad[1].abs().sum().item(),0)
        a=torch.randn(2,4,requires_grad=True);s=torch.randn(2,3,requires_grad=True)
        loss=multitask_loss((a,s),torch.tensor([3,3]),torch.tensor([0,2]));loss.backward()
        self.assertTrue(torch.isfinite(loss));torch.testing.assert_close(s.grad,torch.zeros_like(s))

    def test_metrics_and_ground_truth_stop_mask(self):
        metrics=classification_metrics([0,0,1,2],[0,1,1,2],STEER)
        self.assertAlmostEqual(metrics['macro_f1'],7/9)
        self.assertIsNone(classification_metrics([],[],STEER)['macro_f1'])
        with tempfile.TemporaryDirectory() as directory:
            video=Path(directory)/'v.mp4';make_video(video,[240,240,240])
            rows=[dict(ID='v',path=video,accel=np.array([3,0,3]),steer=np.array([2,0,1]))]
            metrics=evaluate(Recorder().eval(),rows,torch.device('cpu'),2)
            self.assertEqual(metrics['moving_steer']['samples'],1)
            self.assertEqual(metrics['stopped_excluded'],2)

    def test_route_leakage_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            pd.DataFrame([dict(ID='a',route='same',split='train',video='a.mp4'),dict(ID='b',route='same',split='validation',video='b.mp4')]).to_csv(root/'split_manifest.csv',index=False)
            with self.assertRaisesRegex(ValueError,'Route leakage'): load_records(root)

    def test_wrong_fps_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'wrong.mp4';make_video(path,[10,20],20)
            with self.assertRaisesRegex(ValueError,'10fps'): list(video_frames(path))


if __name__=='__main__': unittest.main()