"""Causal feature integrity checks before representation comparison."""
import tempfile
from pathlib import Path
import cv2,numpy as np,torch
import compare_stage3_representations as c

cv2.setNumThreads(1);torch.set_num_threads(2)
root=c.ROOT/'tests';root.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=root) as directory:
 directory=Path(directory)
 rng=np.random.default_rng(20260910)
 image=rng.integers(0,255,(96,96,3),dtype=np.uint8)
 def video(name,moving):
  path=directory/name
  writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(96,96));assert writer.isOpened()
  for i in range(32):writer.write(np.roll(image,i,axis=1) if moving else image)
  writer.release();return path
 path=video('moving.avi',True)
 dense,_=c.extract_video(path,list(range(32)))
 sparse,_=c.extract_video(path,[20,31])
 np.testing.assert_allclose(sparse,dense[[20,31]],rtol=0,atol=0)
 # Features of an earlier endpoint do not depend on requested later endpoints.
 prefix,_=c.extract_video(path,[20]);np.testing.assert_array_equal(prefix[0],dense[20])
 still,_=c.extract_video(video('still.avi',False),[15,31])
 assert np.max(np.abs(still[:,c.IMAGE_DIM:])) < 1e-4
 assert np.linalg.norm(dense[20,c.IMAGE_DIM:]) > .1
 np.testing.assert_array_equal(still[0,:c.IMAGE_DIM],still[1,:c.IMAGE_DIM])
 print('PASS: sparse/dense equality, endpoint causality, static-zero/moving-nonzero optical flow, static appearance equality')
