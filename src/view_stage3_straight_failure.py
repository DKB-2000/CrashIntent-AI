"""Export a fixed six-frame diagnostic contact sheet, no labels changed."""
from pathlib import Path
import cv2
import numpy as np
P=Path(__file__).resolve().parents[1]
c=cv2.VideoCapture(str(P/'artifacts/stage3-civic-holdout-score-20260914/CIVIC_HOLDOUT_001.avi'))
tiles=[]
for t in [486,496,504,513,522,532]:
    c.set(cv2.CAP_PROP_POS_FRAMES,t)
    ok,f=c.read();assert ok
    f=cv2.resize(f,(480,270))
    cv2.putText(f,f'{t/10:.1f}s',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
    tiles.append(f)
c.release()
assert cv2.imwrite(str(P/'artifacts/stage3-low-speed-diagnostic-20260914/straight-001.jpg'),np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))
