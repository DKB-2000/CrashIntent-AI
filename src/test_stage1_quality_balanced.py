import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
import prepare_stage1_quality_balanced as q

class QualityTests(unittest.TestCase):
    def test_matched_quality_and_no_screen_cues_in_original(self):
        base=q.synth._effect_parameters(random.Random(42))
        for profile in q.PROFILES:
            a=q.matched_effects(base,profile,'ORIGINAL');b=q.matched_effects(base,profile,'RERECORDED')
            for k in ('noise_sigma','blur_sigma','jpeg_quality'):self.assertEqual(a[k],b[k])
            for group,neutral in q.GROUPS.items():
                if group not in ('noise','blur','jpeg'):
                    for k,v in neutral.items():self.assertEqual(a[k],v)
            if profile=='clean':self.assertEqual((a['noise_sigma'],a['blur_sigma'],a['jpeg_quality']),(0.,0.,None))
        self.assertGreater(q.matched_effects(base,'noise','ORIGINAL')['noise_sigma'],0)

    def test_low_memory_transform_preserves_pixels_and_rng(self):
        frame=np.random.default_rng(55).integers(0,256,(75,96,3),dtype=np.uint8)
        base=q.synth._effect_parameters(random.Random(42))
        for profile in q.PROFILES:
            for label in q.LABELS:
                values=q.matched_effects(base,profile,label)
                effect=q.synth.Effects(**values)
                fields=(q.synth._perspective_matrix(96,75,effect),*q.synth._spatial_fields(96,75,effect))
                a,b=np.random.default_rng(21),np.random.default_rng(21)
                for i in range(2):
                    expected=q.transform(frame,i,10,values,fields,a)
                    actual=q.low_memory_transform(frame,i,10,values,fields,b)
                    np.testing.assert_array_equal(actual,expected)
                self.assertEqual(a.random(),b.random())

    def test_train_only_selection_rejects_holdout_and_changed_split(self):
        rows=[dict(source_id='a',upload_group='train-group',split='train',variant=0,source_sha256='x'),
              dict(source_id='b',upload_group='train-group',split='train',variant=0,source_sha256='x'),
              dict(source_id='c',upload_group='val-group',split='val',variant=0,source_sha256='x')]
        t=pd.DataFrame(rows)
        selected=q.choose_sources(t,rows,{'phone'})
        self.assertEqual(len(selected),1);self.assertNotEqual(selected[0]['source_id'],'c')
        self.assertEqual(selected,q.choose_sources(t.iloc[::-1],rows,{'phone'}))
        with self.assertRaisesRegex(ValueError,'Phone'):q.choose_sources(t,rows,{'a'})
        broken=t.copy();broken.loc[2,'split']='train'
        with self.assertRaisesRegex(ValueError,'Canonical'):q.choose_sources(broken,rows,set())

    def test_real_six_video_render_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'data_raw/ccd/videos/Crash-1500/000001.mp4';source.parent.mkdir(parents=True)
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,48))
            for i in range(3):writer.write(np.full((48,64,3),80+i*20,dtype=np.uint8))
            writer.release()
            out=root/'out';(out/'records').mkdir(parents=True)
            row=dict(source_id='000001',upload_group='train',source_sha256=q.digest(source))
            plan=dict(generator_sha256='test',crf=24);q.write_json(out/'plan.json',plan)
            with patch.object(q,'ROOT',root),patch.object(q,'code_digest',return_value='test'):
                rows=q.render_source(out,row,plan)
                self.assertEqual(len(rows),6);self.assertEqual({r['split'] for r in rows},{'train'})
                self.assertEqual(rows,q.render_source(out,row,plan))
                (out/rows[0]['output_path']).write_bytes(b'corrupt')
                with self.assertRaises(Exception):q.render_source(out,row,plan)

if __name__=='__main__':unittest.main()
