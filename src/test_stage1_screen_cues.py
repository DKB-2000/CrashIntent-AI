import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import prepare_stage1_screen_cues as screen


class ScreenTests(unittest.TestCase):
    def test_quality_pair_and_exact_ablation(self):
        base=screen.quality.synth._effect_parameters(random.Random(123))
        for profile in screen.PROFILES:
            original=screen.quality.matched_effects(base,profile,'ORIGINAL')
            full=screen.quality.matched_effects(base,profile,'RERECORDED')
            for cue in screen.CUES:
                values=screen.parameters(base,profile,cue)
                self.assertEqual([values[k] for k in ('noise_sigma','blur_sigma','jpeg_quality')],[original[k] for k in ('noise_sigma','blur_sigma','jpeg_quality')])
                if cue=='weak_screen':
                    self.assertAlmostEqual(values['perspective_x'],full['perspective_x']*.25)
                    self.assertAlmostEqual(values['scale']-1,(full['scale']-1)*.25)
                    self.assertAlmostEqual(values['flicker_amplitude'],full['flicker_amplitude']*.25)
                    self.assertGreater(values['moire_amplitude'],0)
                else:
                    neutral=screen.GROUPS[cue.removeprefix('no_')]
                    self.assertEqual(values,{**full,**neutral})
        with self.assertRaises(ValueError):screen.parameters(base,'clean','all_removed')

    def test_small_video_render_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'out';source=root/'source';source.mkdir()
            for folder in ('records','videos'):(out/folder).mkdir(parents=True,exist_ok=True)
            path=source/'sample.mp4';writer=screen.quality.LimitedH264Writer(path,64,48,10,24)
            rng=np.random.default_rng(24)
            for _ in range(8):writer.write(rng.integers(0,256,(48,64,3),dtype=np.uint8))
            writer.close()
            row=dict(source_id='sample',upload_group='test',source_sha256=screen.digest(path))
            plan=dict(generator_sha256=screen.code_hash());screen.write_json(out/'plan.json',plan)
            rows=screen.render(out,row,plan,source)
            self.assertEqual(len(rows),9);self.assertTrue(all(r['frames']==8 for r in rows))
            self.assertEqual(screen.render(out,row,plan,source),rows)
            broken=copy.deepcopy(rows);broken[0]['split']='val'
            with self.assertRaises(ValueError):screen.verify_rows(out,row,broken)
            broken=copy.deepcopy(rows);values=json.loads(broken[0]['effects']);values['noise_sigma']+=1;broken[0]['effects']=json.dumps(values)
            with self.assertRaises(ValueError):screen.verify_rows(out,row,broken)
            with self.assertRaises(ValueError):screen.verify_rows(out,row,rows[:-1])
            (out/rows[0]['output_path']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):screen.render(out,row,plan,source)


if __name__=='__main__':unittest.main()
