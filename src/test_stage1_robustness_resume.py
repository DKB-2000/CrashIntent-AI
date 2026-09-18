import json,tempfile,unittest
from pathlib import Path
from test_stage1_robustness_scoring import fixtures
from evaluate_stage1_robustness import load_resume

class ResumeTests(unittest.TestCase):
    def test_verified_prefix_reused_and_model_mismatch_rejected(self):
        table,predictions=fixtures()
        identity=dict(scope='ALL_CONDITIONS',expected_videos=len(table),checkpoint_sha256='model',
                      serving_sha256='code',manifest_sha256='data',precision='CPU_FP32_DIAGNOSTIC',
                      threshold=.5,torch_version='test')
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            (out/'status.json').write_text(json.dumps(dict(identity,status='FAILED')))
            (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions[:11]))
            previous,saved=load_resume(out,table,identity)
            self.assertEqual(saved,predictions[:11])
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                load_resume(out,table,dict(identity,checkpoint_sha256='other'))

    def test_missing_prefix_and_running_run_rejected(self):
        table,predictions=fixtures()
        identity=dict(scope='ALL_CONDITIONS',expected_videos=len(table),checkpoint_sha256='model',
                      serving_sha256='code',manifest_sha256='data',precision='CPU_FP32_DIAGNOSTIC',
                      threshold=.5,torch_version='test')
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            (out/'status.json').write_text(json.dumps(dict(identity,status='FAILED')))
            (out/'predictions.jsonl').write_text(json.dumps(predictions[1])+'\n')
            with self.assertRaisesRegex(ValueError,'ordered prefix'):
                load_resume(out,table,identity)
            (out/'status.json').write_text(json.dumps(dict(identity,status='RUNNING')))
            with self.assertRaisesRegex(ValueError,'stopped FAILED'):
                load_resume(out,table,identity)
if __name__=='__main__':
    unittest.main()
