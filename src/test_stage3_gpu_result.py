"""Synthetic rejection/acceptance tests; these are NOT actual GPU evidence."""
import json,tempfile,unittest,zipfile
from pathlib import Path
from validate_stage3_gpu_result import validate


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.provenance={'kind':'GPU_SMOKE_ONLY','file_sha256':{'src/stage3_pipeline.py':'synthetic-fixture'}}
        self.report=dict(status='PASS',device='cuda',public_cuda_entrypoint_tested=True,checkpoint_reload=True,architecture='mvit_v2_s',training_steps=1,loss=1.2,seconds=1.,prediction_rows=4)
        self.files={'environment.json':json.dumps(dict(cuda_available=True,torch_cuda='fixture',gpu_names=['SYNTHETIC_TEST_GPU'],torch='fixture',torchvision='fixture')),'execution.json':json.dumps({stage:{'exit_code':0} for stage in ('tests','audit','smoke')}),'bundle.json':json.dumps(self.provenance),'tests.log':'synthetic','audit.log':'synthetic','smoke.log':'synthetic','smoke-run/predictions.csv':'ID,sample_index,accel_label,steer_label\n'+''.join(f'SMOKE_S3,{i},CONSTANT,STRAIGHT\n' for i in range(4))}
        with zipfile.ZipFile(self.root/'bundle.zip','w') as z:z.writestr('bundle.json',json.dumps(self.provenance))

    def run_result(self):
        self.files['smoke-run/smoke_report.json']=json.dumps(self.report)
        with zipfile.ZipFile(self.root/'result.zip','w') as z:
            for name,value in self.files.items():z.writestr(name,value)
        return validate(self.root/'result.zip',self.root/'bundle.zip')

    def test_complete_fixture_accepted(self):self.assertEqual(self.run_result()['status'],'PASS')
    def test_cpu_rejected(self):
        self.report['device']='cpu'
        with self.assertRaisesRegex(ValueError,'CUDA'):self.run_result()
    def test_untested_public_function_rejected(self):
        self.report['public_cuda_entrypoint_tested']=False
        with self.assertRaisesRegex(ValueError,'Public'):self.run_result()
    def test_missing_frame_rejected(self):
        self.files['smoke-run/predictions.csv']=self.files['smoke-run/predictions.csv'].replace('SMOKE_S3,2,CONSTANT,STRAIGHT\n','')
        with self.assertRaisesRegex(ValueError,'four'):self.run_result()
    def test_bundle_mismatch_rejected(self):
        self.files['bundle.json']='{}'
        with self.assertRaisesRegex(ValueError,'bundle'):self.run_result()
    def test_failed_test_command_rejected(self):
        self.files['execution.json']=json.dumps({'tests':{'exit_code':1}})
        with self.assertRaisesRegex(ValueError,'tests'):self.run_result()
    def test_invalid_class_rejected(self):
        self.files['smoke-run/predictions.csv']=self.files['smoke-run/predictions.csv'].replace('STRAIGHT','UNKNOWN')
        with self.assertRaisesRegex(ValueError,'steering'):self.run_result()


if __name__=='__main__':unittest.main()