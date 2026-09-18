"""Exercise auto-release gates, stage isolation, registry conflicts and retry safety."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import advance_stage1_release as auto
import daily_submission as release


def make_candidate(path, stage1=b'old', stage2=b'unchanged'):
    code='\n'.join(f'def predict_stage{i}(data_dir, model_dir): pass' for i in (1,2,3))
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('inference.py',code)
        z.writestr('requirements.txt','torch==2.8.0\n')
        z.writestr('model/stage1/best.pt',stage1)
        z.writestr('model/stage2/best.pt',stage2)
        z.writestr('model/stage2/resnet18-f37072fd.pth',b'unchanged')
        z.writestr('model/stage3/best.pt',b'unchanged')


def make_evidence(path, candidate):
    report=dict(status='PASS',python_internet_socket_blocked=True,
                results={f'stage{i}':{'seconds':1} for i in (1,2,3)})
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('integration.json',json.dumps(report))
        z.writestr('candidate-assets.json',json.dumps({'candidate.bin':release.sha(candidate)}))
        z.writestr('install.json',json.dumps({'exit_code':0,'seconds':1}))


class AutoReleaseTests(unittest.TestCase):
    def test_only_stage1_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            make_candidate(root/'base.zip')
            make_candidate(root/'new.zip',stage1=b'new')
            auto.verify_candidate(root/'base.zip',root/'new.zip',hashlib.sha256(b'new').hexdigest())
            make_candidate(root/'bad.zip',stage1=b'new',stage2=b'changed')
            with self.assertRaisesRegex(ValueError,'non-Stage1'):
                auto.verify_candidate(root/'base.zip',root/'bad.zip',hashlib.sha256(b'new').hexdigest())

    def test_pending_validation_never_starts_cloud_work(self):
        state={'status':'WAITING_FOR_VALIDATION'}
        with patch.object(auto,'validated_inputs',return_value=None), patch.object(auto,'log_command') as run:
            auto.advance({'kernel_id':'private/test'},state,Path('.'),lambda **v:state.update(v),run)
            run.assert_not_called()
            self.assertEqual(state['status'],'WAITING_FOR_VALIDATION')

    def test_uncertain_mutations_are_not_repeated(self):
        for phase in ['UPLOAD_STARTED','PUSH_STARTED']:
            with patch.object(auto,'log_command') as run:
                with self.assertRaisesRegex(RuntimeError,'uncertain'):
                    auto.advance({'kernel_id':'private/test'}, {'status':phase}, Path('.'), lambda **v:None, run)
                run.assert_not_called()

    def test_changed_registry_blocks_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            registry=root/'selected.json'
            registry.write_text(json.dumps({'sha256':'other-team-session'}))
            make_candidate(root/'candidate.zip',stage1=b'new')
            make_evidence(root/'evidence.zip',root/'candidate.zip')
            before=registry.read_bytes()
            with patch.object(release,'ROOT',root),patch.object(release,'REGISTRY',registry):
                with self.assertRaisesRegex(ValueError,'registry changed'):
                    release.register_and_publish(root/'candidate.zip',root/'evidence.zip','test','old-registry-hash')
            self.assertEqual(registry.read_bytes(),before)
            self.assertFalse((root/'artifacts/daily-submissions').exists())

    def test_register_publish_and_restart_are_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            registry=root/'selected.json'
            registry.write_text(json.dumps({'sha256':'prior'}))
            original_hash=release.sha(registry)
            make_candidate(root/'candidate.zip',stage1=b'new')
            make_evidence(root/'evidence.zip',root/'candidate.zip')
            with patch.object(release,'ROOT',root),patch.object(release,'REGISTRY',registry):
                first=release.register_and_publish(root/'candidate.zip',root/'evidence.zip','test',original_hash)
                second=release.register_and_publish(root/'candidate.zip',root/'evidence.zip','test',original_hash)
            self.assertEqual(first['zip'],second['zip'])
            self.assertEqual(release.sha(root/first['zip']),release.sha(root/'candidate.zip'))
            self.assertFalse(first['submitted'])

    def test_smoke_validation_cannot_promote(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'train/validated').mkdir(parents=True)
            (root/'stress/full-model-evaluation/result').mkdir(parents=True)
            (root/'train/validated/result-validation.json').write_text(json.dumps({'status':'PASS'}))
            (root/'stress/full-model-evaluation/result/report.json').write_text(
                json.dumps({'status':'PASS','scope':'SUBSET_SMOKE','completed_videos':17}))
            with patch.object(auto,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'Full robustness'):
                    auto.validated_inputs({'training_dir':'train','robustness_dir':'stress','expected_stress_videos':357})

if __name__=='__main__':
    unittest.main()
