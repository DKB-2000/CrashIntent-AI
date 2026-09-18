import json,tempfile,unittest,subprocess
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import patch
import recover_stage1_robustness as recover

class NativeRecoveryTests(unittest.TestCase):
    def run_case(self,progress):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);work=root/'evaluation';result=work/'result';result.mkdir(parents=True)
            release=root/'artifacts/stage1-auto-release-20260910';release.mkdir(parents=True)
            (work/'watch-status.json').write_text(json.dumps({'status':'FAILED'}))
            (result/'status.json').write_text(json.dumps({'status':'FAILED','error':'Native process exited: 3221225477','completed_videos':205}))
            (release/'status.json').write_text(json.dumps({'status':'NEEDS_ATTENTION','gpu_started':False,'dataset_uploaded':False,
                                                         'last_error':'Stage1 robustness preparation/evaluation failed'}))
            calls=[]
            def fake_run(command,**kwargs):
                calls.append(command)
                if len(calls)==1:
                    (result/'status.json').write_text(json.dumps({'status':'RUNNING','completed_videos':206 if progress else 205}))
                    return subprocess.CompletedProcess(command,3221225477)
                (result/'report.json').write_text(json.dumps({'status':'PASS','completed_videos':357}))
                return subprocess.CompletedProcess(command,0)
            with patch.object(recover,'ROOT',root),patch.object(recover,'work',work),patch.object(recover,'stress',root),                 patch.object(recover,'state_path',work/'watch-status.json'),patch.object(recover,'tick_lock',return_value=nullcontext(True)),                 patch.object(recover.subprocess,'run',side_effect=fake_run),patch.object(recover.ctypes.windll.kernel32,'SetThreadExecutionState',return_value=1):
                if progress:
                    recover.main()
                    self.assertEqual(len(calls),2)
                    self.assertEqual(json.loads((release/'status.json').read_text())['status'],'WAITING_FOR_VALIDATION')
                else:
                    with self.assertRaises(subprocess.CalledProcessError):
                        recover.main()
                    self.assertEqual(len(calls),1)
                    self.assertEqual(json.loads((result/'status.json').read_text())['status'],'FAILED')
                    self.assertEqual(json.loads((release/'status.json').read_text())['status'],'NEEDS_ATTENTION')
    def test_progress_retries_and_then_reopens_release(self):
        self.run_case(True)
    def test_repeated_same_point_crash_stops(self):
        self.run_case(False)
if __name__=='__main__':
    unittest.main()
