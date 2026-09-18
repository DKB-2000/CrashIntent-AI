import json
import tempfile
import subprocess
import sys
import time
import unittest
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import patch, MagicMock
import supervise_stage1_robustness as s
from test_stage1_robustness_scoring import fixtures
from evaluate_stage1_robustness import load_resume, verify_predictions


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / 'evaluation'
        self.result = self.work / 'result'
        self.result.mkdir(parents=True)
        for key, value in [('ROOT', self.root), ('WORK', self.work), ('RESULT', self.result), ('CONTROL', self.work / 'supervisor-status.json')]:
            patcher = patch.object(s, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        s.write_json(self.result / 'status.json', dict(status='RUNNING', completed_videos=12))

    def test_windows_venv_child_tree_stops(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            time.sleep(1)
            snapshot = s.processes()
            ids = {child.pid} | {p['pid'] for p in snapshot if p['parent'] == child.pid}
            self.assertGreaterEqual(len(ids), 2, 'Exercise Windows venv redirector')
            s.stop_child(child)
            self.assertFalse(ids & {p['pid'] for p in s.processes()})
        finally:
            s.stop_child(child)

    def test_unrelated_child_prevents_stall_termination(self):
        worker = dict(pid=1, parent=0, command='evaluate', created='123')
        unrelated = dict(pid=2, parent=1, command='other job', created='124')
        with patch.object(s.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'Unexpected child'):
                s.stop_stalled([worker, unrelated], [worker])
            run.assert_not_called()

    def test_single_video_recovery_command_and_invalid_limits(self):
        child=MagicMock(pid=123, returncode=0)
        child.poll.return_value=0
        with patch.object(s.subprocess, 'Popen', return_value=child) as launch:
            self.assertIsNone(s.run_worker({'max_new_videos':1}))
            command=launch.call_args.args[0]
            self.assertEqual(command[command.index('--max-new-videos')+1], '1')
            for limit in (0,26,'1',True):
                with self.assertRaises(ValueError):
                    s.run_worker({'max_new_videos':limit})
            self.assertEqual(launch.call_count,1)

    def test_live_worker_never_duplicates(self):
        with patch.object(s, 'processes', return_value=[]), patch.object(s, 'owned_evaluators', return_value=[{'pid':123}]), patch.object(s, 'run_worker') as run:
            s.tick()
            run.assert_not_called()
            self.assertEqual(s.read_json(s.CONTROL)['status'], 'MONITORING_EXISTING')

    def test_dead_worker_becomes_failed_then_chunk_resumes(self):
        def run(control):
            self.assertEqual(s.read_json(self.result / 'status.json')['status'], 'FAILED')
            s.write_json(self.result / 'status.json', dict(status='PASS', completed_videos=357))
            return None
        with patch.object(s, 'processes', return_value=[]), patch.object(s, 'run_worker', side_effect=run), patch.object(s, 'complete') as complete:
            s.tick()
            complete.assert_called_once()
            self.assertEqual(s.read_json(s.CONTROL)['failures'], 1)

    def test_stall_stops_before_resume(self):
        order=[]
        with patch.object(s, 'processes', return_value=[]), patch.object(s, 'owned_evaluators', return_value=[{'pid':123}]), patch.object(s.time, 'time', return_value=10**12), patch.object(s, 'stop_stalled', side_effect=lambda *a: order.append('stop')), patch.object(s, 'run_worker', side_effect=lambda *a: order.append('run') or 'Worker exited: 1'):
            s.tick()
        self.assertEqual(order, ['stop', 'run'])
        self.assertEqual(s.read_json(s.CONTROL)['status'], 'NEEDS_ATTENTION')

    def test_no_progress_retries_are_bounded(self):
        control={'status':'EVALUATING'}
        self.assertTrue(s.charge_failure(control, 20, 'native'))
        self.assertFalse(s.charge_failure(control, 20, 'native'))
        self.assertEqual(control['status'], 'NEEDS_ATTENTION')

    def test_total_failures_are_bounded_even_with_progress(self):
        control={'status':'EVALUATING'}
        for i in range(4):
            self.assertTrue(s.charge_failure(control, i, 'native'))
        self.assertFalse(s.charge_failure(control, 4, 'native'))

    def test_paused_prefix_validates_and_corrupt_output_blocks(self):
        table, predictions=fixtures()
        identity=dict(scope='ALL_CONDITIONS', expected_videos=len(table), checkpoint_sha256='m', serving_sha256='c', manifest_sha256='d', precision='CPU', threshold=.5, torch_version='test')
        s.write_json(self.result / 'status.json',dict(identity,status='PAUSED'))
        p=self.result / 'predictions.jsonl'
        p.write_text(''.join(json.dumps(row)+'\n' for row in predictions[:2]))
        self.assertEqual(len(load_resume(self.result,table,identity)[1]),2)
        p.write_text(p.read_text()+'{"partial":')
        with self.assertRaises(json.JSONDecodeError):
            load_resume(self.result,table,identity)
        predictions[0]['answer']='INVALID'
        with self.assertRaisesRegex(ValueError,'threshold'):
            verify_predictions(table.iloc[:1],predictions[:1])

    def test_release_busy_does_not_turn_full_pass_into_failure(self):
        release=self.root/'artifacts/stage1-auto-release-20260910'
        release.mkdir(parents=True)
        s.write_json(release/'config.json',{})
        s.write_json(self.result/'report.json',dict(status='PASS',completed_videos=357,scope='ALL_CONDITIONS'))
        s.write_json(self.work/'watch-status.json',dict(status='FAILED'))
        with patch('advance_stage1_release.validated_inputs',return_value={'verified':True}),patch.object(s,'tick_lock',return_value=nullcontext(False)):
            s.complete({})
        self.assertEqual(s.read_json(s.CONTROL)['status'],'WAITING_FOR_RELEASE_LOCK')
        self.assertEqual(s.read_json(self.work/'watch-status.json')['status'],'PASS')

    def test_release_only_reopens_matching_failure(self):
        release=self.root/'artifacts/stage1-auto-release-20260910'
        release.mkdir(parents=True)
        s.write_json(release/'config.json',{})
        s.write_json(self.result/'report.json',dict(status='PASS',completed_videos=357,scope='ALL_CONDITIONS'))
        s.write_json(self.work/'watch-status.json',dict(status='FAILED'))
        for reason, expected in [('Stage1 robustness preparation/evaluation failed','WAITING_FOR_VALIDATION'),('Registry changed','NEEDS_ATTENTION')]:
            s.write_json(release/'status.json',dict(status='NEEDS_ATTENTION',last_error=reason))
            with patch('advance_stage1_release.validated_inputs',return_value={'verified':True}),patch.object(s,'tick_lock',return_value=nullcontext(True)):
                s.complete({})
            self.assertEqual(s.read_json(release/'status.json')['status'],expected)

if __name__=='__main__':
    unittest.main()
