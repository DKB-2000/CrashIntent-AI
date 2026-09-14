"""Wait locally for prepared inputs, audit hashes, build review UI, and freeze evaluation contract."""
import csv
import json
import msvcrt
import os
import subprocess
import sys
import time
from prepare_stage2_final_validation import ROOT, OUT, DATA, sha, save
from score_stage2_final_validation import read_inputs, read_csv, label_gate


def finalize():
    with (OUT/'readiness.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            if (OUT/'evaluation-contract.json').exists():
                raise RuntimeError('Readiness already finalized; inspect existing files')
            started = time.monotonic()
            save(OUT/'readiness-status.json', dict(status='WAITING_FOR_INPUTS', pid=os.getpid(), timeout_seconds=3600, retries=0))
            while not (OUT/'input-validation.json').exists():
                watch = OUT/'watch-status.json'
                if watch.exists() and json.loads(watch.read_text())['status'] == 'FAILED':
                    raise RuntimeError('Input conversion failed; see watch-status.json')
                if time.monotonic()-started > 3600:
                    raise TimeoutError('Input readiness wait exceeded one hour')
                time.sleep(15)
            videos, times = read_inputs()
            initial = read_csv(DATA/'review-decisions.csv')
            gate = label_gate(initial, times)
            assert gate['status'] == 'WAITING_FOR_HUMAN_LABELS' and len(gate['pending']) == 50
            blocked_output = OUT/'must-not-create-unlabeled-predictions'
            blocked = subprocess.run([sys.executable, str(ROOT/'src/predict_stage2_final_validation.py'), '--output', str(blocked_output)],
                                     capture_output=True, text=True, encoding='utf-8', timeout=60)
            assert blocked.returncode != 0 and 'Complete human dispositions and labels before inference' in blocked.stderr
            assert not blocked_output.exists()
            (OUT/'unlabeled-gate-check.log').write_text(blocked.stderr, encoding='utf-8')
            save(OUT/'readiness-status.json', dict(status='VERIFYING_INPUTS', pid=os.getpid()))
            files, amount = 0, 0
            for sid, video in videos.items():
                assert sha(ROOT/video['video_file']) == video['video_sha256']
                rows = read_csv(DATA/video['frame_map'])
                assert len(list((DATA/'images'/sid).glob('*.jpg'))) == len(rows)
                for row in rows:
                    if time.monotonic()-started > 5400:
                        raise TimeoutError('Readiness wait and verification exceeded 90 minutes')
                    path = DATA/row['file']
                    assert path.stat().st_size == int(row['bytes']) and sha(path) == row['sha256']
                    files += 1
                    amount += int(row['bytes'])
            frozen = json.loads((OUT/'frozen-models.json').read_text())
            for entry in frozen.values():
                assert sha(ROOT/entry['file']) == entry['sha256']
            assert json.loads((OUT/'model-smoke.json').read_text())['status'] == 'PASS'
            assert json.loads((OUT/'test-status.json').read_text())['status'] == 'PASS'
            boot = dict(manifest_sha256=sha(DATA/'manifest.json'), rows=initial,
                        videos=[dict(ID=sid, times=[float(t) for t in times[sid]], video_file=v['video_file']) for sid, v in videos.items()])
            template = (ROOT/'src/stage2_final_review.html').read_text(encoding='utf-8')
            (DATA/'review.html').write_text(template.replace('__BOOTSTRAP__', json.dumps(boot, ensure_ascii=False).replace('</', '<\\/')), encoding='utf-8')
            code = {rel: sha(ROOT/rel) for rel in ['src/score_stage2_final_validation.py', 'src/predict_stage2_final_validation.py',
                                                 'src/prepare_stage2_final_validation.py', 'src/stage2_final_review.html']}
            save(OUT/'evaluation-contract.json', dict(status='FROZEN_BEFORE_LABELS', manifest_sha256=sha(DATA/'manifest.json'),
                                                     frozen_models_sha256=sha(OUT/'frozen-models.json'), code_sha256=code,
                                                     model_names=['incumbent', 'ccd_pretrained'], inference_precision='CPU FP32',
                                                     tolerance_seconds='3/10', reviewed_candidates_required=50, target_eligible_minimum=30,
                                                     missing_prediction='Count wrong on each missing item', invalid_prediction='Count wrong on each invalid item',
                                                     ground_truth='Human ACCEPT rows only; all50 dispositions required; no reference_event or AI screen conversion',
                                                     source_metric='Known recording sources only; UNKNOWN excluded from grouping with coverage reported',
                                                     selection='No training or hyperparameter tuning on this pool; no automatic promotion'))
            save(OUT/'label-readiness.json', dict(status='WAITING_FOR_HUMAN_LABELS', pending=50, accepted=0, excluded=0, metrics=None))
            save(OUT/'readiness-validation.json', dict(status='PASS', source_videos=50, frame_files=files, jpeg_bytes=amount,
                                                     all_frame_sha256='PASS', frame_maps='PASS', models='PASS', model_smoke='PASS',
                                                     blank_labels='PASS_PENDING_50', human_labels_created=0, predictions_created=0,
                                                     actual_prediction_command_blocked_before_output=True,
                                                     review_html_sha256=sha(DATA/'review.html'), scope='Preparation only; browser interaction not manually tested'))
            save(OUT/'readiness-status.json', dict(status='READY_FOR_HUMAN_REVIEW', completed=50, total=50, frames=files,
                                                  human_labels=0, predictions=0, pid=os.getpid()))
        except Exception as exc:
            save(OUT/'readiness-status.json', dict(status='FAILED', error=repr(exc), pid=os.getpid()))
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == '__main__':
    finalize()
