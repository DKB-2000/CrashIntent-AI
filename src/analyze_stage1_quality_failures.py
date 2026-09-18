"""Read-only paired error analysis of the completed quality pilot; no fitting."""
import csv
import hashlib
import html
import io
import json
from collections import Counter
from pathlib import Path
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'artifacts/kaggle-stage1-quality-trial-20260911'
OUT = ROOT / 'artifacts/stage1-quality-failure-analysis-20260914'
ARMS = ('baseline', 'existing_recipe', 'matched_quality')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    verified = json.loads((WORK / 'validated-recovery-20260914/result-validation.json').read_text())
    archive = WORK / 'result/stage1-quality-result.zip'
    with archive.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == verified['result_sha256']
    assert verified['status'] == 'PASS'
    with zipfile.ZipFile(archive) as z:
        raw = {a: z.read(f'result/{a}-predictions.jsonl') for a in ARMS}
    predictions = {a: [json.loads(line) for line in raw[a].splitlines()] for a in ARMS}
    assert all(len(p) == 420 for p in predictions.values())
    index = {a: {(r['source_id'], r['condition']): r for r in rows} for a, rows in predictions.items()}
    assert all(len(x) == 420 for x in index.values())
    metadata_path = ROOT / 'artifacts/stage1-robustness-20260910/generation_manifest.csv'
    with metadata_path.open(encoding='utf-8', newline='') as f:
        metadata = {(r['source_id'], r['condition']): r for r in csv.DictReader(f)}
    paired = []
    for b, e, m in zip(*(predictions[a] for a in ARMS)):
        for key in ('path', 'source_id', 'condition', 'target', 'sha256', 'upload_group'):
            assert b[key] == e[key] == m[key]
        for p in (b, e, m):
            assert abs(sum(p['slot_probabilities']) / 3 - p['probability']) < 1e-12
            assert p['answer'] == ('RERECORDED' if p['probability'] >= .5 else 'ORIGINAL')
        full = index['matched_quality'].get((b['source_id'], 'rerecorded_full'))
        row = dict(source_id=b['source_id'], upload_group=b['upload_group'], condition=b['condition'],
                   path=b['path'], target=b['target'], sha256=b['sha256'],
                   baseline=b['probability'], existing=e['probability'], matched=m['probability'],
                   delta=m['probability']-b['probability'],
                   baseline_correct=b['answer']==b['target'], matched_correct=m['answer']==m['target'],
                   lost=b['answer']==b['target'] and m['answer']!=m['target'],
                   gained=b['answer']!=b['target'] and m['answer']==m['target'],
                   positive_slots=sum(p >= .5 for p in m['slot_probabilities']),
                   matched_slots=m['slot_probabilities'], baseline_slots=b['slot_probabilities'],
                   full_matched=full['probability'] if full else None)
        if (b['source_id'], b['condition']) in metadata:
            meta=metadata[b['source_id'], b['condition']]
            assert meta['sha256'] == b['sha256']
            row['effects']=json.loads(meta['effects'])
        paired.append(row)
    lost = [r for r in paired if r['lost']]
    gained = [r for r in paired if r['gained']]
    summary = dict(status='PASS', scope='Paired development analysis, not independent generalization',
                   rows=len(paired), lost=len(lost), gained=len(gained),
                   lost_conditions=dict(Counter(r['condition'] for r in lost)),
                   lost_sources=dict(Counter(r['source_id'] for r in lost)),
                   lost_positive_slots=dict(Counter(r['positive_slots'] for r in lost)),
                   loss_below_point4=sum(r['matched'] < .4 for r in lost),
                   gained_conditions=dict(Counter(r['condition'] for r in gained)),
                   input_result_sha256=verified['result_sha256'],
                   prediction_sha256={a: sha(raw[a]) for a in ARMS},
                   source_manifest_sha256=sha(metadata_path.read_bytes()),
                   analysis_code_sha256=sha(Path(__file__).read_bytes()))
    # Verify counts against the separately validated summaries.
    for c in {r['condition'] for r in paired}:
        rows=[r for r in paired if r['condition']==c]
        for arm, field in [('baseline','baseline_correct'),('matched_quality','matched_correct')]:
            assert sum(not r[field] for r in rows)==verified['condition_summaries'][arm][c]['errors']
    (OUT/'paired.json').write_text(json.dumps(paired, indent=2),encoding='utf-8')
    (OUT/'failures.json').write_text(json.dumps(lost, indent=2),encoding='utf-8')
    with (OUT/'failures.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['source_id','condition','baseline','existing','matched','full_matched','positive_slots','matched_slots']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(lost)
    # Decode the exact hashed input bytes for all failures and paired controls.
    sections=[]; decoded={}
    with zipfile.ZipFile(WORK/'dataset/quality.bin') as inp:
        def frames(p):
            if p['path'] in decoded:return decoded[p['path']]
            data=inp.read(p['path']);assert sha(data)==p['sha256']
            temp=OUT/'decode.mp4';temp.write_bytes(data)
            cap=cv2.VideoCapture(str(temp));selected=[];count=0
            while True:
                ok,frame=cap.read()
                if not ok:break
                if count in (8,25,41):selected.append(cv2.resize(frame,(320,180)))
                count+=1
            cap.release();temp.unlink()
            assert count==50 and len(selected)==3
            decoded[p['path']]=selected
            return selected
        for i,r in enumerate(lost):
            sid=r['source_id'];control=index['matched_quality'][sid,'original_clean']
            full=index['matched_quality'][sid,'rerecorded_full']
            failure=index['matched_quality'][sid,r['condition']]
            canvas=np.zeros((660,960,3),dtype=np.uint8)
            for j,p in enumerate((control,full,failure)):
                cv2.putText(canvas,f"{sid} {p['condition']} P={p['probability']:.4f}",(8,j*220+25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                for k,frame in enumerate(frames(p)):canvas[j*220+40:j*220+220,k*320:(k+1)*320]=frame
            name=f'case-{i+1:02d}.jpg';assert cv2.imwrite(str(OUT/name),canvas)
            sections.append(f'<h2>{html.escape(sid+" / "+r["condition"])}</h2><p>baseline {r["baseline"]:.4f}; existing {r["existing"]:.4f}; matched {r["matched"]:.4f}; slots {r["matched_slots"]}</p><img width="960" src="{name}">')
    summary['decoded_unique_videos']=len(decoded)
    summary['decoded_frames']=len(decoded)*50
    (OUT/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Stage1 quality failures</title><h1>All newly lost cases</h1><p>Rows: original / full synthetic / failed variant. Columns: frames 8, 25, 41. These images are diagnostic snapshots, not all model input frames.</p>'+''.join(sections),encoding='utf-8')
    print(json.dumps(summary))
    print(json.dumps([{k:r[k] for k in ('source_id','condition','baseline','existing','matched','full_matched','positive_slots')} for r in lost]))


if __name__=='__main__':main()
