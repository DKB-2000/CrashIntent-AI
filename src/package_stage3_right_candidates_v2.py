"""Package visually screened candidates with empty human review annotations."""
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import pandas as pd
from review_stage3_human import save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/stage3-right-candidates-20260914'
DATA = ROOT / 'data/stage3-right-review-v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cv2.setNumThreads(1)
    plan = json.loads((OUT / 'plan.json').read_text(encoding='utf-8'))
    for path, expected in plan['inputs'].items():
        assert sha(ROOT / path) == expected
    shortlist = json.loads((OUT / 'shortlist.json').read_text())
    choices = json.loads((OUT / 'visual-screening.json').read_text(encoding='utf-8'))
    assert {r['source_id'] for r in choices} == {r['source_id'] for r in shortlist}
    selected = [r for r in choices if r['decision'] == 'candidate']
    assert selected
    DATA.mkdir(exist_ok=True)
    labels_path = DATA / 'annotations.json'
    assert not labels_path.exists(), 'Refuse overwrite existing human annotations'
    (OUT / 'videos').mkdir(exist_ok=True)
    state = dict(schema=1, scope='Unreviewed source-disjoint right-turn candidates; no ground truth', videos={})
    manifest = []
    sources = set()
    for i, choice in enumerate(selected, 1):
        row = next(r for r in shortlist if r['source_id'] == choice['source_id'])
        assert row['youtube_id'] not in plan['excluded_youtube_ids']
        assert row['youtube_id'] not in sources
        sources.add(row['youtube_id'])
        sid = f'RIGHT2_{i:02}'
        source = ROOT / 'data_raw/ccd/videos/Crash-1500' / (row['source_id']+'.mp4')
        assert sha(source) == row['sha256']
        dest = OUT / 'videos' / (sid+'.mp4')
        shutil.copy2(source, dest)
        assert sha(dest) == row['sha256']
        cap = cv2.VideoCapture(str(dest))
        assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS)-10) < .01
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        assert count == 50
        state['videos'][sid] = dict(path=str(dest.resolve()), sha256=row['sha256'], frames=50,
                                    fps=10.0, assumed_10hz=False, segments=[])
        manifest.append(dict(ID=sid, source_id=row['source_id'], youtube_id=row['youtube_id'],
                             sha256=row['sha256'], frames=50, fps=10, reason=choice['reason']))
    save(state, labels_path)
    labels = pd.read_csv(DATA / 'labels_review.csv')
    assert len(labels) == len(manifest)*50 and labels.reviewed.sum() == labels.valid_steer.sum() == 0
    assert set(labels.steer_label) == set(labels.motion_status) == {'UNKNOWN'}
    sources_doc = dict(scope='Source-disjoint from recorded exclusions; candidate selection biased, not independent generalization or confirmed right turns',
                       method='Early horizontal optical flow ranking; one per source; assistant visual screening. User interval review pending.',
                       plan=str((OUT/'plan.json').relative_to(ROOT)), videos=manifest)
    (DATA / 'sources.json').write_text(json.dumps(sources_doc, indent=2, ensure_ascii=False), encoding='utf-8')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>새 Stage3 검수 후보</title>
<style>body{font:18px sans-serif;max-width:1100px;margin:30px auto;background:#161b22;color:#eee}video{width:100%}button,select{font:inherit;padding:8px;margin:6px}p{line-height:1.6}</style>
<h1>자동차 전방 시점 검수 후보</h1><p>아직 사람 정답이 없는 후보입니다. 자차의 진행 방향을 전후로 확인하고, 충격·가림·판단 불가는 UNKNOWN으로 남깁니다. 이 페이지는 시청용이며 판정을 저장하지 않습니다.</p>
<select id="select"></select><button id="prev">이전 영상</button><button id="next">다음 영상</button><video id="video" controls preload="metadata"></video>
<button id="start">처음부터 재생</button><button id="back">이전 0.1초</button><button id="forward">다음 0.1초</button><select id="speed"><option value="1">정상 속도</option><option value="0.5">절반 속도</option></select><p id="time"></p>
<p>실제 라벨 저장은 프로젝트의 scripts/Start-Stage3RightReviewV2.ps1로 검수 도구를 실행하세요. 기존 CASE_01과는 별도 세트입니다.</p>
<script>const ids=PAYLOAD,s=document.querySelector('#select'),v=document.querySelector('#video');ids.forEach(id=>{const o=document.createElement('option');o.value=id;o.textContent=id;s.appendChild(o)});function show(){v.src='videos/'+s.value+'.mp4';v.load()}s.onchange=show;document.querySelector('#prev').onclick=()=>{s.selectedIndex=Math.max(0,s.selectedIndex-1);show()};document.querySelector('#next').onclick=()=>{s.selectedIndex=Math.min(ids.length-1,s.selectedIndex+1);show()};document.querySelector('#start').onclick=()=>{v.currentTime=0;v.play()};document.querySelector('#back').onclick=()=>{v.pause();v.currentTime=Math.max(0,v.currentTime-.1)};document.querySelector('#forward').onclick=()=>{v.pause();v.currentTime=Math.min(4.9,v.currentTime+.1)};document.querySelector('#speed').onchange=e=>{v.playbackRate=Number(e.target.value)};v.ontimeupdate=()=>{document.querySelector('#time').textContent=v.currentTime.toFixed(1)+'秒 / 5.0秒'};show();</script></html>'''.replace('PAYLOAD', json.dumps([r['ID'] for r in manifest])).replace('秒', '초')
    (OUT / 'review.html').write_text(page, encoding='utf-8')
    result = dict(status='READY_FOR_HUMAN_REVIEW', videos=len(manifest), frames=len(labels),
                  sources=len(sources), human_reviewed_frames=0, confirmed_right_frames=0,
                  checks='Source exclusions, distinct sources, original/copied SHA256, full decode/FPS, empty labels PASS',
                  files={str(p.relative_to(ROOT)):sha(p) for p in DATA.iterdir() if p.is_file()})
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (OUT / 'status.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
