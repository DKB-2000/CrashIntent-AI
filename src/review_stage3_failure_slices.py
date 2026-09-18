"""Build read-only frame/crop evidence for the frozen human comparison."""
import csv
import hashlib
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import cv2
import numpy as np
import torch
import stage3_motion_inference as old
import stage3_short_motion_inference as short

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'artifacts/stage3-human-comparison-20260914'
OUT = ROOT / 'artifacts/stage3-failure-review-20260914'
CLASSES = ['LEFT', 'STRAIGHT', 'RIGHT']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads((INPUT / 'report.json').read_text(encoding='utf-8'))
    audit = json.loads((INPUT / 'independent-validation.json').read_text())
    for name, expected in report['inputs'].items():
        assert digest(ROOT / name) == expected
    for name, expected in audit['artifact_sha256'].items():
        assert digest(INPUT / name) == expected
    videos, bins, frame_rows = [], [], []
    for item in audit['per_video']:
        sid, group = item['ID'], item['group']
        source = next(s for s in report['sources'] if s['ID'] == sid and s['group'] == group)
        path = ROOT / source['path']
        assert digest(path) == source['sha256']
        truth = next(iter(item['truth']))
        assert item['truth'][truth] == 50  # Current fixed whole-clip review only.
        predictions, confidence = {}, {}
        for name, module in [('old', old), ('short', short)]:
            cached = np.load(INPUT / f'{group}-{sid}-{name}.npz')
            features = module.s3_motion_features(path)
            np.testing.assert_array_equal(features, cached['features'])
            model, mean, std = module.s3_motion_load(INPUT / f'{name}.pt', torch.device('cpu'))
            accel, steer = module.s3_motion_logits(model, features, mean, std, torch.device('cpu'))
            np.testing.assert_array_equal(accel, cached['accel'])
            np.testing.assert_array_equal(steer, cached['steer'])
            predictions[name] = [CLASSES[i] for i in steer.argmax(1)]
            probabilities = np.exp(steer - steer.max(1, keepdims=True))
            probabilities /= probabilities.sum(1, keepdims=True)
            confidence[name] = probabilities.max(1).tolist()
        cap = cv2.VideoCapture(str(path))
        assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS) - 10) < .01
        frames = []
        folder = OUT / sid
        folder.mkdir(exist_ok=True)
        for i in range(50):
            ok, frame = cap.read()
            assert ok
            h, w = frame.shape[:2]
            assert (w, h) == (1280, 720)
            scale = 256 / min(h, w)
            nw, nh = max(256, round(w * scale)), max(256, round(h * scale))
            x, y = (nw - 224) // 2, (nh - 224) // 2
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            crop = cv2.resize(resized[y:y + 224, x:x + 224], (96, 96), interpolation=cv2.INTER_AREA)
            shown = frame.copy()
            cv2.rectangle(shown, (round(x * w / nw), round(y * h / nh)),
                          (round((x + 224) * w / nw), round((y + 224) * h / nh)), (0, 255, 255), 3)
            shown = cv2.resize(shown, (640, 360), interpolation=cv2.INTER_AREA)
            assert cv2.imwrite(str(folder / f'{i:02}.jpg'), shown)
            assert cv2.imwrite(str(folder / f'{i:02}-crop.png'), crop)
            tile = np.zeros((408, 640, 3), dtype=np.uint8)
            tile[:360] = shown
            label = f'f{i:02} {i / 10:.1f}s truth={truth} old={predictions["old"][i]} short={predictions["short"][i]}'
            cv2.putText(tile, label, (8, 389), cv2.FONT_HERSHEY_SIMPLEX, .49, (255, 255, 255), 1)
            frames.append(tile)
            frame_rows.append(dict(group=group, ID=sid, sample_index=i, truth=truth,
                                   old=predictions['old'][i], short=predictions['short'][i],
                                   old_confidence=confidence['old'][i], short_confidence=confidence['short'][i]))
        ok, _ = cap.read()
        assert not ok
        cap.release()
        selected = [0, 5, 10, 15, 20, 25, 30, 35, 40, 49]
        sheet = np.vstack([np.hstack([frames[i], frames[j]]) for i, j in zip(selected[::2], selected[1::2])])
        assert cv2.imwrite(str(OUT / f'{sid}-contact.jpg'), sheet)
        for start, end in [(0, 14), (15, 29), (30, 49)]:
            bins.append(dict(group=group, ID=sid, start=start, end=end, truth=truth,
                             count=end-start+1, **{name: sum(p == truth for p in pred[start:end+1])
                                                  for name, pred in predictions.items()}))
        videos.append(dict(ID=sid, group=group, truth=truth, predictions=predictions,
                           source=source['path'], source_sha256=source['sha256']))
        print(sid, 'source/features/logits exact PASS', flush=True)
    with (OUT / 'frames.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)
    content = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Stage3 실패 구간 검토</title>
<style>body{font:17px sans-serif;background:#161b22;color:#eee;max-width:1080px;margin:30px auto}button,select,input{font:inherit;margin:8px}img{max-width:100%}.views{display:flex;gap:16px;align-items:center}#crop{width:288px;image-rendering:pixelated}td,th{padding:5px 14px}a{color:#9bd}</style>
<h1>Stage3 실패 구간 검토</h1><p>사람 라벨은 고정된 기존 판정입니다. 노란 상자는 모델 입력 범위, 오른쪽은 실제 96×96 입력입니다. 이 페이지는 라벨을 저장하거나 변경하지 않습니다.</p>
<select id="video"></select><button id="play">재생 / 정지</button><button id="prev">이전 프레임</button><button id="next">다음 프레임</button>
<p id="info"></p><input style="width:95%" id="frame" type="range" min="0" max="49" value="0"><div class="views"><img id="full" width="640"><img id="crop" alt="모델 입력"></div>
<p id="source"></p><p>←/→: 1프레임 이동. 0~14는 15프레임 이력 축적 구간입니다. 30~49는 고정한 마지막 2초 구간이며 충돌 시점 정답을 의미하지 않습니다. 모델 예측은 화면 움직임 관찰의 보조 자료입니다.</p>
<script>const videos=PAYLOAD;const select=document.querySelector('#video'),slider=document.querySelector('#frame');
for(const v of videos){const o=document.createElement('option');o.textContent=v.ID+' / '+v.group;o.value=v.ID;select.appendChild(o)}
function show(){const v=videos.find(v=>v.ID===select.value),i=Number(slider.value),f=String(i).padStart(2,'0');document.querySelector('#full').src=v.ID+'/'+f+'.jpg';document.querySelector('#crop').src=v.ID+'/'+f+'-crop.png';document.querySelector('#info').textContent=`${v.ID} | ${i}/49 (${(i/10).toFixed(1)}초) | 사람: ${v.truth} | 기존: ${v.predictions.old[i]} | short: ${v.predictions.short[i]}`;document.querySelector('#source').textContent=v.source}
function move(d){slider.value=Math.max(0,Math.min(49,Number(slider.value)+d));show()}
select.onchange=()=>{slider.value=0;show()};slider.oninput=show;let timer=null;document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null}else{timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%50;show()},100)}};
document.querySelector('#prev').onclick=()=>move(-1);document.querySelector('#next').onclick=()=>move(1);document.onkeydown=e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();move(e.key==='ArrowLeft'?-1:1)}};show();</script></html>'''.replace('PAYLOAD', json.dumps(videos))
    (OUT / 'review.html').write_text(content, encoding='utf-8')
    for name, expected in report['inputs'].items():
        assert digest(ROOT / name) == expected
    result = dict(status='PASS', scope='Read-only failure evidence, no human relabeling', videos=videos,
                  slices=bins, verification='7 source hashes, 350 full frames, 14 feature/logit exact replays; human input hashes unchanged',
                  crop_geometry={'source_size': [1280, 720], 'resized_size': [455, 256],
                                 'crop_xywh': [115, 16, 224, 224], 'final_size': [96, 96]},
                  inputs=report['inputs'])
    (OUT / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(bins, indent=2))


if __name__ == '__main__':
    main()
