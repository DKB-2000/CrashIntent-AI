"""Read-only source inventory; no predictions, downloads, or label changes."""
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parents[1]
OUT = P / 'artifacts/stage3-source-inventory-20260914'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def canonical(value):
    return '/'.join(value.replace('\\', '/').split('/')[-2:])


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    manifest_path = P/'artifacts/stage3-comma-chunk1/manifest.json'
    split_path = P/'artifacts/stage3-comma-chunk1-calibrated-v1/split_manifest.csv'
    sample_path = P/'artifacts/stage3-route-cv-20260911/samples.csv'
    manifests = json.loads(manifest_path.read_text(encoding='utf-8'))
    splits = {r['ID']: r for r in read_csv(split_path)}
    samples = read_csv(sample_path)
    sampled = {r['ID'] for r in samples}
    train_routes = {r['route'].split('/')[-1] for r in samples}
    archive_path = P/'data_raw/comma2k19/Chunk_1.http.zip'
    required = ['video.hevc', 'global_pose/frame_times', 'processed_log/CAN/speed/t',
                'processed_log/CAN/speed/value', 'processed_log/CAN/steering_angle/t',
                'processed_log/CAN/steering_angle/value']
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        segments = {n[:-len('/video.hevc')] for n in names if n.endswith('/video.hevc')}
        converted = {r['source_segment'] for r in manifests}
        assert converted <= segments
        excluded = sorted(segments-converted)
        assert excluded == ['Chunk_1/b0c9d2329ad1606b|2018-07-29--12-02-42/31']
        assert all(s.rsplit('/',1)[0] in {r['source_route'] for r in manifests} for s in excluded)
        assert len(splits) == len(manifests) == len({r['ID'] for r in manifests})
        for m in manifests:
            sid, segment, route = m['ID'], m['source_segment'], m['source_route']
            assert splits[sid]['route'] == route
            assert all(segment+'/'+name in names for name in required)
            local = P/'artifacts/stage3-comma-chunk1/segments'/sid
            assert all((local/name).is_file() for name in required)
            video = (split_path.parent/splits[sid]['video']).resolve()
            assert video.is_relative_to(P) and video.is_file()
            rows.append(dict(ID=sid, segment=segment, route=route.split('/')[-1],
                             device_id=route.split('/')[-1].split('|')[0], split=splits[sid]['split'],
                             sampled_for_training=sid in sampled, samples=m['samples'],
                             video=str(video.relative_to(P)),
                             eligibility='USED_TRAIN_ROUTE' if route.split('/')[-1] in train_routes else 'USED_DEVELOPMENT_ROUTE'))
        example_path = P/'data_raw/comma2k19/example-local/source_manifest.json'
        example = json.loads(example_path.read_text(encoding='utf-8'))
        matches = [r for r in rows if canonical(r['segment']) == canonical(example['segment'])]
        assert len(matches) == 1
        match = matches[0]
        hashes = {}
        for name in required:
            a = P/'data_raw/comma2k19/example-local'/name
            b = P/'artifacts/stage3-comma-chunk1/segments'/match['ID']/name
            ha, hb = digest(a), digest(b)
            assert ha == hb, name
            hashes[name] = ha
    raw_roots = sorted(p.name for p in (P/'data_raw').iterdir() if p.is_dir())
    raw_files = [dict(path=str(p.relative_to(P)), bytes=p.stat().st_size)
                 for p in (P/'data_raw/comma2k19').iterdir() if p.is_file()]
    report = dict(status='COMPLETE', scope='Local data_raw and recorded Stage3 provenance; ZIP central-directory inventory, required-file presence, example content SHA256. Not full video decoding or geographic trajectory deduplication.',
                  segments=len(rows), routes=len({r['route'] for r in rows}),
                  device_ids=sorted({r['device_id'] for r in rows}),
                  vehicle_identity='One device ID; vehicle model from project documentation only, not independently verified VIN.',
                  split_segments=dict(Counter(r['split'] for r in rows)),
                  split_routes={s:len({r['route'] for r in rows if r['split']==s}) for s in {r['split'] for r in rows}},
                  sampled_training_videos=len(sampled), train_clips=len(samples),
                  archive_segments=len(segments), unconverted_archive_segments=len(excluded),
                  excluded_segments=excluded, excluded_reason='Previously rejected: CAN out-of-range sample fraction >1%; same existing route. See DATA_SOURCES.md and original sensor audit.',
                  unused_routes=0, eligible_new_sensor_segments=0,
                  example_duplicate=dict(ID=match['ID'],segment=example['segment'],split=match['split'],identical_files_sha256=hashes),
                  raw_data_roots=raw_roots, comma_raw_files=raw_files,
                  other_sources='CCD/human-review and Nexar/DoTA do not supply the synchronized ego steering/speed signals used here; pending review is not a sensor label.',
                  next_step='Acquire a disjoint comma2k19 device/route pilot with synchronized video/speed/steering/timestamps; validate sign, offset, timing before freezing an evaluation-only set. ZOD availability not verified.',
                  input_sha256={str(p.relative_to(P)):digest(p) for p in [Path(__file__),manifest_path,split_path,sample_path,example_path]})
    with (OUT/'segments.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert len(read_csv(OUT/'segments.csv')) == len(rows)
    report['segments_csv_sha256']=digest(OUT/'segments.csv')
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
