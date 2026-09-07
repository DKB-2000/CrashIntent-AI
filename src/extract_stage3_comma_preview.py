"""Extract one explicit comma2k19 segment safely for visual calibration."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def extract(archive, segment, output):
    prefix = segment.rstrip('/') + '/'
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    required = {'video.hevc', 'global_pose/frame_times',
                'processed_log/CAN/steering_angle/t', 'processed_log/CAN/steering_angle/value'}
    with zipfile.ZipFile(archive) as source:
        members = {name[len(prefix):]: name for name in source.namelist()
                   if name.startswith(prefix) and not name.endswith('/')}
        speed = 'processed_log/CAN/speed/' if 'processed_log/CAN/speed/t' in members else 'processed_log/CAN/car_speed/'
        required.update({speed+'t', speed+'value'})
        if required - members.keys():
            raise ValueError(f'Missing files: {sorted(required - members.keys())}')
        output.mkdir(parents=True)
        for relative in sorted(required):
            # Fixed allow-list only; preserve source route in manifest, not filename.
            target_relative = relative.replace('CAN/car_speed/', 'CAN/speed/')
            target = output.joinpath(*PurePosixPath(target_relative).parts).resolve()
            if not target.is_relative_to(output):
                raise ValueError('Archive path escapes destination')
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(members[relative]) as reader, target.open('xb') as writer:
                shutil.copyfileobj(reader, writer)
    (output / 'source_manifest.json').write_text(json.dumps(
        dict(archive=str(archive.resolve()), segment=segment,
             route=segment.rstrip('/').rsplit('/', 1)[0], files=sorted(required)), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--segment', required=True, help='Exact ZIP segment prefix, from segments.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    extract(args.archive, args.segment, args.output_dir)
