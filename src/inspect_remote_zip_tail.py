from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


EOCD = b"PK\x05\x06"
CD = b"PK\x01\x02"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tail", type=Path)
    parser.add_argument("--total-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = args.tail.read_bytes()
    base = args.total_size - len(data)
    eocd_at = data.rfind(EOCD)
    if eocd_at < 0:
        raise RuntimeError("ZIP EOCD not present in downloaded tail")
    if eocd_at + 22 > len(data):
        raise RuntimeError("truncated EOCD")
    (_sig, disk, cd_disk, disk_entries, entries, cd_size, cd_offset,
     comment_len) = struct.unpack_from("<4s4H2LH", data, eocd_at)
    if disk or cd_disk or disk_entries != entries:
        raise RuntimeError("multi-disk ZIP is not supported")
    if cd_offset < base:
        report = {
            "status": "NEED_MORE_TAIL",
            "total_size": args.total_size,
            "tail_size": len(data),
            "central_directory_offset": cd_offset,
            "central_directory_size": cd_size,
            "entries": entries,
            "required_tail_bytes": args.total_size - cd_offset,
        }
    else:
        pos = cd_offset - base
        names = []
        while pos < eocd_at:
            if data[pos:pos + 4] != CD:
                raise RuntimeError(f"bad central-directory signature at {pos + base}")
            fields = struct.unpack_from("<4s6H3L5H2L", data, pos)
            name_len, extra_len, note_len = fields[10], fields[11], fields[12]
            name_start = pos + 46
            raw_name = data[name_start:name_start + name_len]
            try:
                name = raw_name.decode("utf-8")
            except UnicodeDecodeError:
                name = raw_name.decode("cp437")
            names.append(name)
            pos = name_start + name_len + extra_len + note_len
        report = {
            "status": "CENTRAL_DIRECTORY_PARSED",
            "total_size": args.total_size,
            "tail_size": len(data),
            "central_directory_offset": cd_offset,
            "central_directory_size": cd_size,
            "entries_declared": entries,
            "entries_parsed": len(names),
            "names": names,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "names"}, indent=2))


if __name__ == "__main__":
    main()
