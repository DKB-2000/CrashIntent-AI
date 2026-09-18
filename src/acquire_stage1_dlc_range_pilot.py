"""Acquire a small multi-document DLC-2021 pilot with HTTP byte ranges.

Only ZIP central directories and the contiguous ranges for selected clips are read.
Labels come from the official ``or``/``re`` type codes, never from visual review.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import struct
import subprocess
import time
import zlib

import cv2


INV = Path("artifacts/stage1-physical-recapture-inventory-20260916")
OUT = Path("artifacts/stage1-dlc-range-pilot-20260917")
RECORDS = {"or": 6792397, "re": 6792396}
PART_SIZE = 4293918720
WANTED_DOCS = ("aze_passport", "esp_id", "fin_id", "svk_id")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_status(**values) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "status.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(values, heartbeat_unix=time.time())
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def zip64(extra: bytes, usize: int, csize: int, offset: int, disk: int):
    pos = 0
    while pos + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, pos)
        data = extra[pos + 4:pos + 4 + size]
        pos += 4 + size
        if tag != 1:
            continue
        at = 0
        values = []
        for needed, width in ((usize == 0xFFFFFFFF, 8), (csize == 0xFFFFFFFF, 8),
                              (offset == 0xFFFFFFFF, 8), (disk == 0xFFFF, 4)):
            if needed:
                values.append(int.from_bytes(data[at:at + width], "little")); at += width
            else:
                values.append(None)
        return (values[0] if values[0] is not None else usize,
                values[1] if values[1] is not None else csize,
                values[2] if values[2] is not None else offset,
                values[3] if values[3] is not None else disk)
    return usize, csize, offset, disk


def read_central(kind: str):
    data = (INV / f"{kind}.zip.central").read_bytes()
    entries, pos = [], 0
    while pos + 46 <= len(data) and data[pos:pos + 4] == b"PK\x01\x02":
        fields = struct.unpack_from("<4s6H3L5H2L", data, pos)
        method, crc, csize, usize = fields[4], fields[7], fields[8], fields[9]
        nlen, xlen, clen, disk, offset = fields[10], fields[11], fields[12], fields[13], fields[16]
        name = data[pos + 46:pos + 46 + nlen].decode("utf-8")
        extra = data[pos + 46 + nlen:pos + 46 + nlen + xlen]
        usize, csize, offset, disk = zip64(extra, usize, csize, offset, disk)
        entries.append(dict(name=name, method=method, crc=crc, csize=csize, usize=usize,
                            offset=offset, disk=disk))
        pos += 46 + nlen + xlen + clen
    if not entries:
        raise ValueError(f"No central entries parsed for {kind}")
    return entries


def part_name(kind: str, disk: int) -> str:
    if kind == "or":
        if disk != 0: raise ValueError("Unexpected disk for or.zip")
        return "or.zip"
    return f"re.z{disk + 1:02d}" if disk < 8 else "re.zip"


def curl_range(url: str, start: int, end: int, target: Path) -> None:
    command = ["curl.exe", "--fail", "--location", "--retry", "3", "--retry-all-errors",
               "--connect-timeout", "30", "--max-time", "1800", "--range", f"{start}-{end}",
               url, "-o", str(target)]
    subprocess.run(command, check=True)
    expected = end - start + 1
    if target.stat().st_size != expected:
        raise ValueError(f"Range size mismatch: {target}: {target.stat().st_size} != {expected}")


def choose(entries, kind: str):
    groups = {}
    for item in entries:
        p = PurePosixPath(item["name"])
        if p.suffix.lower() != ".jpg" or kind not in p.parts[-2]:
            continue
        doc, clip = p.parts[-3], p.parts[-2]
        if doc in WANTED_DOCS and clip.startswith(f"00.{kind}"):
            groups.setdefault((doc, clip), []).append(item)
    chosen = {}
    for doc in WANTED_DOCS:
        options = [(clip, items) for (d, clip), items in groups.items() if d == doc and len(items) >= 16]
        options.sort(key=lambda x: x[0])
        if not options:
            raise ValueError(f"No {kind} clip for {doc}")
        # Prefer a clip fully contained in one split part.
        match = next(((clip, items) for clip, items in options if len({i['disk'] for i in items}) == 1), None)
        if match is None: raise ValueError(f"No single-part {kind} clip for {doc}")
        chosen[doc] = match
    return chosen


def acquire_clip(kind: str, doc: str, clip: str, items: list[dict]):
    items.sort(key=lambda x: x["offset"])
    disk = items[0]["disk"]
    if any(x["disk"] != disk for x in items): raise ValueError("Clip crosses split part")
    filename = part_name(kind, disk)
    record = RECORDS[kind]
    url = f"https://zenodo.org/api/records/{record}/files/{filename}/content"
    start = items[0]["offset"]
    end = max(x["offset"] + 30 + len(x["name"].encode()) + 4096 + x["csize"] for x in items)
    raw = OUT / "ranges" / f"{doc}-{clip}-{filename}.bin"
    raw.parent.mkdir(parents=True, exist_ok=True)
    curl_range(url, start, end - 1, raw)
    blob = raw.read_bytes()
    frame_dir = OUT / "frames" / kind / doc / clip
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in items:
        at = item["offset"] - start
        if blob[at:at + 4] != b"PK\x03\x04": raise ValueError(f"Bad local header: {item['name']}")
        local = struct.unpack_from("<4s5H3L2H", blob, at)
        nlen, xlen = local[-2], local[-1]
        payload = blob[at + 30 + nlen + xlen:at + 30 + nlen + xlen + item["csize"]]
        decoded = payload if item["method"] == 0 else zlib.decompress(payload, -15)
        if len(decoded) != item["usize"] or (zlib.crc32(decoded) & 0xFFFFFFFF) != item["crc"]:
            raise ValueError(f"CRC/size mismatch: {item['name']}")
        path = frame_dir / PurePosixPath(item["name"]).name
        path.write_bytes(decoded); paths.append(path)
    return sorted(paths)


def make_video(kind: str, doc: str, clip: str, frames: list[Path]):
    first = cv2.imread(str(frames[0])); h, w = first.shape[:2]
    path = OUT / "videos" / f"{doc}__{clip}.mp4"; path.parent.mkdir(exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    if not writer.isOpened(): raise RuntimeError(path)
    for frame in frames:
        image = cv2.imread(str(frame))
        if image is None or image.shape[:2] != (h, w): raise ValueError(frame)
        writer.write(image)
    writer.release()
    cap = cv2.VideoCapture(str(path)); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    if count != len(frames): raise ValueError(f"Video count mismatch: {path}")
    return dict(source_id=f"dlc2021/{doc}/{clip}", document=doc, clip=clip,
                label="ORIGINAL" if kind == "or" else "RERECORDED",
                label_source="DLC-2021 filename type code", frames=len(frames),
                video_path=path.as_posix(), video_sha256=digest(path))


def main():
    save_status(status="RUNNING", pid=os.getpid(), expected_documents=len(WANTED_DOCS), completed_clips=0)
    rows = []
    try:
        for kind in ("or", "re"):
            selected = choose(read_central(kind), kind)
            for doc in WANTED_DOCS:
                clip, items = selected[doc]
                frames = acquire_clip(kind, doc, clip, items)
                rows.append(make_video(kind, doc, clip, frames))
                save_status(completed_clips=len(rows), current=f"{doc}/{clip}")
        manifest = OUT / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        save_status(status="DATA_VALIDATED", completed_clips=len(rows), frames=sum(r["frames"] for r in rows),
                    class_counts={x:sum(r["label"] == x for r in rows) for x in ("ORIGINAL", "RERECORDED")},
                    manifest_sha256=digest(manifest), selected_documents=list(WANTED_DOCS),
                    scope="MULTI_DOCUMENT_RANGE_PILOT")
    except Exception as exc:
        save_status(status="FAILED", error=repr(exc))
        raise


if __name__ == "__main__": main()
