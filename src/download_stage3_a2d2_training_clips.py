"""Acquire causal 16-frame A2D2 clips for the Stage 3 motion pipeline."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import pandas as pd


PLAN = Path("artifacts/stage3-a2d2-training-plan-20260917/report.json")
OUT = Path("artifacts/stage3-a2d2-training-clips-20260917")
RAW = Path("data_raw/a2d2/training-clips-20260917")


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    for attempt in range(50):
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.1)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_name(prefix: str, index: int) -> str:
    return f"{prefix}{index:09d}.png"


def download_one(item: tuple[str, str, str, int], deadline: float) -> dict:
    route, base, prefix, index = item
    name = frame_name(prefix, index)
    destination = RAW / route / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError("six-hour clip acquisition deadline")
            if not destination.exists():
                request = urllib.request.Request(f"{base}/{name}")
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
                temporary = destination.with_suffix(".png.partial")
                temporary.write_bytes(data)
                os.replace(temporary, destination)
            image = cv2.imread(str(destination))
            if image is None or image.shape[:2] != (1208, 1920):
                destination.unlink(missing_ok=True)
                raise ValueError(f"invalid A2D2 image {name}")
            return {"route":route, "index":index, "file":str(destination), "bytes":destination.stat().st_size, "sha256":file_hash(destination)}
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt * 3)
    raise AssertionError("unreachable")


def write_clip(route: str, prefix: str, indices: list[int], destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".temporary.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 402))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {temporary}")
    try:
        for index in indices:
            image = cv2.imread(str(RAW / route / frame_name(prefix, index)))
            if image is None:
                raise ValueError(f"cannot decode frame {index}")
            writer.write(cv2.resize(image, (640, 402), interpolation=cv2.INTER_AREA))
    finally:
        writer.release()
    os.replace(temporary, destination)
    capture = cv2.VideoCapture(str(destination))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if frames != 16 or decoded != 16 or abs(fps - 10.0) > 0.01:
        raise ValueError(f"invalid clip {destination}: frames={frames}, decoded={decoded}, fps={fps}")
    return {"file":str(destination), "bytes":destination.stat().st_size, "sha256":file_hash(destination)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "worker.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("A2D2 training clip acquisition already running") from exc
    os.write(fd, str(os.getpid()).encode("ascii"))
    os.close(fd)
    status_path = OUT / "status.json"
    deadline = time.monotonic() + 12 * 3600
    try:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        if plan.get("status") != "COMPLETE_VALIDATED":
            raise ValueError("A2D2 pilot plan is not validated")
        clips=[]
        for number,row in enumerate(plan["selections"]):
            center=int(row["camera_index"])
            indices=list(range(center-45, center+1, 3))
            if len(indices)!=16 or min(indices)<0:
                raise ValueError(f"invalid causal window at {center}")
            image_url=row["image_url"];base=image_url.rsplit('/',1)[0];prefix=image_url.rsplit('/',1)[1].rsplit('_',1)[0]+'_'
            clips.append({"id":f"A2D2_TRAIN_{number:03d}", "route":row["route"], "base":base, "prefix":prefix, "label":row["label"], "center":center, "indices":indices})
        unique=sorted({(clip['route'],clip['base'],clip['prefix'],i) for clip in clips for i in clip["indices"]})
        completed=[]
        mutex=threading.Lock()
        save(status_path,{"status":"DOWNLOADING", "completed":0, "total":len(unique), "clips":len(clips), "pid":os.getpid()})
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures={executor.submit(download_one,item,deadline):item for item in unique}
            for future in concurrent.futures.as_completed(futures):
                result=future.result()
                with mutex:
                    completed.append(result)
                    save(status_path,{"status":"DOWNLOADING", "completed":len(completed), "total":len(unique), "clips":len(clips), "pid":os.getpid()})
        save(status_path,{"status":"ENCODING", "completed":len(unique), "total":len(unique), "clips_completed":0, "clips":len(clips), "pid":os.getpid()})
        clip_reports=[]
        sample_rows=[]
        for number,clip in enumerate(clips,1):
            video=write_clip(clip["route"],clip["prefix"],clip["indices"], OUT/"videos"/f"{clip['id']}.mp4")
            clip_reports.append({**clip, **video})
            sample_rows.append({"ID":clip["id"], "route":f"A2D2_{clip['route']}", "endpoint":15, "steer_label":clip["label"]})
            save(status_path,{"status":"ENCODING", "completed":len(unique), "total":len(unique), "clips_completed":number, "clips":len(clips), "pid":os.getpid()})
        pd.DataFrame(sample_rows).to_csv(OUT/"samples.csv",index=False)
        report={
            "status":"ACQUIRED_VALIDATED", "unique_images":len(unique), "clips":len(clips),
            "raw_bytes":sum(x["bytes"] for x in completed),
            "video_bytes":sum(x["bytes"] for x in clip_reports),
            "history":"16 frames at 10fps, causal center-45:center inclusive from 30fps source",
            "class_counts":{label:sum(c["label"]==label for c in clips) for label in ("LEFT","STRAIGHT","RIGHT")},
            "images":sorted(completed,key=lambda x:(x["route"],x["index"])), "clip_records":clip_reports,
        }
        save(OUT/"report.json",report)
        save(status_path,{"status":"ACQUIRED_VALIDATED", "completed":len(unique), "total":len(unique), "clips_completed":len(clips), "clips":len(clips), "report":str(OUT/"report.json")})
    except Exception as exc:
        save(status_path,{"status":"FAILED", "error":repr(exc)})
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
