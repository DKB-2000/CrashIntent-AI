"""Acquire all split parts of official MM-AU train detection images.

Parts are validated independently, then exposed as one concatenated gzip stream.
Only images belonging to the frozen selected candidate list are extracted.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw/mm-au-detection-images-train-20260918"
PARTS = DATA / "parts"
EXTRACTED = DATA / "selected"
ART = ROOT / "artifacts/stage2-mmau-detection-images-train-20260918"
SELECTION = ROOT / "artifacts/stage2-mmau-scene-selection-20260918/selected_2500.csv"
LABELS = ROOT / "data_raw/mm-au-detection-labels-20260917/labels.tar.gz"
LEGACY_AW = ROOT / "data_raw/mm-au-detection-images-pilot-20260918/train_part_aw.tar.gz"
API = "https://huggingface.co/api/datasets/JeffreyChou/MM-AU/tree/main?recursive=true&limit=1000"
BASE = "https://huggingface.co/datasets/JeffreyChou/MM-AU/resolve/main/"
PREFIX = "MMAU_Det_paper/images/train_chunks/"
DEADLINE_SECONDS = 12 * 60 * 60
LABEL_PATTERN = re.compile(r"^labels/train/(\d+)_(\d+)_(\d+)\.txt$", re.I)
# Most paper images use 16-hex names, but 60 official COCO entries contain
# numeric/scientific-notation basenames (for example ``8.34E+20.jpg``).
# Match any single safe basename while preserving the exact 295,013-image gate.
IMAGE_PATTERN = re.compile(r"^train/([^/\\]+\.(?:jpg|jpeg|png))$", re.I)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(30):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(.1)


def save(status: str, **extra) -> None:
    atomic_json(ART / "status.json", {
        "status": status, "updated_unix": time.time(),
        "data_dir": str(DATA.relative_to(ROOT)), **extra,
    })


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def remote_manifest() -> list[dict]:
    request = urllib.request.Request(API, headers={"User-Agent": "crashvideo-mmau-train/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = json.load(response)
    files = []
    for item in data:
        path = item.get("path", "")
        if item.get("type") == "file" and path.startswith(PREFIX):
            lfs = item.get("lfs") or {}
            files.append({"path": path, "name": Path(path).name,
                          "size": int(item["size"]), "sha256": lfs.get("oid")})
    files.sort(key=lambda x: x["name"])
    if len(files) != 23 or any(not x["sha256"] for x in files):
        raise RuntimeError(f"Unexpected official train manifest: {len(files)} files")
    atomic_json(ART / "remote-manifest.json", {
        "status": "VALIDATED", "files": files,
        "total_bytes": sum(x["size"] for x in files),
    })
    return files


def adopt_legacy(manifest: list[dict]) -> None:
    if not LEGACY_AW.exists():
        return
    item = next(x for x in manifest if x["name"] == "train_part_aw")
    destination = PARTS / item["name"]
    if destination.exists():
        return
    if LEGACY_AW.stat().st_size == item["size"] and sha256(LEGACY_AW) == item["sha256"]:
        LEGACY_AW.replace(destination)


def fetch(item: dict, index: int, total: int, deadline: float) -> None:
    target = PARTS / item["name"]
    partial = target.with_suffix(target.suffix + ".part")
    if target.exists():
        if target.stat().st_size == item["size"] and sha256(target) == item["sha256"]:
            return
        target.replace(partial)
    for attempt in range(1, 4):
        if time.monotonic() >= deadline:
            raise TimeoutError("12-hour MM-AU train acquisition deadline")
        current = partial.stat().st_size if partial.exists() else 0
        if current > item["size"]:
            partial.unlink(); current = 0
        headers = {"User-Agent": "crashvideo-mmau-train/1.0"}
        if current:
            headers["Range"] = f"bytes={current}-"
        save("DOWNLOADING", part_index=index, parts_total=total, part=item["name"],
             part_bytes=current, part_expected=item["size"])
        try:
            request = urllib.request.Request(BASE + item["path"] + "?download=true", headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                if current and response.status != 206:
                    partial.unlink(missing_ok=True)
                    raise RuntimeError(f"Range resume rejected: HTTP {response.status}")
                with partial.open("ab" if current else "wb") as output:
                    last = time.monotonic()
                    while True:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("12-hour MM-AU train acquisition deadline")
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        if time.monotonic() - last >= 30:
                            output.flush()
                            save("DOWNLOADING", part_index=index, parts_total=total,
                                 part=item["name"], part_bytes=output.tell(), part_expected=item["size"])
                            last = time.monotonic()
            if partial.stat().st_size != item["size"]:
                raise RuntimeError(f"Size mismatch for {item['name']}: {partial.stat().st_size}")
            partial.replace(target)
            if sha256(target) != item["sha256"]:
                target.replace(partial)
                raise RuntimeError(f"SHA256 mismatch for {item['name']}")
            return
        except Exception as exc:
            save("RETRYING" if attempt < 3 else "FAILED", part_index=index, parts_total=total,
                 part=item["name"], part_bytes=partial.stat().st_size if partial.exists() else 0,
                 part_expected=item["size"], attempt=attempt, error=repr(exc))
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


class ConcatenatedParts(io.RawIOBase):
    def __init__(self, paths: list[Path]):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        view = memoryview(buffer)
        written = 0
        while written < len(view):
            if self.current is None:
                try:
                    self.current = next(self.paths).open("rb")
                except StopIteration:
                    break
            count = self.current.readinto(view[written:])
            if count:
                written += count
            else:
                self.current.close(); self.current = None
        return written

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def iter_json_array(stream, key: str):
    """Yield objects from one top-level JSON array without loading it all."""
    marker = f'"{key}": ['
    decoder = json.JSONDecoder()
    buffer = ""
    while marker not in buffer:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            raise RuntimeError(f"JSON array not found: {key}")
        buffer += chunk.decode("utf-8")
        if len(buffer) > len(marker) + 2 * 1024 * 1024:
            buffer = buffer[-(len(marker) + 2 * 1024 * 1024):]
    buffer = buffer.split(marker, 1)[1]
    while True:
        buffer = buffer.lstrip(" \t\r\n,")
        if buffer.startswith("]"):
            return
        try:
            value, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                raise RuntimeError(f"Truncated JSON array: {key}")
            buffer += chunk.decode("utf-8")
            continue
        yield value
        buffer = buffer[end:]


def parse_yolo(raw: str) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        boxes.append((int(fields[0]), *(float(x) for x in fields[1:])))
    return boxes


def boxes_match(left, right, tolerance: float = 7e-4) -> bool:
    if len(left) != len(right):
        return False
    left = sorted(left)
    right = sorted(right)
    return all(a[0] == b[0] and all(abs(x - y) <= tolerance for x, y in zip(a[1:], b[1:]))
               for a, b in zip(left, right))


MASK64 = (1 << 64) - 1


def box_token(box, scale: int) -> int:
    values = (int(box[0]), *(int(round(float(value) * scale)) for value in box[1:]))
    value = 1469598103934665603
    for item in values:
        value ^= item & MASK64
        value = (value * 1099511628211) & MASK64
    return value


def box_fingerprint(boxes, scale: int) -> tuple[int, int, int]:
    total = xor = 0
    for box in boxes:
        token = box_token(box, scale)
        total = (total + token) & MASK64
        xor ^= token
    return len(boxes), total, xor


def build_image_mapping(selected: set[str]) -> tuple[dict[str, tuple[str, int]], dict]:
    """Map hashed COCO image names to original video/frame identities.

    The official archive stores the same train samples twice: train.json uses
    hashed image names while the YOLO members retain type_video_frame names.
    The two filename spaces are joined by content fingerprints of object class
    and normalized bbox multisets. Empty detection rows are intentionally
    omitted because they have no information with which to identify an image.
    """
    if not LABELS.exists():
        raise FileNotFoundError(LABELS)
    with tarfile.open(LABELS, "r:gz") as archive:
        member = archive.getmember("labels/train.json")
        source = archive.extractfile(member)
        if source is None:
            raise RuntimeError("Cannot read labels/train.json")
        images = list(iter_json_array(source, "images"))

    if len(images) != 295013:
        raise RuntimeError(f"Unexpected train image count: {len(images)}")
    max_image_id = max(int(image["id"]) for image in images)
    filenames = [None] * (max_image_id + 1)
    widths = [0] * (max_image_id + 1)
    heights = [0] * (max_image_id + 1)
    for image in images:
        image_id = int(image["id"])
        filenames[image_id] = image["file_name"]
        widths[image_id] = int(image["width"])
        heights[image_id] = int(image["height"])
    if len({name for name in filenames if name is not None}) != len(images):
        raise RuntimeError("Duplicate hashed train image names")

    targets = []
    train_label_count = empty_selected = 0
    with tarfile.open(LABELS, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            match = LABEL_PATTERN.match(member.name)
            if not match:
                continue
            typ, video, frame = match.groups()
            train_label_count += 1
            key = f"{int(typ)}_{int(video)}"
            if key in selected:
                raw = archive.extractfile(member).read().decode("utf-8", "replace")
                boxes = parse_yolo(raw)
                if not boxes:
                    empty_selected += 1
                    continue
                targets.append({"key": key, "frame": int(frame),
                                "fp3": box_fingerprint(boxes, 1000),
                                "fp4": box_fingerprint(boxes, 10000)})
    if train_label_count != len(images):
        raise RuntimeError(f"Unexpected train label count: {train_label_count}/{len(images)}")
    if len(targets) < len(selected):
        raise RuntimeError(f"Too few non-empty target frames: {len(targets)}")

    counts = [0] * (max_image_id + 1)
    sums3 = [0] * (max_image_id + 1); xors3 = [0] * (max_image_id + 1)
    sums4 = [0] * (max_image_id + 1); xors4 = [0] * (max_image_id + 1)
    with tarfile.open(LABELS, "r:gz") as archive:
        source = archive.extractfile("labels/train.json")
        if source is None:
            raise RuntimeError("Cannot reread labels/train.json")
        for annotation in iter_json_array(source, "annotations"):
            image_id = int(annotation["image_id"])
            width, height = widths[image_id], heights[image_id]
            x, y, box_width, box_height = (float(v) for v in annotation["bbox"])
            box = (int(annotation["category_id"]) - 1,
                   (x + box_width / 2) / width, (y + box_height / 2) / height,
                   box_width / width, box_height / height)
            token3 = box_token(box, 1000); token4 = box_token(box, 10000)
            counts[image_id] += 1
            sums3[image_id] = (sums3[image_id] + token3) & MASK64; xors3[image_id] ^= token3
            sums4[image_id] = (sums4[image_id] + token4) & MASK64; xors4[image_id] ^= token4

    wanted_full = {(target["fp3"], target["fp4"]) for target in targets}
    wanted_coarse = {target["fp3"] for target in targets}
    images_full = {}; images_coarse = {}
    for image_id in range(1, max_image_id + 1):
        if not counts[image_id] or filenames[image_id] is None:
            continue
        fp3 = (counts[image_id], sums3[image_id], xors3[image_id])
        fp4 = (counts[image_id], sums4[image_id], xors4[image_id])
        if (fp3, fp4) in wanted_full:
            images_full.setdefault((fp3, fp4), []).append(filenames[image_id])
        if fp3 in wanted_coarse:
            images_coarse.setdefault(fp3, []).append(filenames[image_id])

    targets_full = {}
    for target in targets:
        targets_full.setdefault((target["fp3"], target["fp4"]), []).append(target)
    mapping = {}; mapped_targets = set(); full_matches = coarse_matches = 0
    for fingerprint, rows in targets_full.items():
        names = images_full.get(fingerprint, [])
        if len(rows) == len(names) == 1:
            mapping[names[0]] = (rows[0]["key"], rows[0]["frame"])
            mapped_targets.add(id(rows[0])); full_matches += 1
    targets_coarse = {}
    for target in targets:
        if id(target) not in mapped_targets:
            targets_coarse.setdefault(target["fp3"], []).append(target)
    for fingerprint, rows in targets_coarse.items():
        names = [name for name in images_coarse.get(fingerprint, []) if name not in mapping]
        if len(rows) == len(names) == 1:
            mapping[names[0]] = (rows[0]["key"], rows[0]["frame"])
            mapped_targets.add(id(rows[0])); coarse_matches += 1

    coverage = len(mapping) / len(targets)
    return mapping, {"train_images": len(images), "train_labels": train_label_count,
                     "selected_nonempty_labels": len(targets), "selected_empty_labels_omitted": empty_selected,
                     "mapped_selected_images": len(mapping), "mapping_coverage": coverage,
                     "mapping_full_matches": full_matches, "mapping_coarse_matches": coarse_matches}


def load_cached_mapping(selected: set[str]):
    report_path = ART / "mapping-report.json"
    csv_path = ART / "content-mapping.csv"
    if not report_path.exists() or not csv_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mapping = {}
    with csv_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["video_key"]
            if key not in selected:
                raise RuntimeError(f"Cached mapping contains unexpected video: {key}")
            mapping[row["hashed_file"]] = (key, int(row["frame"]))
    if len(mapping) != int(report.get("mapped_selected_images", -1)):
        raise RuntimeError("Cached mapping cardinality mismatch")
    if int(report.get("selected_videos_expected", -1)) != len(selected):
        raise RuntimeError("Cached mapping selection cardinality mismatch")
    return mapping, report


def extract_selected(manifest: list[dict], deadline: float) -> dict:
    with SELECTION.open(encoding="utf-8", newline="") as stream:
        selected_rows = {r["video_key"]: r for r in csv.DictReader(stream) if r["label_split"] == "train"}
    selected = set(selected_rows)
    cached = load_cached_mapping(selected)
    image_mapping, mapping_report = cached if cached is not None else build_image_mapping(selected)
    mapped_frames = {}
    for key, frame in image_mapping.values():
        mapped_frames.setdefault(key, set()).add(frame)
    covered_videos = set(mapped_frames)
    near_ai = {key for key, frames in mapped_frames.items()
               if any(abs(frame - int(selected_rows[key]["t_ai"])) <= 5 for frame in frames)}
    near_co = {key for key, frames in mapped_frames.items()
               if any(abs(frame - int(selected_rows[key]["t_co"])) <= 5 for frame in frames)}
    mapping_report.update({
        "selected_videos_expected": len(selected),
        "selected_videos_mapped": len(covered_videos),
        "selected_videos_with_t_ai_neighborhood": len(near_ai),
        "selected_videos_with_t_co_neighborhood": len(near_co),
        "missing_videos": sorted(selected - covered_videos),
        "missing_t_ai_neighborhood": sorted(selected - near_ai),
        "missing_t_co_neighborhood": sorted(selected - near_co),
    })
    atomic_json(ART / "mapping-report.json", mapping_report)
    with (ART / "content-mapping.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("hashed_file", "video_key", "frame"))
        for hashed_file, (key, frame) in sorted(image_mapping.items()):
            writer.writerow((hashed_file, key, frame))
    if mapping_report["mapping_coverage"] < .85:
        raise RuntimeError(f"Content mapping coverage too low: {mapping_report['mapping_coverage']:.4%}")
    video_ratio = len(covered_videos) / len(selected)
    ai_ratio = len(near_ai) / len(selected)
    co_ratio = len(near_co) / len(selected)
    if video_ratio < .999 or ai_ratio < .93 or co_ratio < .93:
        raise RuntimeError(
            "Semantic mapping coverage incomplete: "
            f"videos={len(covered_videos)}/{len(selected)}, "
            f"t_ai={len(near_ai)}/{len(selected)}, t_co={len(near_co)}/{len(selected)}")
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    image_files = selected_images = members = 0
    selected_videos = set()
    samples = []
    raw = ConcatenatedParts([PARTS / x["name"] for x in manifest])
    buffered = io.BufferedReader(raw, 8 * 1024 * 1024)
    with tarfile.open(fileobj=buffered, mode="r|gz") as archive:
        for member in archive:
            if time.monotonic() >= deadline:
                raise TimeoutError("12-hour MM-AU train extraction deadline")
            members += 1
            if not member.isfile():
                continue
            if len(samples) < 30:
                samples.append(member.name)
            match = IMAGE_PATTERN.match(member.name)
            if not match:
                continue
            image_files += 1
            identity = image_mapping.get(match.group(1))
            if identity is None:
                continue
            key, frame = identity
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read {member.name}")
            folder = EXTRACTED / key
            folder.mkdir(exist_ok=True)
            suffix = Path(match.group(1)).suffix.lower()
            destination = folder / f"frame_{frame:06d}{suffix}"
            with destination.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            selected_images += 1; selected_videos.add(key)
            if selected_images % 1000 == 0:
                save("EXTRACTING", members_scanned=members, image_files_scanned=image_files,
                     selected_images=selected_images, selected_videos=len(selected_videos))
    if image_files != 295013:
        raise RuntimeError(f"Unexpected train images in archive: {image_files}")
    if selected_images != len(image_mapping):
        raise RuntimeError(f"Selected extraction incomplete: {selected_images}/{len(image_mapping)}")
    if selected_videos != covered_videos:
        raise RuntimeError(f"Extracted video coverage differs from validated mapping: "
                           f"{len(selected_videos)}/{len(covered_videos)}")
    return {"members": members, "image_files": image_files,
            "selected_train_candidates_expected": len(selected),
            "selected_train_candidates_usable": len(selected_videos),
            "selected_videos_extracted": len(selected_videos),
            "selected_images_extracted": selected_images, "sample_members": samples,
            "acquisition_scope": "Validated unique content matches only; ambiguous and empty-label frames omitted",
            **mapping_report}


def main() -> None:
    started = time.monotonic(); deadline = started + DEADLINE_SECONDS
    PARTS.mkdir(parents=True, exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
    save("STARTING")
    try:
        manifest = remote_manifest(); adopt_legacy(manifest)
        for index, item in enumerate(manifest, 1):
            fetch(item, index, len(manifest), deadline)
        save("VERIFYING_PARTS", parts_complete=len(manifest),
             bytes=sum((PARTS / x["name"]).stat().st_size for x in manifest))
        result = extract_selected(manifest, deadline)
        result.update({"status": "ACQUIRED_VALIDATED_PARTIAL", "parts": len(manifest),
                       "archive_bytes": sum(x["size"] for x in manifest),
                       "elapsed_seconds": time.monotonic() - started})
        atomic_json(ART / "report.json", result)
        save("ACQUIRED_VALIDATED_PARTIAL", parts_complete=len(manifest),
             selected_videos=result["selected_videos_extracted"],
             selected_images=result["selected_images_extracted"],
             report=str((ART / "report.json").relative_to(ROOT)))
    except Exception as exc:
        save("FAILED", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
