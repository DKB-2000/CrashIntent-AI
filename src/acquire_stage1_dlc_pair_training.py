"""Range-acquire 40 automatically labelled DLC-2021 original/recapture pairs."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import time

import acquire_stage1_dlc_range_pilot as base

OUT = Path("artifacts/stage1-dlc-pair-training-20260917")
DOCS = ("aze_passport", "esp_id", "est_id", "fin_id", "grc_passport",
        "lva_passport", "rus_internalpassport", "srb_passport", "svk_id")
# The leading alb_id ranges repeatedly return upstream 504. Preserve 40 pairs by
# taking a fifth template from four later document types whose ranges are healthy.
EXTRA_TEMPLATE_DOCS = {"aze_passport", "esp_id", "fin_id", "svk_id"}
HOLDOUT_DOCS = {"est_id", "srb_passport"}


def templates(doc):
    return ("00", "01", "02", "03", "04") if doc in EXTRA_TEMPLATE_DOCS else ("00", "01", "02", "03")


def save(**values):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "status.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(values, heartbeat_unix=time.time())
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(old, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def groups(kind):
    result = {}
    for item in base.read_central(kind):
        parts = item["name"].split("/")
        if not item["name"].lower().endswith(".jpg") or len(parts) < 3:
            continue
        doc, clip = parts[-3], parts[-2]
        if doc in DOCS and f".{kind}" in clip:
            result.setdefault((doc, clip), []).append(item)
    return result


def select(kind, available):
    selected = {}
    for doc in DOCS:
        for template in templates(doc):
            prefix = f"{template}.{kind}"
            options = sorted((clip, items) for (d, clip), items in available.items()
                             if d == doc and clip.startswith(prefix) and len(items) >= 16)
            match = next(((clip, sorted(items, key=lambda x: x["offset"])) for clip, items in options
                          if len({x["disk"] for x in items}) == 1), None)
            if match is None:
                raise ValueError(f"No complete {kind} clip: {doc}/{template}")
            clip, items = match
            center = (len(items) - 16) // 2
            selected[(doc, template)] = (clip, items[center:center + 16])
    return selected


def main():
    started = time.monotonic()
    base.OUT = OUT
    save(status="RUNNING", pid=os.getpid(), expected_pairs=40, expected_clips=80, completed_clips=0)
    rows = []
    try:
        choices = {kind: select(kind, groups(kind)) for kind in ("or", "re")}
        for doc in DOCS:
            for template in templates(doc):
                pair_id = f"{doc}/{template}"
                for kind in ("or", "re"):
                    if time.monotonic() - started > 4 * 60 * 60:
                        raise TimeoutError("Four-hour acquisition budget reached")
                    clip, items = choices[kind][(doc, template)]
                    frames = base.acquire_clip(kind, doc, clip, items)
                    row = base.make_video(kind, doc, clip, frames)
                    row.update(pair_id=pair_id, template=template,
                               split="val" if doc in HOLDOUT_DOCS else "train")
                    rows.append(row)
                    save(completed_clips=len(rows), current=row["source_id"])
        for pair in {r["pair_id"] for r in rows}:
            part = [r for r in rows if r["pair_id"] == pair]
            if len(part) != 2 or {r["label"] for r in part} != {"ORIGINAL", "RERECORDED"}:
                raise ValueError(f"Broken pair: {pair}")
            if len({r["split"] for r in part}) != 1:
                raise ValueError(f"Pair split leakage: {pair}")
        manifest = OUT / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        save(status="DATA_VALIDATED", completed_clips=80, completed_pairs=40,
             train_pairs=sum(r["label"] == "ORIGINAL" and r["split"] == "train" for r in rows),
             val_pairs=sum(r["label"] == "ORIGINAL" and r["split"] == "val" for r in rows),
             documents=len(DOCS), frames=sum(r["frames"] for r in rows), excluded_documents=["alb_id"],
             manifest_sha256=base.digest(manifest), holdout_documents=sorted(HOLDOUT_DOCS),
             label_policy="Official DLC-2021 filename type codes only; no human labels")
    except Exception as exc:
        save(status="FAILED", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
