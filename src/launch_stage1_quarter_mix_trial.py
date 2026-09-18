"""Upload the code-only delta once, then watch and validate a bounded GPU trial."""
import argparse
import hashlib
import json
import time

import watch_stage1_quality_trial as watch

watch.WORK = watch.ROOT / "artifacts/kaggle-stage1-quarter-mix-trial-20260916"
watch.DATASET = "biadis/crashintent-stage1-quarter-mix-assets"
watch.KERNEL = "biadis/crashintent-stage1-quarter-mix-trial"


def main(resume):
    work = watch.WORK
    bundle = json.loads((work / "bundle.json").read_text(encoding="utf-8"))
    if (bundle["dataset"], bundle["kernel"]) != (watch.DATASET, watch.KERNEL):
        raise ValueError("Destination mismatch")
    for path, key in (("upload/quarter.bin", "upload_sha256"), ("kernel/Stage1_Quarter.ipynb", "notebook_sha256"),
                      ("kernel/kernel-metadata.json", "metadata_sha256")):
        with (work / path).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != bundle[key]:
                raise ValueError("Prepared payload changed")
    if resume:
        if not (work / "upload-attempt.json").exists():
            raise ValueError("No previous upload attempt")
        watch.main(False)
        return
    approval = json.loads((work / "upload-approval.json").read_text(encoding="utf-8"))
    if approval.get("approved") is not True or any(approval.get(key) != bundle[key] for key in
                                                  ("dataset", "kernel", "upload_sha256", "notebook_sha256", "metadata_sha256")):
        raise ValueError("Specific upload/launch approval does not match payload")
    with watch.tick_lock(work / "upload.lock") as acquired:
        if not acquired:
            raise RuntimeError("Upload already running")
        with (work / "upload-attempt.json").open("x") as stream:
            json.dump({"dataset": watch.DATASET, "sha256": bundle["upload_sha256"], "started_unix": time.time()}, stream)
        watch.update(status="UPLOADING", monitor_status="ACTIVE", uploaded=False, gpu_started=False, last_error=None)
        response = watch.run([watch.KAGGLE, "datasets", "create", "-p", work / "upload", "--quiet"], 1800)
        (work / "upload.log").write_text(response, encoding="utf-8")
        watch.update(status="UPLOADED", uploaded=True)
    watch.main(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-monitor", action="store_true")
    args = parser.parse_args()
    try:
        main(args.resume_monitor)
    except Exception as error:
        watch.update(monitor_status="FAILED", last_error=repr(error))
        raise
