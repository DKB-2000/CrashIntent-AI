"""Prepare a code-only GPU comparison using the already uploaded Stage1 assets."""
import ast
import hashlib
import json
import zipfile
from pathlib import Path

from stage1_quality_trial_runner import audit_rows, schedule, training_rows

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "artifacts/kaggle-stage1-screen-trial-20260914"
OUT = ROOT / "artifacts/kaggle-stage1-quarter-mix-trial-20260916"
DATASET = "biadis/crashintent-stage1-quarter-mix-assets"
KERNEL = "biadis/crashintent-stage1-quarter-mix-trial"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    old = json.loads((OLD / "bundle.json").read_text(encoding="utf-8"))
    if sha(OLD / "dataset/quality.bin") != old["sha256"]:
        raise ValueError("Previously validated bundle changed")
    prior = json.loads((OLD / "validated/result-validation.json").read_text(encoding="utf-8"))
    if prior["status"] != "PASS":
        raise ValueError("Screen trial was not validated")
    with zipfile.ZipFile(OLD / "dataset/quality.bin") as source:
        plan = json.loads(source.read("plan.json"))
        manifest = json.loads(source.read("manifest.json"))
        plan.update(arms=["baseline", "mixed_50", "mixed_25", "screen_mix_25"],
                    design="Fixed 4 source pairs/update, 3 existing + 1 quality. Screen arm replaces only the quality RERECORDED input; same initial model, source/slot schedule, updates, learning rate, evaluation and threshold.")
        train, val = audit_rows(plan["rows"], plan["excluded"])
        sources = sorted({r["source_id"] for r in train})
        for step, entries in enumerate(schedule(sources)):
            control = training_rows(train, "mixed_25", step, entries)
            candidate = training_rows(train, "screen_mix_25", step, entries)
            if len(control) != 8 or len(candidate) != 8 or sum(a != b for a, b in zip(control, candidate)) != 1:
                raise ValueError("Unmatched quarter-mix schedule")
        runner = (ROOT / "src/stage1_quality_trial_runner.py").read_text(encoding="utf-8-sig")
        ast.parse(runner)
        replacements = {"runner.py": runner.encode(), "plan.json": json.dumps(plan, indent=2).encode()}
        manifest.update({name: hashlib.sha256(data).hexdigest() for name, data in replacements.items()})
        replacements["manifest.json"] = json.dumps(manifest, indent=2).encode()
        (OUT / "dataset").mkdir(parents=True, exist_ok=False)
        (OUT / "upload").mkdir()
        (OUT / "kernel").mkdir()
        with zipfile.ZipFile(OUT / "dataset/quality.bin", "x", zipfile.ZIP_STORED) as full:
            for name in source.namelist():
                data = replacements.get(name)
                if data is None:
                    data = source.read(name)
                    if hashlib.sha256(data).hexdigest() != manifest[name]:
                        raise ValueError("Source member changed")
                full.writestr(name, data)
        with zipfile.ZipFile(OUT / "upload/quarter.bin", "x", zipfile.ZIP_STORED) as delta:
            for name, data in replacements.items():
                delta.writestr(name, data)
    delta_sha = sha(OUT / "upload/quarter.bin")
    write(OUT / "upload/quarter-assets.json", {"sha256": delta_sha})
    write(OUT / "upload/dataset-metadata.json", {"id": DATASET, "title": "CrashIntent Stage1 Quarter Mix Code Assets",
                                                   "licenses": [{"name": "other"}], "description": "Private code/config only; reuse existing train-only CCD assets."})
    notebook = json.loads((OLD / "kernel/Stage1_Screen.ipynb").read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][0]["source"])
    needle = "venv=Path(tempfile.mkdtemp(dir='/tmp',prefix='pinned-'))"
    insertion = f"""quarter=next(Path('/kaggle/input').rglob('quarter-assets.json')).parent
with (quarter/'quarter.bin').open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()=={delta_sha!r}
with zipfile.ZipFile(quarter/'quarter.bin') as z:
 for name in z.namelist(): assert (work/name).resolve().is_relative_to(work.resolve())
 z.extractall(work)
"""
    if code.count(needle) != 1:
        raise ValueError("Notebook template mismatch")
    code = code.replace(needle, insertion + needle)
    ast.parse(code)
    notebook["cells"][0]["source"] = code.splitlines(keepends=True)
    write(OUT / "kernel/Stage1_Quarter.ipynb", notebook)
    metadata = json.loads((OLD / "kernel/kernel-metadata.json").read_text(encoding="utf-8"))
    metadata.update(id=KERNEL, title="CrashIntent Stage1 Quarter Mix Trial", code_file="Stage1_Quarter.ipynb",
                    dataset_sources=[old["source_dataset"], old["dataset"], DATASET])
    write(OUT / "kernel/kernel-metadata.json", metadata)
    report = {"status": "PREPARED_NOT_UPLOADED", "dataset": DATASET, "kernel": KERNEL,
              "source_datasets": metadata["dataset_sources"], "sha256": sha(OUT / "dataset/quality.bin"),
              "bytes": (OUT / "dataset/quality.bin").stat().st_size, "upload_sha256": delta_sha,
              "upload_bytes": (OUT / "upload/quarter.bin").stat().st_size,
              "notebook_sha256": sha(OUT / "kernel/Stage1_Quarter.ipynb"),
              "metadata_sha256": sha(OUT / "kernel/kernel-metadata.json"),
              "training_videos": len(train), "evaluation_videos": len(val),
              "prediction_rows": len(val) * 4, "optimizer_updates": 128 * 3}
    write(OUT / "bundle.json", report)
    write(OUT / "remote-run.json", {"status": "PREPARED", "monitor_status": "NOT_STARTED", "dataset": DATASET,
                                     "kernel": KERNEL, "uploaded": False, "gpu_started": False})
    print(json.dumps(report))


if __name__ == "__main__":
    main()
