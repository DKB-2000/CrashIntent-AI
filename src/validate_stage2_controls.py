"""Recheck existing Stage2 controls against source labels and saved model outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import stage2_pipeline as p

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--base-zip", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = json.loads((args.run_dir / "report.json").read_text())
    labels = {}
    for split in ("train", "validation"):
        path = args.split_dir / f"labels_{split}.csv"
        assert sha(path) == report["split_sha256"][split], "Changed source labels"
        labels[split] = pd.read_csv(path, dtype={"ID": str, "source_id": str})
        assert not labels[split].ID.duplicated().any()
    assert not set(labels["train"].source_id) & set(labels["validation"].source_id)
    assert not set(labels["train"].ID) & set(labels["validation"].ID)
    features = torch.load(args.run_dir / "features.pt", map_location="cpu", weights_only=True)
    for row in pd.concat(list(labels.values())).itertuples():
        assert features[row.ID].shape == (row.frames, 512)
        assert torch.isfinite(features[row.ID]).all()
    # Load the actual packaged class to check structural and prediction compatibility.
    import ast, zipfile
    from torch import nn
    with zipfile.ZipFile(args.base_zip) as archive:
        tree = ast.parse(archive.read("inference.py").decode("utf-8"))
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_Stage2Temporal"]
        assert len(classes) == 1
        namespace = {"torch": torch, "nn": nn}
        exec(compile(ast.Module(body=classes, type_ignores=[]), "<packaged Stage2Temporal>", "exec"), namespace)
        import io
        packaged_backbone = torch.load(io.BytesIO(archive.read("model/stage2/resnet18-f37072fd.pth")), map_location="cpu", weights_only=True)
        training_backbone = torch.load(args.split_dir / "training/model/stage2/resnet18-f37072fd.pth", map_location="cpu", weights_only=True)
        assert packaged_backbone.keys() == training_backbone.keys(), "Backbone keys differ"
        assert all(torch.equal(packaged_backbone[k], training_backbone[k]) for k in packaged_backbone), "Backbone tensors differ"
    result = dict(status="PASS", scope="Cached-feature development validation; not official score or full GPU integration",
                  split_sha256=report["split_sha256"], features_sha256=sha(args.run_dir/"features.pt"),
                  validation_rows=len(labels["validation"]), validation_sources=labels["validation"].source_id.nunique(),
                  candidates=[])
    truth = labels["validation"].set_index("ID")
    for candidate in report["candidates"]:
        folder = args.run_dir / candidate["name"]
        pred = pd.read_csv(folder / "validation_predictions.csv", dtype={"ID": str})
        assert list(pred.columns) == p.OUTPUT_COLUMNS and not pred.isna().any().any()
        assert not pred.ID.duplicated().any() and set(pred.ID) == set(truth.index)
        pred = pred.set_index("ID").loc[truth.index]
        for col in ("collision_frame", "entry_frame"):
            assert np.equal(pred[col], np.floor(pred[col])).all()
            assert ((pred[col] >= 0) & (pred[col] < truth.frames)).all()
        assert set(pred.evasion_space) <= {0, 1} and set(pred.entry_side) <= {"LEFT", "RIGHT"}
        metrics = {}
        for col, target in (("collision_frame", "t_collision"), ("entry_frame", "t_entry")):
            error = (pred[col] - truth[target]).abs() / truth.fps
            metrics[col + "_within_0.3s"] = float((error <= .3 + 1e-9).mean())
            metrics[col + "_mae_seconds"] = float(error.mean())
        for col in ("evasion_space", "entry_side"):
            metrics[col + "_accuracy"] = float((pred[col] == truth[col]).mean())
        metrics["development_mean"] = float(np.mean([metrics[k] for k in
            ("collision_frame_within_0.3s", "entry_frame_within_0.3s", "evasion_space_accuracy", "entry_side_accuracy")]))
        for key, value in metrics.items():
            assert abs(value - candidate["selected_metrics"][key]) < 1e-10
        state = torch.load(folder/"best.pt", map_location="cpu", weights_only=True)["model"]
        assert all(torch.isfinite(v).all() for v in state.values())
        local, packaged = p.Stage2Temporal().eval(), namespace["_Stage2Temporal"]().eval()
        local.load_state_dict(state, strict=True)
        packaged.load_state_dict(state, strict=True)
        with torch.inference_mode():
            for sid in truth.index:
                x = features[sid].unsqueeze(0)
                c, e, scene = local(x)
                pc, pe, ps = packaged(x)
                assert torch.equal(c, pc) and torch.equal(e, pe) and torch.equal(scene, ps)
                expected = pred.loc[sid]
                assert int(c) == expected.collision_frame and int(e) == expected.entry_frame
                assert int(scene[0,:2].argmax()) == expected.evasion_space
                assert ["LEFT","RIGHT"][int(scene[0,2:].argmax())] == expected.entry_side
        result["candidates"].append(dict(name=candidate["name"], metrics=metrics,
            checkpoint_sha256=sha(folder/"best.pt"), cached_predictions_match=True,
            packaged_model_outputs_match=True, epochs_observed=len(candidate["history"])))
    selected = max(result["candidates"], key=lambda c:c["metrics"]["development_mean"])["name"]
    assert selected == report["selected"]
    result["selected"] = selected
    result["reported_seconds"] = report["seconds"]
    result["budget_caveat"] = "Historical report says COMPLETED but last candidate has fewer than 25 epochs; elapsed wall time exceeds configured cap. Completion/budget not certified."
    (args.run_dir/"independent-validation.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
