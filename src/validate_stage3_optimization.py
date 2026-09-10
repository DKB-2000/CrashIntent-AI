"""Verify fixed-sample Stage3 optimization results against local inputs and source."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(test,message):
    if not test:raise ValueError(message)
def ce(logits,targets):
    logits=np.asarray(logits,dtype=np.float64)
    shifted=logits-logits.max(1,keepdims=True)
    return np.mean(np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(targets)),targets])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-zip",type=Path,required=True)
    parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args()
    with zipfile.ZipFile(args.result_zip) as z:
        require(z.testzip() is None,"ZIP CRC failure")
        result=json.loads(z.read("diagnostic-run/optimization.json"))
        install=json.loads(z.read("install.json"))
        source=json.loads(z.read("optimization-assets.json"))
    require(source==json.loads((args.run_dir/"dataset/optimization-assets.json").read_text()),"Wrong source manifest")
    for name,digest in source.items():
        require(sha(args.run_dir/"dataset"/name)==digest,"Source file changed")
    inputs=json.loads((args.run_dir/"input-validation.json").read_text())
    require(result["samples"]==inputs["samples"],"Different diagnostic samples")
    require(result["input_sha256"]==inputs["input_sha256"],"Different input tensor bytes")
    require(result["input_shape"]==[10,3,16,224,224],"Wrong input shape")
    require(result["checkpoint_sha256"]==sha(Path("artifacts/kaggle-stage3-precision/dataset/diagnostic-best.pt")),"Wrong checkpoint")
    require(result["label_sha256"]==sha(Path("artifacts/stage3-comma-chunk1-calibrated-v1/labels_train_candidate.csv")),"Different training labels")
    require(install["exit_code"]==0 and install["seconds"]<=600,"Install failed or over budget")
    ta=np.array([r["accel"] for r in result["samples"]])
    ts=np.array([r["steer"] for r in result["samples"]]);moving=ta!=3
    specs=[("scratch_b2_lr1e4","scratch",2,1e-4),("scratch_b10_lr1e4","scratch",10,1e-4),
           ("scratch_b10_lr3e5","scratch",10,3e-5),("trained_b10_lr3e5","trained",10,3e-5)]
    require(0<len(result["runs"])<=4,"Unexpected run count")
    runs=[]
    for run,spec in zip(result["runs"],specs):
        require((run["name"],run["initialization"],run["effective_batch"],run["lr"])==spec,"Run configuration differs")
        require(run["precision"]=="fp32" and run["microbatch"]==2,"Precision or microbatch differs")
        require(run["samples_seen"]==run["optimizer_steps"]*run["effective_batch"],"Sample accounting differs")
        require(run["history"][0]["step"]==0,"Missing initial measurement")
        for h in run["history"]:
            aa=np.asarray(h["accel_logits"]);ss=np.asarray(h["steer_logits"])
            require(aa.shape==(10,4) and ss.shape==(10,3),"Logit shape differs")
            require(np.isfinite(aa).all() and np.isfinite(ss).all(),"Nonfinite logits")
            require(aa.argmax(1).tolist()==h["accel_predictions"] and ss.argmax(1).tolist()==h["steer_predictions"],"Argmax mismatch")
            accuracy_a=float((aa.argmax(1)==ta).mean());accuracy_s=float((ss.argmax(1)[moving]==ts[moving]).mean())
            require(abs(accuracy_a-h["accel_accuracy"])<1e-6 and abs(accuracy_s-h["steer_accuracy"])<1e-6,"Accuracy mismatch")
            require(abs(ce(aa,ta)+ce(ss[moving],ts[moving])-h["loss"])<1e-5,"Loss mismatch")
            require(h["samples_seen"]==h["step"]*run["effective_batch"],"History sample count differs")
        last=run["history"][-1]
        require(last["step"]==run["optimizer_steps"],"Missing final measurement")
        if run["status"]=="MEMORIZED":
            require(last["accel_accuracy"]==last["steer_accuracy"]==1.,"False memorization claim")
        require(all(np.isfinite(v) for v in run["weight_changes"].values()),"Nonfinite weights")
        runs.append(dict(name=run["name"],status=run["status"],seconds=run["seconds"],
            steps=run["optimizer_steps"],samples_seen=run["samples_seen"],initial_loss=run["history"][0]["loss"],
            final_loss=last["loss"],accel_accuracy=last["accel_accuracy"],steer_accuracy=last["steer_accuracy"]))
    if result["status"]=="COMPLETED":
        require(len(runs)==4 and all(r["status"] in {"MEMORIZED","STEP_LIMIT"} for r in runs),"False complete claim")
    summary=dict(status="PASS",scope="Training diagnostic integrity; not held-out performance",
        experiment_status=result["status"],input_sha256=result["input_sha256"],result_sha256=sha(args.result_zip),
        seconds=result["seconds"],install_seconds=install["seconds"],runs=runs)
    (args.run_dir/"result-validation.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
