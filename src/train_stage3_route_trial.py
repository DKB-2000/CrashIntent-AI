"""Bounded Stage3 scratch FP32 trial with balanced train clips and dense held-out routes."""
import argparse,hashlib,json,random,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import stage3_pipeline as p
from diagnose_stage3_precision import disable_regularization
from diagnose_stage3_optimization import accumulated_backward

def choose_videos(records):
    rng=np.random.default_rng(p.SEED)
    routes=defaultdict(list)
    for i,row in enumerate(records):routes[row["route"]].append(i)
    chosen=set()
    for route in sorted(routes):
        remaining=routes[route].copy();rng.shuffle(remaining);covered=set()
        def signatures(i):
            r=records[i]
            return {(int(r["accel"][t]),int(r["steer"][t]) if r["accel"][t]!=3 else None)
                    for t in range(15,len(r["accel"]),16)}
        for _ in range(min(2,len(remaining))):
            i=max(remaining,key=lambda i:len(signatures(i)-covered))
            remaining.remove(i);chosen.add(i);covered.update(signatures(i))
    return chosen

def select_training(records,per_group=20,all_train_videos=False):
    """Round robin over train routes; never inspect validation labels to select training."""
    rng=np.random.default_rng(p.SEED)
    groups=[(a,s) for a in range(3) for s in range(3)]+[(3,None)]
    selected=[];route_counts=Counter();id_counts=Counter();eligible=set(range(len(records))) if all_train_videos else choose_videos(records)
    for group,(a,s) in enumerate(groups):
        pool=[]
        for ri,row in enumerate(records):
            if ri not in eligible:continue
            for end in range(15,len(row["accel"]),16):
                if row["accel"][end]==a and (s is None or row["steer"][end]==s):
                    pool.append(dict(record=ri,ID=row["ID"],route=row["route"],endpoint=end,
                        accel=int(a),steer=int(row["steer"][end]),group=group))
        if len(pool)<per_group:raise ValueError(f"Insufficient training group {group}")
        rng.shuffle(pool)
        for _ in range(per_group):
            index=min(range(len(pool)),key=lambda i:(route_counts[pool[i]["route"]],id_counts[pool[i]["ID"]]))
            row=pool.pop(index);selected.append(row);route_counts[row["route"]]+=1;id_counts[row["ID"]]+=1
    return selected

def balanced_batches(samples,epoch):
    rng=np.random.default_rng(p.SEED+epoch);groups=defaultdict(list)
    for i,row in enumerate(samples):groups[row["group"]].append(i)
    if set(groups)!=set(range(10)) or len({len(v) for v in groups.values()})!=1:
        raise ValueError("Expected ten equally sized groups")
    for values in groups.values():rng.shuffle(values)
    for k in range(len(groups[0])):
        batch=[groups[g][k] for g in range(10)];rng.shuffle(batch);yield batch

def score_frame(frame,truth_a,truth_s):
    a=frame.accel_label.map(dict(zip(p.ACCEL,range(4)))).to_numpy()
    s=frame.steer_label.map(dict(zip(p.STEER,range(3)))).to_numpy()
    ta=np.asarray(truth_a);ts=np.asarray(truth_s);moving=ta!=3
    return dict(accel=p.classification_metrics(ta,a,p.ACCEL),
        moving_steer=p.classification_metrics(ts[moving],s[moving],p.STEER),
        accel_prediction_counts=frame.accel_label.value_counts().to_dict(),
        steer_prediction_counts=frame.steer_label.value_counts().to_dict(),
        moving_steer_prediction_counts=frame.loc[moving,"steer_label"].value_counts().to_dict(),
        stopped_excluded=int((~moving).sum()))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--max-seconds",type=float,default=2940)
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--per-group",type=int,default=20)
    parser.add_argument("--all-train-videos",action="store_true")
    parser.add_argument("--initial-checkpoint",type=Path)
    parser.add_argument("--reference-report",type=Path)
    args=parser.parse_args()
    if bool(args.initial_checkpoint)!=bool(args.reference_report):parser.error("Checkpoint and reference report must be supplied together")
    if not 0<args.max_seconds<=6900 or args.epochs<1 or not 1<=args.per_group<=100:parser.error("Invalid budget")
    start=time.monotonic();deadline=start+args.max_seconds
    torch.set_num_threads(4);p.cv2.setNumThreads(1);device=p.device_for("cuda")
    random.seed(p.SEED);np.random.seed(p.SEED);torch.manual_seed(p.SEED)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    records=p.load_records(args.dataset_dir)
    samples=select_training(records["train"],args.per_group,args.all_train_videos)
    assert not {r["route"] for r in samples}&{r["route"] for r in records["validation"]}
    manifest_names=("split_manifest.csv","labels_train_candidate.csv","labels_validation_candidate.csv")
    fingerprints={n:hashlib.sha256((args.dataset_dir/n).read_bytes()).hexdigest() for n in manifest_names}
    report=dict(status="RUNNING",scope="Small balanced training set; original complete validation routes; provisional external labels",
        config=dict(seed=p.SEED,initialization="checkpoint_weights_optimizer_reset" if args.initial_checkpoint else "scratch",precision="fp32",learning_rate=3e-5,
            effective_batch=10,microbatch=2,weight_decay=0.,epochs_requested=args.epochs,per_group=args.per_group,all_train_videos=args.all_train_videos),
        samples=samples,data_sha256=fingerprints,history=[],optimizer_steps=0,samples_seen=0,
        planned_validation_ids=[r["ID"] for r in records["validation"]],gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,validation_complete=False)
    def save():
        report["seconds"]=time.monotonic()-start
        (args.output_dir/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    save();clips=torch.empty((len(samples),3,16,224,224),dtype=torch.uint8)
    for ri in sorted({r["record"] for r in samples}):
        if time.monotonic()>start+(1200 if args.all_train_videos else 500):raise TimeoutError("Decode budget exceeded")
        frames=list(p.video_frames(records["train"][ri]["path"]))
        for i,row in enumerate(samples):
            if row["record"]==ri:
                clips[i]=torch.from_numpy(np.stack([frames[t] for t in p.causal_indices(row["endpoint"])]).copy()).permute(1,0,2,3).contiguous()
        print("cached",records["train"][ri]["ID"],flush=True)
    del frames
    digest=hashlib.sha256()
    for clip in clips:digest.update(clip.numpy().tobytes())
    report["training_input_sha256"]=digest.hexdigest()
    ta=torch.tensor([r["accel"] for r in samples],device=device)
    ts=torch.tensor([r["steer"] for r in samples],device=device)
    def normalized(ids):return (clips[ids].float()/255-.45)/.225
    if args.initial_checkpoint:
        reference_report=json.loads(args.reference_report.read_text())
        initial_hash=hashlib.sha256(args.initial_checkpoint.read_bytes()).hexdigest()
        assert initial_hash==reference_report["checkpoint_sha256"]
        assert report["training_input_sha256"]==reference_report["training_input_sha256"]
        assert samples==reference_report["samples"] and fingerprints==reference_report["data_sha256"]
        model=p.load_model(args.initial_checkpoint,device)
        report["initial_checkpoint_sha256"]=initial_hash
        report["optimizer_reset"]=True
    else:model=p.Stage3MViT().to(device)
    report["disabled_modules"]=disable_regularization(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=0.)
    def measure_training(epoch):
        model.eval();outputs=[]
        with torch.inference_mode():
            for offset in range(0,len(samples),2):
                a,s=model(normalized(slice(offset,offset+2)).to(device))
                if not torch.isfinite(a).all() or not torch.isfinite(s).all():raise ValueError("Nonfinite logits")
                outputs.extend(dict(ID=samples[i]["ID"],sample_index=samples[i]["endpoint"],
                    accel_label=p.ACCEL[ai],steer_label=p.STEER[si])
                    for i,ai,si in zip(range(offset,min(offset+2,len(samples))),a.argmax(1).tolist(),s.argmax(1).tolist()))
        frame=pd.DataFrame(outputs,columns=p.OUTPUT_COLUMNS)
        metrics=score_frame(frame,ta.cpu().numpy(),ts.cpu().numpy())
        frame.to_csv(args.output_dir/"train_predictions.csv",index=False)
        report["train_metrics"]=metrics
        print("train_evaluation",epoch,json.dumps(metrics),flush=True)
        return metrics
    report["initial_train_metrics"]=measure_training(0)
    if args.initial_checkpoint:assert report["initial_train_metrics"]==reference_report["train_metrics"]
    save()
    train_deadline=start+(min(4800,args.max_seconds-1800) if args.all_train_videos else min(1150,args.max_seconds*.40))
    for epoch in range(1,args.epochs+1):
        losses=[];completed=True;model.train()
        for ids in balanced_batches(samples,epoch):
            if time.monotonic()>=train_deadline:completed=False;break
            index=torch.tensor(ids,device=device);optimizer.zero_grad(set_to_none=True)
            loss=accumulated_backward(model,normalized(ids),ta[index],ts[index])
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();losses.append(loss);report["optimizer_steps"]+=1;report["samples_seen"]+=len(ids)
        row=dict(epoch=epoch,epoch_complete=completed,updates=len(losses),mean_loss=float(np.mean(losses)) if losses else None)
        report["history"].append(row);save();print("epoch",json.dumps(row),flush=True)
        if not completed:break
    report["train_metrics"]=measure_training(report["history"][-1]["epoch"])
    checkpoint=args.output_dir/"best.pt"
    # Fixed final checkpoint; no validation-driven selection in this trial.
    p.save_checkpoint(model,checkpoint,dict(experiment="route-trial",config=report["config"],data_sha256=fingerprints))
    reference_x=normalized(slice(0,2)).to(device)
    with torch.inference_mode():reference=tuple(v.detach().cpu() for v in model(reference_x))
    del model,optimizer;torch.cuda.empty_cache()
    model=p.load_model(checkpoint,device)
    with torch.inference_mode():actual=tuple(v.detach().cpu() for v in model(reference_x))
    for x,y in zip(reference,actual):torch.testing.assert_close(x,y,rtol=0,atol=0)
    report["reload_logits_exact"]=True
    report["checkpoint_sha256"]=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    parts=[];va=[];vs=[]
    for row in records["validation"]:
        if time.monotonic()>=deadline-100:break
        prediction=p.predict_video(model,row["path"],row["ID"],device,4)
        if len(prediction)!=len(row["accel"]):raise ValueError("Validation count differs")
        parts.append(prediction);va.extend(row["accel"]);vs.extend(row["steer"])
        print("validation",row["ID"],len(prediction),"elapsed",time.monotonic()-start,flush=True)
    if parts:
        predictions=pd.concat(parts,ignore_index=True)
        predictions.to_csv(args.output_dir/"validation_predictions.csv",index=False)
        report["validation"]=score_frame(predictions,va,vs)
        baseline=predictions.copy();baseline["accel_label"]="CONSTANT";baseline["steer_label"]="STRAIGHT"
        report["constant_baseline"]=score_frame(baseline,va,vs)
        report["validation_complete"]=len(parts)==len(records["validation"])
    report["status"]="COMPLETED" if report["validation_complete"] and all(h["epoch_complete"] for h in report["history"]) and len(report["history"])==args.epochs else "PARTIAL"
    save();print("RESULT",json.dumps({k:v for k,v in report.items() if k not in {"samples","history"}}),flush=True)
if __name__=="__main__":main()
