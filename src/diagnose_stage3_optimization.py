"""Training-only optimization controls on ten fixed Stage3 clips."""
import argparse,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import stage3_pipeline as p
from diagnose_stage3_precision import select_samples,disable_regularization

def accumulated_backward(model, clips, ta, ts, microbatch=2):
    """Exactly the full-batch loss: accel/N + moving steer/Nmoving."""
    moving_count=int((ta!=3).sum())
    total=0.
    for start in range(0,len(clips),microbatch):
        end=start+microbatch
        a,s=model(clips[start:end].to(ta.device))
        valid=ta[start:end]!=3
        loss=F.cross_entropy(a,ta[start:end],reduction="sum")/len(clips)
        loss=loss+(F.cross_entropy(s[valid],ts[start:end][valid],reduction="sum")/moving_count if valid.any() else s.sum()*0)
        if not torch.isfinite(loss):raise ValueError("Nonfinite loss")
        loss.backward();total+=float(loss.detach())
    return total

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=120)
    parser.add_argument("--max-seconds",type=float,default=2400)
    args=parser.parse_args()
    if not 0<args.max_seconds<=2700 or args.steps<1:parser.error("Invalid budget")
    started=time.monotonic();args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);p.cv2.setNumThreads(1);device=p.device_for("cuda")
    records=p.load_records(args.dataset_dir)["train"];samples=select_samples(records)
    clips=[]
    for row in samples:
        frames=list(p.video_frames(records[row["record"]]["path"]))
        clips.append(p.clip_tensor([frames[t] for t in p.causal_indices(row["endpoint"])]))
    clips=torch.stack(clips);del frames
    ta=torch.tensor([r["accel"] for r in samples],device=device)
    ts=torch.tensor([r["steer"] for r in samples],device=device)
    specs=[("scratch_b2_lr1e4","scratch",2,1e-4),("scratch_b10_lr1e4","scratch",10,1e-4),
           ("scratch_b10_lr3e5","scratch",10,3e-5),("trained_b10_lr3e5","trained",10,3e-5)]
    result=dict(status="RUNNING",scope="Fixed training clips only; not generalization or submission",
        samples=samples,input_shape=list(clips.shape),input_sha256=hashlib.sha256(clips.numpy().tobytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        label_sha256=hashlib.sha256((args.dataset_dir/"labels_train_candidate.csv").read_bytes()).hexdigest(),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),runs=[])
    def save():
        result["seconds"]=time.monotonic()-started
        (args.output_dir/"optimization.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    for name,initialization,batch,lr in specs:
        if time.monotonic()-started>=args.max_seconds:break
        random.seed(p.SEED);np.random.seed(p.SEED);torch.manual_seed(p.SEED)
        model=p.Stage3MViT().to(device) if initialization=="scratch" else p.load_model(args.checkpoint,device)
        disabled=disable_regularization(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
        tracked={"backbone":next(model.backbone.parameters()),"accel":model.accel.weight,"steer":model.steer.weight}
        before={k:v.detach().clone() for k,v in tracked.items()}
        run=dict(name=name,initialization=initialization,effective_batch=batch,microbatch=2,lr=lr,
            precision="fp32",disabled_modules=disabled,optimizer_steps=0,samples_seen=0,history=[],gradient_norms=[],status="RUNNING")
        result["runs"].append(run);rng=np.random.default_rng(p.SEED);run_start=time.monotonic()
        def measure():
            model.eval()
            with torch.inference_mode():
                values=[model(x.to(device)) for x in clips.split(2)]
                a=torch.cat([v[0] for v in values]);s=torch.cat([v[1] for v in values])
                if not torch.isfinite(a).all() or not torch.isfinite(s).all():raise ValueError("Nonfinite logits")
                loss=float(p.multitask_loss((a,s),ta,ts));moving=ta!=3
                row=dict(step=run["optimizer_steps"],samples_seen=run["samples_seen"],loss=loss,
                    accel_accuracy=float((a.argmax(1)==ta).float().mean()),
                    steer_accuracy=float((s.argmax(1)[moving]==ts[moving]).float().mean()),
                    accel_predictions=a.argmax(1).tolist(),steer_predictions=s.argmax(1).tolist(),
                    accel_logits=a.tolist(),steer_logits=s.tolist())
            run["history"].append(row);save()
            print(name,"step",row["step"],"seen",row["samples_seen"],"loss",loss,
                  "accuracy",row["accel_accuracy"],row["steer_accuracy"],flush=True)
            return row["accel_accuracy"]==1. and row["steer_accuracy"]==1.
        measure()
        try:
            for step in range(1,args.steps+1):
                if time.monotonic()-started>=args.max_seconds or time.monotonic()-run_start>=600:
                    run["status"]="TIME_LIMIT";break
                ids=np.arange(10) if batch==10 else rng.choice(10,size=2,replace=False)
                idx=torch.as_tensor(ids,device=device)
                model.train();optimizer.zero_grad(set_to_none=True)
                accumulated_backward(model,clips[ids],ta[idx],ts[idx])
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();run["optimizer_steps"]+=1;run["samples_seen"]+=len(ids)
                if step==1 or step%10==0:run["gradient_norms"].append(dict(step=step,before_clip=float(norm)))
                if step%20==0 or step==args.steps:
                    if measure():run["status"]="MEMORIZED";break
            else:run["status"]="STEP_LIMIT"
            if run["history"][-1]["step"]!=run["optimizer_steps"]:measure()
        except (RuntimeError,ValueError) as error:
            run["status"]="ERROR";run["error"]=str(error)
        run["seconds"]=time.monotonic()-run_start
        run["weight_changes"]={k:float((v.detach()-before[k]).norm()) for k,v in tracked.items()}
        save();del model,optimizer,tracked,before;torch.cuda.empty_cache()
    result["status"]="COMPLETED" if len(result["runs"])==4 and all(r["status"] in {"MEMORIZED","STEP_LIMIT"} for r in result["runs"]) else "PARTIAL"
    save()
if __name__=="__main__":main()
