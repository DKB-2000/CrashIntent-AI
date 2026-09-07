"""Convert accepted comma2k19 ZIP segments into a combined Stage3 dataset."""
from __future__ import annotations
import argparse, json, re, subprocess, zipfile, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from extract_stage3_comma_preview import extract

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--archive",type=Path,required=True); ap.add_argument("--audit-csv",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--limit",type=int); ap.add_argument("--start",type=int,default=0); ap.add_argument("--workers",type=int,default=1); ap.add_argument("--keep-segments",action="store_true"); a=ap.parse_args()
    root=a.output_dir.resolve(); root.mkdir(parents=True,exist_ok=True); work=root/"segments"; work.mkdir(parents=True,exist_ok=True); labels=[]; manifest=[]
    rows=pd.read_csv(a.audit_csv).sort_values(["route","segment"]).to_dict("records")
    if a.limit: rows=rows[a.start:a.start+a.limit]
    else: rows=rows[a.start:]
    def one(item):
        i,row=item; seg=str(row["segment"]); sid=f"COMMA2K19_C1_{i:04d}"; segdir=work/sid; out=root/"converted"/sid
        if not (out/"conversion_report.json").is_file():
            if segdir.exists():
                prior=json.loads((segdir/"source_manifest.json").read_text(encoding="utf-8"))
                if prior["segment"].rstrip("/") != seg.rstrip("/"):
                    raise ValueError(f"Source mismatch for {sid}")
            else:
                extract(a.archive,seg,segdir)
            cmd=[sys.executable, "src/prepare_stage3_comma2k19.py", "--segment-dir", str(segdir), "--output-dir", str(out), "--id", sid, "--steer-offset", str(row["steering_median"])]
            out.mkdir(parents=True,exist_ok=True)
            with (out/"conversion.log").open("w",encoding="utf-8") as log:
                subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        frame=pd.read_csv(out/"labels.csv")
        report=json.loads((out/"conversion_report.json").read_text(encoding="utf-8"))
        if report["status"] != "PASS" or report["id"] != sid or len(frame) != report["output_frames"] or not (out/"videos"/f"{sid}.mp4").is_file():
            raise ValueError(f"Incomplete conversion for {sid}")
        print(f"[{i+1}] {sid}: {len(frame)} samples",flush=True)
        return frame,dict(ID=sid,source_segment=seg,source_route=row["route"],samples=len(frame),steer_offset=float(row["steering_median"]))
    with ThreadPoolExecutor(max_workers=max(1,a.workers)) as pool:
        results=list(pool.map(one, enumerate(rows,start=a.start)))
    labels=[x[0] for x in results]; manifest=[x[1] for x in results]
    combined=pd.concat(labels,ignore_index=True); combined.to_csv(root/"labels.csv",index=False); (root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(dict(status="PASS",segments=len(manifest),samples=len(combined),routes=len({x["source_route"] for x in manifest}),labels=str(root/"labels.csv")),indent=2))
if __name__=="__main__": main()
