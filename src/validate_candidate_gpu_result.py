"""Validate GPU integration evidence against the exact candidate and public fixtures."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile
import cv2
from daily_submission import inspect_zip

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(value, message):
    if not value:
        raise ValueError(message)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-zip",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--fixtures",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    digest=inspect_zip(args.candidate)
    with zipfile.ZipFile(args.result_zip) as result, zipfile.ZipFile(args.candidate) as candidate, zipfile.ZipFile(args.fixtures) as fixtures:
        require(result.testzip() is None and fixtures.testzip() is None,"CRC failure")
        manifest=json.loads(result.read("candidate-assets.json"))
        require(manifest=={"candidate.bin":digest,"public-fixtures.bin":sha(args.fixtures)},"Evidence input hashes differ")
        report=json.loads(result.read("integration.json"))
        install=json.loads(result.read("install.json"))
        require(report["status"]=="PASS" and report["python_internet_socket_blocked"] is True,"GPU run did not pass")
        require(install["exit_code"]==0 and 0<=install["seconds"]<=600,"Install failed or exceeded budget")
        require(install["requirements_sha256"]==hashlib.sha256(candidate.read("requirements.txt")).hexdigest(),"Requirements hash differs")
        for line in candidate.read("requirements.txt").decode().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            name,version=line.split("==")
            require(report["versions"][name].split("+")[0]==version,"Installed version differs")
        require("T4" in report["gpu"],"Unexpected GPU")
        require(sum(v["seconds"] for v in report["results"].values())<=2400,"Inference exceeded check budget")
        contracts={
            "stage1":["ID","answer"],
            "stage2":["ID","collision_frame","entry_frame","evasion_space","entry_side"],
            "stage3":["ID","sample_index","accel_label","steer_label"],
        }
        checked={}
        with tempfile.TemporaryDirectory(prefix="candidate-fixture-check-") as temporary:
            for stage,columns in contracts.items():
                reader=csv.DictReader(io.StringIO(result.read(stage+".csv").decode("utf-8-sig")))
                require(reader.fieldnames==columns,"Output columns differ")
                rows=list(reader)
                require(rows and all(all(row[c] not in ("",None) for c in columns) for row in rows),"Missing values")
                names=[n for n in fixtures.namelist() if n.startswith(stage+"/videos/") and n.endswith(".mp4")]
                counts={}
                for name in names:
                    sid=Path(name).stem
                    require(sid not in counts,"Duplicate fixture ID")
                    if stage=="stage1":
                        counts[sid]=1
                    else:
                        video=Path(temporary)/(stage+"-"+Path(name).name)
                        video.write_bytes(fixtures.read(name))
                        cap=cv2.VideoCapture(str(video))
                        require(cap.isOpened(),"Fixture not decodable")
                        count=0
                        while cap.read()[0]:
                            count+=1
                        cap.release()
                        require(count>0,"Empty fixture")
                        counts[sid]=count
                require(set(r["ID"] for r in rows)==set(counts),"Output IDs differ")
                require(len(rows)==report["results"][stage]["rows"],"Report row count differs")
                if stage=="stage1":
                    require(len(rows)==len(counts) and all(r["answer"] in {"ORIGINAL","RERECORDED"} for r in rows),"Invalid Stage1 output")
                elif stage=="stage2":
                    require(len(rows)==len(counts),"Duplicate Stage2 IDs")
                    for row in rows:
                        require(row["evasion_space"] in {"0","1"} and row["entry_side"] in {"LEFT","RIGHT"},"Invalid Stage2 category")
                        require(all(0<=int(row[c])<counts[row["ID"]] for c in ("entry_frame","collision_frame")),"Frame out of bounds")
                else:
                    for sid,count in counts.items():
                        group=[r for r in rows if r["ID"]==sid]
                        require([int(r["sample_index"]) for r in group]==list(range(count)),"Stage3 sample indices differ")
                    require(all(r["accel_label"] in {"ACCELERATING","DECELERATING","CONSTANT","STOPPED"} and r["steer_label"] in {"LEFT","STRAIGHT","RIGHT"} for r in rows),"Invalid Stage3 category")
                checked[stage]=dict(rows=len(rows),video_frames=counts,seconds=report["results"][stage]["seconds"])
        validation=dict(status="PASS",scope="Exact candidate, pinned environment, GPU public fixtures and output contracts; not hidden-set score or runtime guarantee",
            candidate_sha256=digest,result_zip_sha256=sha(args.result_zip),gpu=report["gpu"],
            install_seconds=install["seconds"],results=checked,submitted=False)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(validation,indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps(validation,indent=2))

if __name__=="__main__":
    main()
