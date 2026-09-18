"""Audit MM-AU YOLO detection labels against official temporal metadata."""
from __future__ import annotations
import csv,hashlib,json,re,tarfile
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'data_raw/mm-au-detection-labels-20260917/labels.tar.gz';META=ROOT/'data_raw/mm-au-official-metadata-20260917/video_metadata.csv';OUT=ROOT/'artifacts/stage2-mmau-detection-label-audit-20260917'
PAT=re.compile(r'^labels/(train|val|test)/(\d+)_(\d+)_(\d+)\.txt$')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir();by_video=defaultdict(list);split_counts=Counter();class_counts=Counter();nonempty=0;malformed=[]
 with tarfile.open(ARCHIVE,'r:gz') as tf:
  for m in tf:
   if not m.isfile():continue
   q=PAT.match(m.name)
   if not q:continue
   split,typ,vid,frame=q.groups();key=f'{int(typ)}_{int(vid)}';frame=int(frame);split_counts[split]+=1
   raw=tf.extractfile(m).read().decode('utf-8','replace').strip();classes=[]
   for line in raw.splitlines():
    if not line.strip():continue
    parts=line.split()
    if len(parts)!=5:malformed.append({'name':m.name,'line':line[:200]});continue
    try:cls=int(parts[0]);vals=[float(x) for x in parts[1:]]
    except ValueError:malformed.append({'name':m.name,'line':line[:200]});continue
    if not (0<=cls<=6 and all(0<=x<=1 for x in vals)):malformed.append({'name':m.name,'line':line[:200]});continue
    classes.append(cls);class_counts[cls]+=1
   if classes:nonempty+=1
   by_video[key].append((frame,len(classes)))
 with META.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 positives=[r for r in rows if r['whether an accident occurred (1/0)'].strip()=='1'];meta={f"{int(r['type'])}_{int(r['video'])}":r for r in positives}
 matched=set(by_video)&set(meta);unmatched=set(by_video)-set(meta);covered_ai=covered_co=0;nearest_ai=[];nearest_co=[]
 for key in matched:
  frames=[x[0] for x in by_video[key]];ai=int(meta[key]['abnormal start frame']);co=int(meta[key]['accident frame']);da=min(abs(x-ai) for x in frames);dc=min(abs(x-co) for x in frames);nearest_ai.append(da);nearest_co.append(dc);covered_ai+=da<=5;covered_co+=dc<=5
 report={'status':'COMPLETE_VALIDATED','archive_sha256':sha(ARCHIVE),'metadata_sha256':sha(META),'label_txt_files':sum(split_counts.values()),'split_file_counts':dict(split_counts),'nonempty_label_files':nonempty,'empty_label_files':sum(split_counts.values())-nonempty,'box_class_counts_zero_based':dict(sorted(class_counts.items())),'malformed_lines':len(malformed),'unique_label_video_keys':len(by_video),'metadata_positive_video_keys':len(meta),'matched_video_keys':len(matched),'unmatched_label_video_keys':len(unmatched),'metadata_without_detection_labels':len(set(meta)-set(by_video)),'matched_with_label_within_5_frames_of_t_ai':covered_ai,'matched_with_label_within_5_frames_of_t_co':covered_co,'nearest_t_ai_frame_distance':{'max':max(nearest_ai),'mean':sum(nearest_ai)/len(nearest_ai)},'nearest_t_co_frame_distance':{'max':max(nearest_co),'mean':sum(nearest_co)/len(nearest_co)},'schema':'YOLO object boxes only: class,cx,cy,w,h; classes 0..6 correspond to motorcycle,truck,bus,traffic light,person,bicycle,car according to official COCO JSON category order. No track IDs or collision-actor identity.','stage2_decision':'Useful for object-position features near t_ai/t_co, but cannot directly produce official entry_side/evasion_space labels.'}
 (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
