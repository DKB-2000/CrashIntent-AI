"""Audit official MM-AU and Nexar metadata for Stage2 temporal-label use."""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MMAU=ROOT/'data_raw/mm-au-official-metadata-20260917/video_metadata.csv'
NEXAR=ROOT/'data_raw/stage2-validation-nexar-20260914/metadata.csv'
OUT=ROOT/'artifacts/stage2-labeled-dataset-audit-v2-20260917'

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def integer(v):return int(str(v).strip())

def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True)
 with MMAU.open(encoding='utf-8-sig',newline='') as f:mm=list(csv.DictReader(f))
 with NEXAR.open(encoding='utf-8-sig',newline='') as f:nx=list(csv.DictReader(f))
 mm_norm=[];bad=[]
 for r in mm:
  try:
   ai=integer(r['abnormal start frame']);co=integer(r['accident frame']);ae=integer(r['abnormal end frame']);total=integer(r['total frames']);positive=integer(r['whether an accident occurred (1/0)'])
  except Exception as exc:bad.append({'hashcode':r.get('hashcode'),'error':repr(exc)});continue
  ordered=0<=ai<=co<=ae<=total
  if positive==1 and ordered:
   mm_norm.append({'dataset':'MM-AU','video_id':r['hashcode'],'video_name':r['VideoName'],'source_group':r['video'],
    'fps':'','frames':total,'entry_frame_proxy':ai,'collision_frame':co,'entry_time_seconds':'','collision_time_seconds':'',
    'accident_type':r['type'],'scene':r['scenes'],'label_origin':'official_t_ai_t_co','temporal_mapping':'t_ai_proxy_to_entry;t_co_to_collision','entry_side':'','evasion_space':''})
  elif positive==1:bad.append({'hashcode':r.get('hashcode'),'error':'invalid temporal ordering'})
 nx_norm=[];nx_bad=[]
 for r in nx:
  try:alert=float(r['time_of_alert']);event=float(r['time_of_event'])
  except Exception as exc:nx_bad.append({'file_name':r.get('file_name'),'error':repr(exc)});continue
  if not (0<=alert<=event):nx_bad.append({'file_name':r.get('file_name'),'error':'invalid time ordering'});continue
  nx_norm.append({'dataset':'Nexar','video_id':Path(r['file_name']).stem,'video_name':r['file_name'],'source_group':'nexar_train_positive',
   'fps':30,'frames':'','entry_frame_proxy':round(alert*30),'collision_frame':round(event*30),'entry_time_seconds':alert,'collision_time_seconds':event,
   'accident_type':'','scene':r['scene'],'label_origin':'official_time_of_alert_time_of_event','temporal_mapping':'alert_proxy_to_entry;event_to_collision','entry_side':'','evasion_space':''})
 fields=list(mm_norm[0])
 for name,rows in [('mmau_stage2_temporal.csv',mm_norm),('nexar_stage2_temporal.csv',nx_norm)]:
  with (OUT/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 report={'status':'COMPLETE_VALIDATED','inputs':{'mmau':{'path':str(MMAU.relative_to(ROOT)),'sha256':sha(MMAU),'rows':len(mm)},'nexar':{'path':str(NEXAR.relative_to(ROOT)),'sha256':sha(NEXAR),'rows':len(nx)}},
  'mmau':{'positive_rows':sum(str(r['whether an accident occurred (1/0)']).strip()=='1' for r in mm),'negative_rows':sum(str(r['whether an accident occurred (1/0)']).strip()=='0' for r in mm),'usable_positive_temporal_rows':len(mm_norm),'rejected_or_invalid':len(bad),'unique_hashcodes':len({r['video_id'] for r in mm_norm}),'type_counts':dict(sorted(Counter(r['accident_type'] for r in mm_norm).items())),'fps_in_metadata':False,'side_label_official':False,'evasion_label_official':False},
  'nexar':{'usable_positive_temporal_rows':len(nx_norm),'rejected_or_invalid':len(nx_bad),'unique_video_ids':len({r['video_id'] for r in nx_norm}),'fps_documented':30,'side_label_official':False,'evasion_label_official':False},
  'mapping_limits':['MM-AU t_ai is abnormal/accident-window start, not proven equivalent to competition entry_frame.','Nexar time_of_alert is action-needed time, not proven equivalent to competition entry_frame.','Neither metadata source provides official entry_side or evasion_space labels.'],
  'decision':'TEMPORAL_LABEL_SOURCE_ACCEPTED_FOR_ACQUISITION_AUDIT; DO_NOT_TREAT_PROXY ENTRY LABELS AS COMPETITION GROUND TRUTH BEFORE VIDEO CALIBRATION',
  'errors':{'mmau_first_20':bad[:20],'nexar_first_20':nx_bad[:20]}}
 (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

if __name__=='__main__':main()
