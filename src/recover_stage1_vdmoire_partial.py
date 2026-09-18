"""Recover complete VDMoire clips from the bounded prefix of nested tcl.zip."""
import collections,hashlib,json,struct,zlib
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]; SOURCE=ROOT/'artifacts/stage1-rawvdemoire-scout-20260918/dropbox-original.html'; OUT=ROOT/'artifacts/stage1-vdmoire-pilot-20260918'
def entries():
 size=SOURCE.stat().st_size; result=[]
 with SOURCE.open('rb') as stream:
  stream.seek(106)
  while stream.tell()+30<=size:
   offset=stream.tell()
   if stream.read(4)!=b'PK\x03\x04': break
   header=stream.read(26); ver,flag,method,mt,md,crc,compressed,uncompressed,name_len,extra_len=struct.unpack('<HHHHHIIIHH',header); name=stream.read(name_len).decode('utf-8'); stream.seek(extra_len,1); data_offset=stream.tell()
   if flag&8 or compressed==0xffffffff or data_offset+compressed>size: break
   result.append({'name':name,'method':method,'crc':crc,'compressed':compressed,'uncompressed':uncompressed,'offset':data_offset}); stream.seek(compressed,1)
 return result
def read(entry):
 with SOURCE.open('rb') as stream: stream.seek(entry['offset']); data=stream.read(entry['compressed'])
 value=zlib.decompress(data,-15) if entry['method']==8 else data
 if len(value)!=entry['uncompressed'] or (zlib.crc32(value)&0xffffffff)!=entry['crc']: raise ValueError('CRC mismatch: '+entry['name'])
 return value
def main():
 OUT.mkdir(exist_ok=False); found=entries(); groups=collections.defaultdict(list)
 for entry in found:
  if entry['name'].lower().endswith('.jpg'): groups[str(Path(entry['name']).parent).replace('\\','/')].append(entry)
 complete={name:values for name,values in groups.items() if len(values)>=180}; selected=sorted(complete)[:8]; clips=[]
 for index,name in enumerate(selected):
  values=sorted(complete[name],key=lambda x:x['name']); chosen=[values[round(i*(len(values)-1)/47)] for i in range(48)]; frames=[]
  for entry in chosen:
   frame=cv2.imdecode(np.frombuffer(read(entry),np.uint8),cv2.IMREAD_COLOR)
   if frame is None: raise ValueError('JPEG decode failed')
   frames.append(frame)
  height,width=frames[0].shape[:2]; path=OUT/f'vdmoire_tcl_{index:02d}.mp4'; writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),16,(width,height))
  for frame in frames: writer.write(frame)
  writer.release(); capture=cv2.VideoCapture(str(path)); count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); ok,first=capture.read(); capture.release()
  if count!=48 or not ok: raise ValueError('Video validation failed')
  clips.append({'path':str(path.relative_to(ROOT)).replace('\\','/'),'source_folder':name,'frames':count,'width':width,'height':height,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'label':'RERECORDED'})
 report={'status':'ACQUIRED_VALIDATED','source':'CVMI-Lab/VideoDemoireing official Dropbox original TCL archive prefix','source_prefix_bytes':SOURCE.stat().st_size,'indexed_complete_members':len(found),'video_folders_seen':len(groups),'complete_180_frame_folders':len(complete),'recovered_clips':len(clips),'clips':clips,'license_note':'Repository LICENSE applies to code; dataset-specific license not stated. Diagnostic only until clarified.','limitations':['Only RERECORDED clips recovered; no paired ground truth in this archive prefix.','Prefix download was stopped; full 33.9 GiB archive was not acquired.']}; (OUT/'status.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report))
if __name__=='__main__': main()
