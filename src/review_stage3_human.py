"""Local blind Stage3 steering review. No model predictions, training, or upload."""
import argparse,csv,hashlib,json,os
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[1]
STEER={'1':'LEFT','2':'STRAIGHT','3':'RIGHT','0':'UNKNOWN'}
MOTION={'m':'MOVING','t':'STOPPED','u':'UNKNOWN'}

def add_segment(rows,start,end,steer,motion,count):
    if not 0<=start<=end<count:raise ValueError('Invalid interval: set I then O (inclusive).')
    if steer not in STEER.values() or motion not in MOTION.values():raise ValueError('Invalid label')
    if any(start<=r['end'] and end>=r['start'] for r in rows):raise ValueError('Overlap: undo previous interval with Z first.')
    return rows+[dict(start=start,end=end,steer=steer,motion=motion)]

def dense_rows(sid,count,segments):
    labels={}
    for row in segments:
        for i in range(row['start'],row['end']+1):
            if i in labels:raise ValueError('Overlapping saved intervals')
            labels[i]=row
    for i in range(count):
        r=labels.get(i,{});steer=r.get('steer','UNKNOWN');motion=r.get('motion','UNKNOWN')
        yield dict(ID=sid,sample_index=i,steer_label=steer,motion_status=motion,
                   reviewed=int(i in labels),valid_steer=int(steer!='UNKNOWN' and motion=='MOVING'))

def save(state,path):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(temp,path)
    csvpath=path.with_name('labels_review.csv');temp=csvpath.with_suffix('.tmp')
    with temp.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['ID','sample_index','steer_label','motion_status','reviewed','valid_steer']);writer.writeheader()
        for sid,v in state['videos'].items():writer.writerows(dense_rows(sid,v['frames'],v['segments']))
    os.replace(temp,csvpath)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video-dir',type=Path,default=ROOT/'Baseline/data/stage3/videos')
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/stage3-human-review')
    p.add_argument('--check',action='store_true',help='Validate inputs and saved annotations without opening a window')
    p.add_argument('--assume-10hz',action='store_true',help='Only for independently verified 10Hz video with incorrect FPS metadata; does not resample')
    args=p.parse_args();videos=sorted(args.video_dir.glob('*.mp4'))
    if not videos:raise ValueError('No MP4 videos found')
    if len({f.stem for f in videos})!=len(videos):raise ValueError('Duplicate video IDs')
    args.output_dir.mkdir(parents=True,exist_ok=True);path=args.output_dir/'annotations.json'
    state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(schema=1,scope='Human development review; not competition labels',videos={})
    assert state['schema']==1
    if set(state['videos'])-{v.stem for v in videos}:raise ValueError('Saved videos missing: use the original video directory or a new output directory')
    for f in videos:
        cap=cv2.VideoCapture(str(f));fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ok,_=cap.read();cap.release()
        if not ok or (abs(fps-10)>.01 and not args.assume_10hz) or count<1:raise ValueError(f'{f.name}: requires decodable 10 FPS video; got {fps} FPS')
        digest=hashlib.sha256(f.read_bytes()).hexdigest()
        if f.stem in state['videos']:
            v=state['videos'][f.stem]
            if v['sha256']!=digest or v['frames']!=count:raise ValueError('Source changed: '+f.name)
            rows=[]
            for r in v['segments']:rows=add_segment(rows,r['start'],r['end'],r['steer'],r['motion'],count)
        else:state['videos'][f.stem]=dict(path=str(f.resolve()),sha256=digest,frames=count,fps=fps,assumed_10hz=args.assume_10hz,segments=[])
    save(state,path)
    if args.check:
        print(json.dumps(dict(status='PASS',videos=len(videos),frames=sum(v['frames'] for v in state['videos'].values()),output=str(path)),indent=2));return
    vi=0;index=0;start=None;end=None;steer='UNKNOWN';motion='UNKNOWN';playing=False;message=''
    cap=None;active=None;last=-2
    window='Stage3 human review - focus this window for keys'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    try:
        while True:
            f=videos[vi];v=state['videos'][f.stem]
            if active!=f:
                if cap:cap.release()
                cap=cv2.VideoCapture(str(f));active=f;last=-2
            if index!=last+1:cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=cap.read()
            if not ok:raise ValueError(f'Decode failed: {f.name} frame {index}')
            last=index;h,w=frame.shape[:2];frame=cv2.resize(frame,(960,round(h*960/w)))
            existing=next((r for r in v['segments'] if r['start']<=index<=r['end']),None)
            lines=[f'{vi+1}/{len(videos)} {f.stem}  frame {index}/{v["frames"]-1} ({index/10:.1f}s)',
                   f'Range I={start} O={end}  choice: {steer} / {motion}',
                   'SPACE play | A/D frame | J/K +/-1s | I/O start/end | 1 LEFT 2 STRAIGHT 3 RIGHT 0 UNKNOWN',
                   'M moving T stopped U unsure | S save range | Z undo last range | N/B next/previous video | Q quit',
                   'Saved at cursor: '+str(existing),message]
            for row,line in enumerate(lines):
                cv2.putText(frame,line,(10,22+row*22),cv2.FONT_HERSHEY_SIMPLEX,.46,(0,0,0),3)
                cv2.putText(frame,line,(10,22+row*22),cv2.FONT_HERSHEY_SIMPLEX,.46,(0,255,255),1)
            cv2.imshow(window,frame);key=cv2.waitKeyEx(100 if playing else 30)
            if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            if key in [27,ord('q')]:break
            if key==-1:
                if playing:index=min(index+1,v['frames']-1);playing=index<v['frames']-1
                continue
            ch=chr(key).lower() if 0<=key<256 else ''
            try:
                if ch==' ':playing=not playing
                elif ch in 'adjk' and ch:
                    playing=False;index=max(0,min(v['frames']-1,index+dict(a=-1,d=1,j=-10,k=10)[ch]))
                elif ch=='i':start=index;playing=False
                elif ch=='o':end=index;playing=False
                elif ch in STEER:steer=STEER[ch]
                elif ch in MOTION:motion=MOTION[ch]
                elif ch=='s':
                    if start is None or end is None:raise ValueError('Set I and O first')
                    v['segments']=add_segment(v['segments'],start,end,steer,motion,v['frames']);save(state,path);start=end=None;message='Saved automatically.'
                elif ch=='z':
                    if v['segments']:v['segments'].pop();save(state,path);message='Undid last saved range in this video.'
                elif ch in ['n','b']:
                    vi=(vi+(1 if ch=='n' else -1))%len(videos);index=0;start=end=None;steer=motion='UNKNOWN';playing=False;message=''
            except ValueError as e:message=str(e)
    finally:
        if cap:cap.release()
        cv2.destroyAllWindows();save(state,path)
    print('Saved:',path)

if __name__=='__main__':main()
