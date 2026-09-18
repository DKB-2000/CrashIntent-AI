const fs=require('fs'),path=require('path'),crypto=require('crypto');
const root=path.resolve(__dirname,'..');
const at=(...p)=>path.join(root,...p);
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const contactPath=at('artifacts','stage2-contact-review-20260914','contact-review.csv');
const predsDir=at('artifacts','stage2-unlabeled-comparison-20260914');
const out=at('artifacts','stage2-contact-window-audit-20260916');
const parse=p=>{const lines=fs.readFileSync(p,'utf8').replace(/^\uFEFF/,'').trim().split(/\r?\n/);const header=lines.shift().replaceAll('"','').split(',');return lines.map(line=>{const values=line.match(/(?:"(?:[^"]|"")*"|[^,]*)(?:,|$)/g).map(x=>x.replace(/,$/,'').replace(/^"|"$/g,'').replaceAll('""','"'));return Object.fromEntries(header.map((k,i)=>[k,values[i]]));});};
const contact=parse(contactPath);
const incumbent=new Map(parse(path.join(predsDir,'incumbent-predictions.csv')).map(r=>[r.ID,r]));
const pretrained=new Map(parse(path.join(predsDir,'ccd_pretrained-predictions.csv')).map(r=>[r.ID,r]));
if(contact.length!==50||incumbent.size!==50||pretrained.size!==50)throw Error('Expected 50 records each');
const rows=contact.map(c=>{
 const id=c.ID;const a=incumbent.get(id),b=pretrained.get(id);if(!a||!b)throw Error('Missing '+id);
 const mapPath=at('data','stage2-validation-nexar-20260914','frame-maps',id+'.csv');
 const lines=fs.readFileSync(mapPath,'utf8').trim().split(/\r?\n/);const h=lines.shift().split(',');const col=h.indexOf('timestamp_seconds');
 const times=lines.map(x=>+x.split(',')[col]);const start=+c.contact_window_start_seconds,end=+c.contact_window_end_seconds;
 const frame=n=>{n=+n;if(!Number.isInteger(n)||n<0||n>=times.length)throw Error('Out of range '+id);return times[n];};
 const ac=frame(a.collision_frame),bc=frame(b.collision_frame),ae=frame(a.entry_frame),be=frame(b.entry_frame);
 const inWindow=t=>t>=start&&t<=end;
 return {ID:id,ai_contact_screen:c.ai_contact_screen,window_start_seconds:start,window_end_seconds:end,
   incumbent_collision_seconds:ac,pretrained_collision_seconds:bc,incumbent_collision_in_screen_window:inWindow(ac),pretrained_collision_in_screen_window:inWindow(bc),
   incumbent_entry_seconds:ae,pretrained_entry_seconds:be,incumbent_entry_after_collision:ae>ac,pretrained_entry_after_collision:be>bc,
   collision_predictions_difference_seconds:Math.abs(ac-bc),
   scope:'MODEL_WINDOW_ALIGNMENT_ONLY_NOT_ACCURACY'};
});
fs.mkdirSync(out,{recursive:true});
const csvPath=path.join(out,'window-audit.csv');const keys=Object.keys(rows[0]);
fs.writeFileSync(csvPath,keys.join(',')+'\n'+rows.map(r=>keys.map(k=>JSON.stringify(r[k])).join(',')).join('\n')+'\n','utf8');
const byStatus={};for(const status of ['CONTACT_SUSPECT','NO_CONTACT_VISIBLE','UNRESOLVED']){
 const subset=rows.filter(r=>r.ai_contact_screen===status);byStatus[status]={videos:subset.length,incumbent_collision_in_window:subset.filter(r=>r.incumbent_collision_in_screen_window).length,pretrained_collision_in_window:subset.filter(r=>r.pretrained_collision_in_screen_window).length};}
const report={status:'WINDOW_ALIGNMENT_AUDIT_COMPLETE_NOT_ACCURACY',videos:50,by_ai_screen:byStatus,
 entry_after_collision:{incumbent:rows.filter(r=>r.incumbent_entry_after_collision).length,pretrained:rows.filter(r=>r.pretrained_entry_after_collision).length},
 caveat:'AI screen windows were selected near Nexar reference events and are not collision ground truth. A prediction outside the window may be right; inside may be wrong. No eligibility or accuracy claim.',
 hashes:{contact_csv:sha(contactPath),incumbent_csv:sha(path.join(predsDir,'incumbent-predictions.csv')),pretrained_csv:sha(path.join(predsDir,'ccd_pretrained-predictions.csv')),output_csv:sha(csvPath)}};
fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n','utf8');console.log(JSON.stringify(report,null,2));
