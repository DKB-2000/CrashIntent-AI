const fs=require('fs'),path=require('path'),crypto=require('crypto');
const root=path.resolve(__dirname,'..');
const at=(...p)=>path.join(root,...p);
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const dir=at('artifacts','stage2-source-cv-20260910');
const out=at('artifacts','stage2-ordered-oof-audit-20260916');
const configs=['exact_5','exact_15','soft_mixed_15'];
function read(file){const lines=fs.readFileSync(file,'utf8').trim().split(/\r?\n/);const keys=lines.shift().split(',');return lines.map(line=>{const v=line.split(',');if(v.length!==keys.length)throw Error('CSV field count '+file);return Object.fromEntries(keys.map((k,i)=>[k,v[i]]));});}
const files=configs.map(c=>path.join(dir,`${c}_oof.csv`));
const inputs=Object.fromEntries(configs.map((c,i)=>[c,read(files[i])]));
for(const c of configs)if(inputs[c].length!==66||new Set(inputs[c].map(r=>r.ID)).size!==66)throw Error('Expected 66 distinct OOF rows');
const truth=inputs[configs[0]];
if(configs.slice(1).some(c=>inputs[c].some((r,i)=>r.ID!==truth[i].ID||r.t_collision!==truth[i].t_collision||r.t_entry!==truth[i].t_entry||r.source_id!==truth[i].source_id)))throw Error('OOF rows/truth differ');
const truthViolations=truth.filter(r=>+r.t_entry>+r.t_collision);
const rows=[];const report={status:'ORDERED_OOF_DIAGNOSTIC_ONLY',truth_entry_after_collision:truthViolations.length,truth_examples:truthViolations.map(r=>r.ID),conditions:{},limitations:'Saved predicted frames only. Clamp is an exploratory diagnostic, not joint logit decoding; scene labels cannot be recomputed. Reused development folds and proxy/manual labels are not independent confirmation.'};
for(const c of configs){let original=0,clamped=0,violation=0,gained=0,lost=0,unchangedWrong=0;const bySource=new Map();
 for(const r of inputs[c]){const pred=+r.pred_entry_frame,collision=+r.pred_collision_frame,gt=+r.t_entry,fps=+r.fps;const fixed=Math.min(pred,collision);const before=Math.abs(pred-gt)/fps<=0.3+1e-9,after=Math.abs(fixed-gt)/fps<=0.3+1e-9;
  if(pred>collision)violation++;if(before)original++;if(after)clamped++;if(!before&&after)gained++;if(before&&!after)lost++;if(!before&&!after)unchangedWrong++;
  const s=bySource.get(r.source_id)||{videos:0,violations:0,entry_ok_before:0,entry_ok_clamped:0};s.videos++;s.violations+=pred>collision;s.entry_ok_before+=before;s.entry_ok_clamped+=after;bySource.set(r.source_id,s);
  rows.push({condition:c,ID:r.ID,source_id:r.source_id,t_collision:+r.t_collision,t_entry:gt,pred_collision_frame:collision,pred_entry_frame:pred,clamped_entry_frame:fixed,violation:pred>collision,entry_ok_before:before,entry_ok_clamped:after});
 }
 report.conditions[c]={videos:66,pred_entry_after_collision:violation,entry_ok_before:original,entry_ok_clamped:clamped,net_change:clamped-original,gained,lost,unchanged_wrong:unchangedWrong,source_groups:bySource.size,source_violations:[...bySource.entries()].filter(([_,s])=>s.violations).map(([source_id,s])=>({source_id,...s}))};
}
fs.mkdirSync(out,{recursive:true});const keys=Object.keys(rows[0]);const csv=path.join(out,'paired.csv');fs.writeFileSync(csv,keys.join(',')+'\n'+rows.map(r=>keys.map(k=>JSON.stringify(r[k])).join(',')).join('\n')+'\n','utf8');
report.input_sha256=Object.fromEntries(configs.map((c,i)=>[c,sha(files[i])]));report.output_sha256=sha(csv);fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n','utf8');console.log(JSON.stringify(report,null,2));
