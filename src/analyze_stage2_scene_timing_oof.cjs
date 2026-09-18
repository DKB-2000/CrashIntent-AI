const fs=require('fs'),path=require('path'),crypto=require('crypto');
const root=path.resolve(__dirname,'..');const at=(...p)=>path.join(root,...p);
const dir=at('artifacts','stage2-source-cv-20260910'),out=at('artifacts','stage2-scene-timing-oof-20260916');
const configs=['exact_5','exact_15','soft_mixed_15'];
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const result={status:'SCENE_TIMING_OOF_DIAGNOSTIC_ONLY',conditions:{},limitations:'Conditioning on prediction correctness is observational; it cannot prove scene errors are caused by timing. Same 66 previously reused development videos; no new model or ground truth.'};
for(const name of configs){const file=path.join(dir,name+'_oof.csv');const lines=fs.readFileSync(file,'utf8').trim().split(/\r?\n/);const head=lines.shift().split(',');
 const rows=lines.map(line=>{const v=line.split(',');if(v.length!==head.length)throw Error('Bad CSV');return Object.fromEntries(head.map((k,i)=>[k,v[i]]));});
 if(rows.length!==66)throw Error('Expected 66');
 const groups={both_times_correct:[],one_time_correct:[],neither_time_correct:[]};
 for(const r of rows){const c=r.collision_frame_ok==='True',e=r.entry_frame_ok==='True';groups[c&&e?'both_times_correct':c||e?'one_time_correct':'neither_time_correct'].push(r);}
 result.conditions[name]={input_sha256:sha(file),groups:Object.fromEntries(Object.entries(groups).map(([g,rs])=>[g,{videos:rs.length,evasion_correct:rs.filter(r=>r.evasion_space_ok==='True').length,direction_correct:rs.filter(r=>r.entry_side_ok==='True').length,both_scene_correct:rs.filter(r=>r.evasion_space_ok==='True'&&r.entry_side_ok==='True').length}]))};
}
fs.mkdirSync(out,{recursive:true});fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(result,null,2)+'\n','utf8');console.log(JSON.stringify(result,null,2));
