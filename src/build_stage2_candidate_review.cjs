const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const root = path.resolve(__dirname, '..');
const at = (...p) => path.join(root, ...p);
const hash = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const input = at('artifacts', 'stage2-unlabeled-comparison-20260914');
const output = at('artifacts', 'stage2-ai-candidate-review-20260916');
const images = at('data', 'stage2-validation-nexar-20260914', 'images');
const frames = at('data', 'stage2-validation-nexar-20260914', 'frame-maps');
const contact = new Map(JSON.parse(fs.readFileSync(at('artifacts','stage2-contact-review-20260914','observations.json'),'utf8')).map(r => [r.ID,r]));
const direction = new Map(JSON.parse(fs.readFileSync(at('artifacts','stage2-ai-direction-test-20260914','ai-observations.json'),'utf8')).rows.map(r => [r.ID,r]));
const readPred = name => {
  const lines = fs.readFileSync(path.join(input,name),'utf8').trim().split(/\r?\n/).slice(1);
  return new Map(lines.map(line => { const [ID,collision_frame,entry_frame,evasion_space,entry_side] = line.split(','); return [ID,{collision_frame:+collision_frame,entry_frame:+entry_frame,evasion_space,entry_side}]; }));
};
const incumbent = readPred('incumbent-predictions.csv');
const pretrained = readPred('ccd_pretrained-predictions.csv');
const ids = [...incumbent.keys()].sort();
if (ids.length !== 50 || ids.some(id => !pretrained.has(id) || !contact.has(id))) throw Error('Missing candidate records');
const esc = x => String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const items = ids.map(id => {
  const lines = fs.readFileSync(path.join(frames,`${id}.csv`),'utf8').trim().split(/\r?\n/);
  const header = lines.shift().split(',');
  const timeCol = header.indexOf('timestamp_seconds');
  const time = lines.map(line => Number(line.split(',')[timeCol]));
  const a = incumbent.get(id), b = pretrained.get(id);
  for (const n of [a.collision_frame,a.entry_frame,b.collision_frame,b.entry_frame]) if (!Number.isInteger(n) || n < 0 || n >= time.length) throw Error(`Out-of-range frame ${id}: ${n}`);
  const d = direction.get(id), c = contact.get(id);
  return {id,frames:time.length,contact:c.ai_contact_screen,contact_note:c.ai_contact_note,
    ai_direction:d?.entry_side || null,ai_direction_note:d?.note || null,
    incumbent:a,pretrained:b,
    time:{incumbent_collision:time[a.collision_frame],incumbent_entry:time[a.entry_frame],pretrained_collision:time[b.collision_frame],pretrained_entry:time[b.entry_frame]},
    disagreement:{collision_seconds:Math.abs(time[a.collision_frame]-time[b.collision_frame]),entry_seconds:Math.abs(time[a.entry_frame]-time[b.entry_frame]),evasion:a.evasion_space!==b.evasion_space,side:a.entry_side!==b.entry_side},
  };
}).sort((a,b) => (b.ai_direction!==null)-(a.ai_direction!==null) || (b.contact==='CONTACT_SUSPECT')-(a.contact==='CONTACT_SUSPECT') || b.disagreement.collision_seconds-a.disagreement.collision_seconds);
fs.mkdirSync(output,{recursive:true});
const img = (id,n,label) => `<figure><a target="_blank" href="../../data/stage2-validation-nexar-20260914/images/${id}/frame_${String(n).padStart(6,'0')}.jpg"><img loading="lazy" src="../../data/stage2-validation-nexar-20260914/images/${id}/frame_${String(n).padStart(6,'0')}.jpg"></a><figcaption>${esc(label)} · ${n}</figcaption></figure>`;
const card = r => `<section id="${r.id}"><h2>${r.id}</h2><p>AI 접촉 선별: ${esc(r.contact)} · ${esc(r.contact_note)}</p><p>AI 방향 참고: ${esc(r.ai_direction||'보류')} · ${esc(r.ai_direction_note||'판독값 없음')}</p><p>모델 충돌 후보 차이 ${r.disagreement.collision_seconds.toFixed(2)}초, 진입 후보 차이 ${r.disagreement.entry_seconds.toFixed(2)}초. 이 수치는 모델 간 차이입니다.</p><div class="grid">${img(r.id,r.incumbent.collision_frame,`기존 충돌 후보 ${r.time.incumbent_collision.toFixed(2)}초`)}${img(r.id,r.pretrained.collision_frame,`사전학습 충돌 후보 ${r.time.pretrained_collision.toFixed(2)}초`)}${img(r.id,r.incumbent.entry_frame,`기존 진입 후보 ${r.time.incumbent_entry.toFixed(2)}초`)}${img(r.id,r.pretrained.entry_frame,`사전학습 진입 후보 ${r.time.pretrained_entry.toFixed(2)}초`)}</div><p>회피 후보: 기존 ${r.incumbent.evasion_space} / 사전학습 ${r.pretrained.evasion_space}; 방향 후보: 기존 ${r.incumbent.entry_side} / 사전학습 ${r.pretrained.entry_side}.</p><p><a href="../../data/stage2-validation-nexar-20260914/review.html">전체 영상 검수 화면</a>에서 사건 전후 연속 프레임을 확인해야 합니다.</p></section>`;
const html = `<!doctype html><html lang="ko"><meta charset="utf-8"><title>Stage2 AI 후보 근거 검토</title><style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:24px;background:#f5f6f8;color:#17202b}section{background:white;padding:20px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}figure{margin:0}img{width:100%;height:230px;object-fit:contain;background:#111}figcaption{font-size:14px}a{color:#075c9d}</style><h1>Stage2 AI 후보 근거 검토</h1><p>두 모델의 후보 프레임과 기존 AI 선별을 함께 보여줍니다. 모델 출력과 AI 선별은 정답이 아닙니다. 네 항목의 독립 정답은 아직 0건입니다. 예측을 보며 검수하면 블라인드 채점에 사용할 수 없으므로 이 화면은 진단용으로만 사용하세요.</p><nav>${items.map(r=>`<a href="#${r.id}">${r.id}</a>`).join(' · ')}</nav>${items.map(card).join('')}</html>`;
fs.writeFileSync(path.join(output,'review.html'),html,'utf8');
fs.writeFileSync(path.join(output,'candidates.json'),JSON.stringify(items,null,2)+'\n','utf8');
const report={status:'MODEL_CANDIDATE_EVIDENCE_READY_UNVERIFIED',videos:items.length,candidate_frames:items.length*4,ai_direction_references:items.filter(r=>r.ai_direction).length,full_four_field_labels:0,source_hashes:{incumbent_predictions:hash(path.join(input,'incumbent-predictions.csv')),pretrained_predictions:hash(path.join(input,'ccd_pretrained-predictions.csv'))},output_hashes:{review_html:hash(path.join(output,'review.html')),candidates_json:hash(path.join(output,'candidates.json'))},warning:'Model outputs are candidate locations, not labels; this prediction-exposed page cannot be used for blind final scoring.'};
fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n','utf8');
console.log(JSON.stringify(report,null,2));
