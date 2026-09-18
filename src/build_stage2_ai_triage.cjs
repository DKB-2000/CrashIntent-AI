const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const root = path.resolve(__dirname, '..');
const at = (...p) => path.join(root, ...p);
const hash = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const contactPath = at('artifacts', 'stage2-contact-review-20260914', 'observations.json');
const directionPath = at('artifacts', 'stage2-ai-direction-test-20260914', 'ai-observations.json');
const worksheetPath = at('data', 'stage2-validation-nexar-20260914', 'review-decisions.csv');
const out = at('artifacts', 'stage2-ai-triage-20260916');
const originalHash = hash(worksheetPath);
const contact = new Map(JSON.parse(fs.readFileSync(contactPath, 'utf8')).map(x => [x.ID, x]));
const direction = new Map(JSON.parse(fs.readFileSync(directionPath, 'utf8')).rows.map(x => [x.ID, x]));
const lines = fs.readFileSync(worksheetPath, 'utf8').replace(/^\uFEFF/, '').trim().split(/\r?\n/);
const header = lines.shift().split(',');
if (lines.length !== 50 || new Set(lines.map(x => x.split(',')[0])).size !== 50) throw Error('Expected 50 distinct worksheet candidates');
const rows = lines.map(line => {
  const values = line.split(',');
  const id = values[0];
  if (values[1] !== 'PENDING' || ['collision_frame', 'entry_frame', 'evasion_space', 'entry_side'].some(k => values[header.indexOf(k)])) throw Error('Worksheet contains reviewed labels');
  const c = contact.get(id), d = direction.get(id);
  if (!c) throw Error(`Missing contact observation: ${id}`);
  const side = d?.entry_side || '';
  const evidence = at('artifacts', 'stage2-contact-review-20260914', `${id}.png`);
  return {
    ID: id,
    priority: side ? '1_DIRECTION_REFERENCE' : c.ai_contact_screen === 'CONTACT_SUSPECT' ? '2_CONTACT_SUSPECT' : c.ai_contact_screen === 'UNRESOLVED' ? '3_UNRESOLVED' : '4_NO_CONTACT_VISIBLE',
    ai_contact_screen: c.ai_contact_screen,
    ai_contact_note: c.ai_contact_note,
    ai_entry_side_reference: side,
    ai_entry_side_note: d?.note || '',
    ai_collision_frame: '', ai_entry_frame: '', ai_evasion_space: '',
    evidence_sheet: fs.existsSync(evidence) ? path.relative(root, evidence).replaceAll('\\', '/') : '',
    evidence_sha256: fs.existsSync(evidence) ? hash(evidence) : '',
    review_disposition: 'ABSTAIN_UNVERIFIED',
  };
}).sort((a,b) => a.priority.localeCompare(b.priority) || a.ID.localeCompare(b.ID));
const escape = x => `"${String(x).replaceAll('"', '""')}"`;
const keys = Object.keys(rows[0]);
fs.mkdirSync(out, {recursive: true});
const csvPath = path.join(out, 'triage.csv');
fs.writeFileSync(csvPath, keys.join(',') + '\n' + rows.map(r => keys.map(k => escape(r[k])).join(',')).join('\n') + '\n', 'utf8');
if (hash(worksheetPath) !== originalHash) throw Error('Worksheet changed during triage');
const report = {
  status: 'AI_TRIAGE_READY_UNVERIFIED', candidates: rows.length,
  direction_references: rows.filter(r => r.ai_entry_side_reference).length,
  full_four_field_labels: 0,
  contact_suspects: rows.filter(r => r.ai_contact_screen === 'CONTACT_SUSPECT').length,
  worksheet_sha256_before_after: originalHash, triage_sha256: hash(csvPath),
  sources: {contact_observations_sha256: hash(contactPath), direction_observations_sha256: hash(directionPath)},
  limitations: 'Contact screening and six nonblind direction references only; no collision/entry frame or evasion ground truth. Triage is not accepted review labels and cannot enter final scoring.'
};
fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 2) + '\n', 'utf8');
console.log(JSON.stringify(report, null, 2));
