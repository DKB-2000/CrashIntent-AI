"""Small, nonblind AI-reference test; never writes human labels or changes models."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/stage2-ai-direction-test-20260914'
PRED = ROOT/'artifacts/stage2-unlabeled-comparison-20260914'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    source = OUT/'ai-observations.json'
    observation = json.loads(source.read_text(encoding='utf-8'))
    prior_path = ROOT/'artifacts/stage2-contact-review-20260914/observations.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    rows = observation['rows']
    assert len(rows) == len({r['ID'] for r in rows}) == 13
    assert {r['ID'] for r in rows} == {r['ID'] for r in prior if r['ai_contact_screen']=='CONTACT_SUSPECT'}
    assert all(r['entry_side'] in (None,'LEFT','RIGHT') for r in rows)
    digests = {'ai-observations.json':sha(source), str(prior_path.relative_to(ROOT)):sha(prior_path)}
    for r in rows:
        p = prior_path.parent/(r['ID']+'.png')
        digests[str(p.relative_to(ROOT))] = sha(p)
    validation = json.loads((PRED/'validation.json').read_text())
    predictions = {}
    for name in ('incumbent','ccd_pretrained'):
        path = PRED/(name+'-predictions.csv')
        assert sha(path) == validation['output_hashes'][path.name]
        with path.open(encoding='utf-8', newline='') as f:
            raw = list(csv.DictReader(f))
        predictions[name] = {r['ID']:r['entry_side'] for r in raw}
        assert len(predictions[name]) == len(raw) == 50
        digests[str(path.relative_to(ROOT))] = sha(path)
    selected = [r for r in rows if r['entry_side'] is not None]
    paired = [dict(ID=r['ID'], ai_reference=r['entry_side'],
        **{n:p[r['ID']] for n,p in predictions.items()}) for r in selected]
    scores = {n:dict(matches=sum(r[n]==r['ai_reference'] for r in paired), evaluated=len(paired)) for n in predictions}
    for s in scores.values():
        s['ai_reference_agreement'] = s['matches']/s['evaluated']
    # Recount the same comparison with a transition table, including both wrong/correct cases.
    table = {key:0 for key in ('both_match','incumbent_only','ccd_pretrained_only','neither_match')}
    for r in paired:
        a,b = (r[n]==r['ai_reference'] for n in predictions)
        table['both_match' if a and b else 'incumbent_only' if a else 'ccd_pretrained_only' if b else 'neither_match'] += 1
    assert scores['incumbent']['matches']==table['both_match']+table['incumbent_only']
    assert scores['ccd_pretrained']['matches']==table['both_match']+table['ccd_pretrained_only']
    label_path = ROOT/'data/stage2-validation-nexar-20260914/review-decisions.csv'
    protocol = json.loads((PRED/'protocol.json').read_text())
    assert sha(label_path)==protocol['protected_hashes'][str(label_path.relative_to(ROOT))]
    report = dict(status='COMPLETE_VALIDATED', scope=observation['scope'], reviewed=13, evaluated=len(selected),
        abstained=13-len(selected), scores=scores, paired_counts=table, paired=paired,
        limitations=['Only entry direction', 'AI reference, not human ground truth', 'Nonblind reviewer',
            'Contact-suspect selected subset', 'Short video windows', 'Not enough to establish overall superiority'],
        historical_official_scores=dict(incumbent=0.2096390642, ccd_pretrained=0.1966181317,
            source='docs/submission-history.md; user-reported server results, server ZIP hash unverified'),
        validation='PASS: selection, evidence hashes, prediction hashes, independent counts, human worksheet unchanged',
        input_hashes=digests)
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='input_hashes'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
