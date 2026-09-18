import copy
import csv
import json
from collections import Counter
from pathlib import Path
import unittest
from stage1_quality_trial_runner import audit_rows,schedule,training_rows


class ScreenTrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        plan=json.loads((root/'artifacts/kaggle-stage1-quality-trial-20260911/plan.json').read_text())
        with (root/'artifacts/stage1-screen-cue-data-20260914/generation_manifest.csv').open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):plan['rows'].append(dict(source_id=r['source_id'],upload_group=r['upload_group'],label=r['label'],path='screen/'+Path(r['output_path']).name,sha256=r['sha256'],role='screen_cue',profile=r['profile'],cue=r['cue']))
        cls.plan=plan;cls.train,_=audit_rows(plan['rows'],plan['excluded'])

    def test_only_quality_rerecorded_replaced(self):
        counts=Counter()
        for step,entries in enumerate(schedule(sorted({r['source_id'] for r in self.train}))):
            old=training_rows(self.train,'mixed_50',step,entries);new=training_rows(self.train,'screen_mix_50',step,entries)
            changed=0
            for (a,slot),(b,newslot) in zip(old,new):
                self.assertEqual((a['source_id'],a['profile'] if a['role']=='matched_quality' else None,a['label'],slot),(b['source_id'],b['profile'] if b['role'] in ('matched_quality','screen_cue') else None,b['label'],newslot))
                if a['role']=='matched_quality' and a['label']=='RERECORDED':
                    self.assertEqual(b['role'],'screen_cue');changed+=1;counts[b['cue']]+=1
                else:self.assertEqual(a,b)
            self.assertEqual(changed,2)
            self.assertEqual(Counter(r['label'] for r,s in new),{'ORIGINAL':4,'RERECORDED':4})
        self.assertEqual(sum(counts.values()),256);self.assertLessEqual(max(counts.values())-min(counts.values()),2)

    def test_cue_coverage_and_leakage_rejected(self):
        for key,value in [('label','ORIGINAL'),('source_id','bad_source'),('upload_group','bad_group'),('cue','invalid')]:
            broken=copy.deepcopy(self.plan['rows']);broken[-1][key]=value
            with self.assertRaises(ValueError):audit_rows(broken,self.plan['excluded'])
        with self.assertRaises(ValueError):audit_rows(self.plan['rows'][:-1],self.plan['excluded'])

    def test_quarter_mix_preserves_existing_recipe_and_pair_balance(self):
        counts=Counter()
        for step,entries in enumerate(schedule(sorted({r['source_id'] for r in self.train}))):
            control=training_rows(self.train,'mixed_25',step,entries)
            candidate=training_rows(self.train,'screen_mix_25',step,entries)
            self.assertEqual(Counter(r['label'] for r,_ in candidate),{'ORIGINAL':4,'RERECORDED':4})
            self.assertEqual(Counter(r['role'] for r,_ in control),{'existing_recipe':6,'matched_quality':2})
            self.assertEqual(Counter(r['role'] for r,_ in candidate),{'existing_recipe':6,'matched_quality':1,'screen_cue':1})
            changed=[]
            for (a,slot),(b,newslot) in zip(control,candidate):
                self.assertEqual(slot,newslot)
                self.assertEqual((a['source_id'],a['label']),(b['source_id'],b['label']))
                if a!=b:changed.append((a,b))
            self.assertEqual(len(changed),1)
            a,b=changed[0]
            self.assertEqual((a['role'],a['label'],a['profile']),("matched_quality","RERECORDED",b['profile']))
            self.assertEqual(b['role'],'screen_cue')
            counts[b['cue']]+=1
        self.assertEqual(sum(counts.values()),128)
        self.assertLessEqual(max(counts.values())-min(counts.values()),3)


if __name__=='__main__':unittest.main()
