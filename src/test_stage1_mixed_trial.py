import json
import unittest
from collections import Counter
from pathlib import Path
from stage1_quality_trial_runner import schedule,training_rows,audit_rows


class MixedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plan=json.loads((Path(__file__).resolve().parents[1]/'artifacts/kaggle-stage1-quality-trial-20260911/plan.json').read_text())
        cls.train,_=audit_rows(plan['rows'],plan['excluded'])
        cls.batches=schedule(sorted({r['source_id'] for r in cls.train}))

    def test_exact_previous_arm_sample_sequence(self):
        for arm in ('existing_recipe','matched_quality'):
            for step,entries in enumerate(self.batches):
                expected=[]
                for entry in entries:
                    for label in ('ORIGINAL','RERECORDED'):
                        candidates=[r for r in self.train if r['role']==arm and r['source_id']==entry['source_id'] and r['label']==label]
                        if arm=='matched_quality':candidates=[r for r in candidates if r['profile']==('clean','noise','quality_mix')[entry['profile']]]
                        else:candidates=[r for r in candidates if int(r['variant'])==(0 if label=='ORIGINAL' else 1+entry['profile']%2)]
                        self.assertEqual(len(candidates),1);expected.append((candidates[0],entry['slot']))
                self.assertEqual(training_rows(self.train,arm,step,entries),expected)

    def test_balanced_pairs_and_rotating_recipes(self):
        for step,entries in enumerate(self.batches):
            selected=training_rows(self.train,'mixed_50',step,entries)
            self.assertEqual(Counter((r['role'],r['label']) for r,s in selected),Counter({(role,label):2 for role in ('matched_quality','existing_recipe') for label in ('ORIGINAL','RERECORDED')}))
            for pair,entry in enumerate(entries):
                a,b=selected[pair*2:pair*2+2]
                self.assertEqual((a[0]['source_id'],b[0]['source_id']),(entry['source_id'],)*2)
                self.assertEqual((a[1],b[1]),(entry['slot'],)*2)
                self.assertEqual(a[0]['role'],b[0]['role'])
                next_role=training_rows(self.train,'mixed_50',step+1,entries)[pair*2][0]['role']
                self.assertNotEqual(a[0]['role'],next_role)


if __name__=='__main__':unittest.main()
