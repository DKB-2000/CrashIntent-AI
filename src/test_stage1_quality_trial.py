import copy,unittest
from stage1_quality_trial_runner import schedule,rates,STEPS,audit_rows
from validate_stage1_quality_trial import verify_predictions,macro_f1,verify_summary

class TrialTests(unittest.TestCase):
    def test_summary_roundoff_only(self):
        expected={'original_noise':dict(videos=21,errors=19,positive_rate=19/21,mean_probability=0.8472893540821377)}
        saved=copy.deepcopy(expected)
        saved['original_noise']['mean_probability']=0.8472893540821379
        verify_summary(expected,saved)
        for key,value in [('mean_probability',0.8473),('mean_probability',float('nan')),('errors',18),('videos',20),('positive_rate',0.9)]:
            broken=copy.deepcopy(saved);broken['original_noise'][key]=value
            with self.assertRaises(ValueError):verify_summary(expected,broken)
        with self.assertRaises(ValueError):verify_summary(expected,{})

    def test_identical_fixed_training_schedule(self):
        first=schedule(['a','b','c'])
        self.assertEqual(first,schedule(['a','b','c']))
        self.assertEqual(len(first),STEPS)
        self.assertTrue(all(len(batch)==4 for batch in first))
        self.assertEqual({r['profile'] for batch in first for r in batch},{0,1,2})
        self.assertEqual({r['slot'] for batch in first for r in batch},{0,1,2})

    def test_arm_coverage_and_group_leakage(self):
        rows=[]
        for i in range(85):
            common=dict(source_id=f't{i}',upload_group=f'g{i}')
            for variant in range(3):rows.append(dict(common,path=f'old/{i}/{variant}',role='existing_recipe',variant=variant,label='ORIGINAL' if variant==0 else 'RERECORDED'))
            for profile in ('clean','noise','quality_mix'):
                for label in ('ORIGINAL','RERECORDED'):rows.append(dict(common,path=f'quality/{i}/{profile}/{label}',role='matched_quality',profile=profile,label=label))
        for i in range(420):rows.append(dict(source_id=f'v{i%21}',upload_group=f'vg{i%21}',path=f'val/{i}',role='evaluation',suite='stress' if i<357 else 'normal',label='ORIGINAL'))
        train,val=audit_rows(rows,{'phone'})
        self.assertEqual((len(train),len(val)),(765,420))
        with self.assertRaises(ValueError):audit_rows(rows[1:],{'phone'})
        with self.assertRaises(ValueError):audit_rows(rows,{'t0'})
        broken=copy.deepcopy(rows);broken[-1]['upload_group']='g0'
        with self.assertRaisesRegex(ValueError,'leakage'):audit_rows(broken,set())

    def test_saved_output_recomputation_rejects_wrong_probability_and_identity(self):
        row=dict(path='x',source_id='s',upload_group='g',condition='original_noise',suite='stress',sha256='h',label='ORIGINAL')
        p={k:v for k,v in row.items() if k!='label'}
        p.update(target='ORIGINAL',probability=.5,answer='RERECORDED',slot_logits=[[0.,0.]]*3,slot_probabilities=[.5]*3)
        verify_predictions([row],[p]);self.assertEqual(rates([p])['original_noise']['errors'],1)
        for key,value in [('probability',.9),('source_id','other'),('answer','ORIGINAL')]:
            broken=copy.deepcopy(p);broken[key]=value
            with self.assertRaises(ValueError):verify_predictions([row],[broken])
        with self.assertRaises(ValueError):verify_predictions([row],[])
        self.assertEqual(macro_f1([p]),0)

if __name__=='__main__':unittest.main()
