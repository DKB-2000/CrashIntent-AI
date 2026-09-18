"""Numeric execution correction preserving baseline thread count."""
import json,torch
import stage3_temporal_motion_experiment as e
import compare_stage3_representations as c
if __name__=='__main__':
 proof=json.loads((e.ROOT/'baseline-thread-repro.json').read_text());assert proof['threads']==1 and not proof['exact']
 amendment=e.ROOT/'training-thread-correction.json'
 record=dict(reason='One-thread retrain differs from original two-thread weights by 7.38e-7; restore original two CPU threads for all new arms before any new arm metrics',original_plan_sha256=c.sha(e.ROOT/'plan.json'),numeric_proof_sha256=c.sha(e.ROOT/'baseline-thread-repro.json'),training_threads=2,extraction_cv2_threads=1,script_sha256=c.sha(__file__))
 if amendment.exists():assert json.loads(amendment.read_text())==record
 else:c.save(amendment,record)
 torch.set_num_threads(2);c.cv2.setNumThreads(1);e.train()
