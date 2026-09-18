"""Single private notebook push; bounded recovery/validation using existing assets."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import watch_stage1_quality_trial as watch

watch.WORK=watch.ROOT/'artifacts/kaggle-stage1-mixed-trial-20260914'
watch.KERNEL='biadis/crashintent-stage1-mixed-trial'


def main(launch):
    work=watch.WORK
    bundle=json.loads((work/'bundle.json').read_text(encoding='utf-8'))
    if bundle['kernel']!=watch.KERNEL or bundle['dataset']!=watch.DATASET:raise ValueError('Destination mismatch')
    for filename,key in [('Stage1_Mixed.ipynb','notebook_sha256'),('kernel-metadata.json','metadata_sha256')]:
        if hashlib.sha256((work/'kernel'/filename).read_bytes()).hexdigest()!=bundle[key]:raise ValueError('Prepared notebook changed')
    if launch:
        if (work/'approval-required.json').exists():
            approval=json.loads((work/'upload-approval.json').read_text(encoding='utf-8'))
            if approval.get('approved') is not True or approval.get('kernel')!=watch.KERNEL or approval.get('notebook_sha256')!=bundle['notebook_sha256'] or approval.get('metadata_sha256')!=bundle['metadata_sha256']:
                raise RuntimeError('Explicit approval for this notebook and destination required')
        with watch.tick_lock(work/'watch.lock') as acquired:
            if not acquired:raise RuntimeError('Monitor already running')
            with (work/'launch-attempt.json').open('x') as f:
                json.dump(dict(kernel=watch.KERNEL,notebook_sha256=bundle['notebook_sha256'],created_unix=time.time()),f)
            watch.update(status='LAUNCHING',monitor_status='ACTIVE',dataset_reused=True,upload_bytes=bundle['upload_bytes'])
            response=watch.run([watch.KAGGLE,'kernels','push','-p',work/'kernel'],180)
            (work/'push.log').write_text(response,encoding='utf-8')
            match=re.search(r'Kernel version (\d+) successfully pushed',response)
            if not match:raise RuntimeError('Unrecognized push response; inspect remote, do not repeat push')
            watch.update(status='SUBMITTED',kernel_version=int(match[1]),gpu_started=False)
    elif not (work/'launch-attempt.json').exists():raise RuntimeError('No prior launch to monitor')
    watch.main(launch=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    try:main(a.launch)
    except Exception as error:
        watch.update(monitor_status='FAILED',last_error=repr(error));raise
