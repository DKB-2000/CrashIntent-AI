"""Upload the prepared private pilot once, then launch and watch to validation.
Run only after the user approves this payload and new Kaggle destination.
"""
import json,time
from prepare_stage1_robustness import digest
import watch_stage1_quality_trial as watch


def main():
    work=watch.WORK
    approval=json.loads((work/'upload-approval.json').read_text())
    bundle=json.loads((work/'bundle.json').read_text())
    if approval.get('approved') is not True or approval.get('dataset')!=watch.DATASET or approval.get('bundle_sha256')!=bundle['sha256']:
        raise RuntimeError('Specific payload/destination approval required')
    with watch.tick_lock(work/'upload.lock') as acquired:
        if not acquired:raise RuntimeError('Uploader already active')
        if (work/'upload-attempt.json').exists():raise RuntimeError('Upload already attempted; inspect remote state before retry')
        if digest(work/'dataset/quality.bin')!=bundle['sha256']:raise ValueError('Prepared bundle changed')
        metadata=json.loads((work/'dataset/dataset-metadata.json').read_text())
        if metadata['id']!=watch.DATASET:raise ValueError('Unexpected destination')
        with (work/'upload-attempt.json').open('x') as f:json.dump(dict(dataset=watch.DATASET,sha256=bundle['sha256'],bytes=bundle['bytes'],started_unix=time.time()),f)
        watch.update(status='UPLOADING',monitor_status='ACTIVE',dataset=watch.DATASET,kernel=watch.KERNEL,bundle_sha256=bundle['sha256'])
        response=watch.run([watch.KAGGLE,'datasets','create','-p',work/'dataset','--quiet'],1800)
        (work/'upload.log').write_text(response,encoding='utf-8')
        watch.update(status='UPLOADED',uploaded=True)
    watch.main(launch=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        watch.update(monitor_status='FAILED',last_error=repr(error));raise
