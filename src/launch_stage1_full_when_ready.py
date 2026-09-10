"""Start the explicitly approved full run once its dataset is ready; never retry a push."""
import json
import subprocess
import time
from pathlib import Path
import watch_stage1_full as watch


def main():
    marker = watch.BUNDLE / 'launch-attempt.json'
    if marker.exists():
        raise RuntimeError('Launch already attempted; inspect remote status before retrying')
    watch.update(status='WAITING_FOR_DATASET', monitor_status='WAITING', user_approved=True)
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        try:
            state = watch.run([watch.KAGGLE, 'datasets', 'status', 'biadis/crashintent-stage1-full-v1-assets'])
        except RuntimeError as error:
            watch.update(status='WAITING_FOR_DATASET', last_dataset_response=str(error)[-500:])
        else:
            if state.strip().lower() == 'ready':
                break
            watch.update(status='WAITING_FOR_DATASET', last_dataset_response=state.strip())
        time.sleep(30)
    else:
        raise TimeoutError('Dataset did not become ready within one hour; GPU not started')
    # Exclusive marker prevents a duplicate push even if a response is lost.
    with marker.open('x', encoding='utf-8') as output:
        json.dump({'kernel': watch.KERNEL, 'dataset_ready': True}, output)
    watch.update(status='LAUNCHING', uploaded=True)
    response = watch.run([watch.KAGGLE, 'kernels', 'push', '-p', watch.BUNDLE / 'kernel'], timeout=180)
    (watch.BUNDLE / 'push.log').write_text(response, encoding='utf-8')
    watch.update(status='SUBMITTED', push_response=response.strip())
    watch.main()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        watch.update(monitor_status='FAILED', last_error=repr(error))
        raise
