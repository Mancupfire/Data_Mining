#!/usr/bin/env python3
"""
Master pipeline: train all models on all datasets, then summarize.
Runs sequentially to avoid OOM on CPU.
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path

BASE   = Path(__file__).parent
LOG    = BASE / 'run.log'
PYTHON = BASE / 'venv' / 'bin' / 'python'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def run(cmd, **kwargs):
    log.info(f"running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=BASE, **kwargs)
    return result.returncode == 0


def dataset_ready(ds):
    return (BASE / 'data' / ds / 'stats.txt').exists()


def has_results(model, ds):
    return (BASE / 'results' / f'{model}_{ds}_results.txt').exists()


def train_and_eval(model, ds, epochs=200, batch=256):
    if has_results(model, ds):
        log.info(f"  already done: {model} {ds} -- skipping")
        return True
    ok = run(f'{PYTHON} train_{model}.py --dataset {ds} --epochs {epochs} --batch_size {batch}')
    if not ok:
        log.error(f"  failed: {model} {ds}")
    return ok


def main():
    log.info("=" * 60)
    log.info("pipeline start")
    log.info("=" * 60)

    # Wait for scientific to be preprocessed if not ready
    if not dataset_ready('scientific'):
        log.info("waiting for scientific dataset...")
        raw = BASE / 'data' / 'raw' / 'Industrial_and_Scientific.jsonl.gz'
        wait_limit = 3600
        elapsed = 0
        while elapsed < wait_limit:
            size = raw.stat().st_size if raw.exists() else 0
            if size > 50 * 1024 * 1024:
                log.info(f"  scientific raw looks ready ({size/(1<<20):.0f} MB), preprocessing...")
                run(f'{PYTHON} preprocess.py --datasets scientific')
                break
            log.info(f"  waiting... (scientific raw: {size/(1<<20):.1f} MB)")
            time.sleep(60)
            elapsed += 60
        else:
            log.warning("timed out waiting for scientific -- skipping")

    datasets = [ds for ds in ['beauty', 'scientific', 'sports'] if dataset_ready(ds)]
    log.info(f"available datasets: {datasets}")

    for ds in datasets:
        train_and_eval('sasrec', ds)

    for ds in datasets:
        train_and_eval('gru4rec', ds)

    run(f'{PYTHON} summarize.py')

    log.info("=" * 60)
    log.info("pipeline done")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
