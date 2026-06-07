#!/usr/bin/env python3
"""
Download Amazon Review datasets.
Primary URLs: Amazon 2023 (McAuley lab)
Fallback: Amazon 5-core datasets
"""

import os
import logging
import urllib.request
import urllib.error
from pathlib import Path

BASE    = Path(__file__).parent
RAW_DIR = BASE / 'data' / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(BASE / 'run.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

DATASETS = {
    'beauty': {
        'primary': 'https://jmcauley.ucsd.edu/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz',
        'filename': 'All_Beauty.jsonl.gz',
        'fallbacks': [
            'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz',
        ],
    },
    'sports': {
        'primary': 'https://jmcauley.ucsd.edu/data/amazon_2023/raw/review_categories/Sports_and_Outdoors.jsonl.gz',
        'filename': 'Sports_and_Outdoors.jsonl.gz',
        'fallbacks': [
            'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz',
        ],
    },
    'scientific': {
        'primary': 'https://jmcauley.ucsd.edu/data/amazon_2023/raw/review_categories/Industrial_and_Scientific.jsonl.gz',
        'filename': 'Industrial_and_Scientific.jsonl.gz',
        'fallbacks': [
            'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Industrial_and_Scientific_5.json.gz',
        ],
    },
}


def download_file(url, dest):
    log.info(f"  downloading {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as resp, \
             open(dest, 'wb') as out:
            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            chunk = 1 << 20  # 1 MB
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                out.write(data)
                downloaded += len(data)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded/(1<<20):.1f}/{total/(1<<20):.1f} MB ({pct:.0f}%)",
                          end='', flush=True)
        print()
        log.info(f"  saved to {dest}")
        return True
    except Exception as e:
        log.warning(f"  failed: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    for name, info in DATASETS.items():
        dest = RAW_DIR / info['filename']
        if dest.exists() and dest.stat().st_size > 1000:
            log.info(f"{name}: already downloaded")
            continue

        log.info(f"=== downloading {name} ===")
        success = download_file(info['primary'], dest)

        if not success:
            for fallback_url in info.get('fallbacks', []):
                success = download_file(fallback_url, dest)
                if success:
                    break

        if not success:
            log.error(f"failed to download {name} -- will skip preprocessing")


if __name__ == '__main__':
    main()
