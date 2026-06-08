# DATA VALIDATION REPORT

Date: 2026-06-04. Repo: `/mnt/disk1/backup_user/minh.ntn/ETEGRec`

## TL;DR
The previously-extracted `dataset/{scientific,game,instrument}/` folders were **truncated to ~4 MB each** (corrupt). I re-extracted all three zips. **All files now load correctly.** The truncated copies were moved to `dataset/_truncated_backup_20260604_180333/` (not deleted).

## Zips vs. extracted (the truncation bug)
| File | Size in zip | Old extracted | After re-extract |
|---|---|---|---|
| game/game.train.jsonl | 104.7 MB | 4.0 MB (cut mid-line) | 100 MB ✅ |
| game/game_emb_256.npy | 26.2 MB | 4.0 MB (BROKEN) | 25 MB ✅ |
| scientific/scientific.train.jsonl | 46.4 MB | 4.0 MB | 45 MB ✅ |
| scientific/scientific_emb_256.npy | 26.5 MB | 3.8 MB (BROKEN) | 25.2 MB ✅ |
| instrument/instrument.train.jsonl | 65.7 MB | 4.0 MB | 63 MB ✅ |
| instrument/instrument_emb_256.npy | 25.2 MB | 4.0 MB (BROKEN) | 24 MB ✅ |

Every old `.npy` raised `ValueError: Failed to read all data for array` (header declared the full shape, body was cut). **Fixed.**

## `.npy` load status (after re-extract)
| File | shape | dtype | status |
|---|---|---|---|
| scientific/scientific_emb_256.npy | (25848, 256) | float32 | OK |
| game/game_emb_256.npy | (25612, 256) | float32 | OK |
| instrument/instrument_emb_256.npy | (24587, 256) | float32 | OK |
| scientific/item_features.npy | (25849, 256) | float32 | OK — **regenerated** |

## Note on `item_features.npy`
**`item_features.npy` is NOT inside `scientific.zip`** (the zip has 6 files; no feature file). It is produced by `generate_features.py`, which prepends one zero padding row to `scientific_emb_256.npy` → `(25849, 256)`. I ran it; row 0 is all-zero (the PAD slot). Only `scientific` has this file / a `generate_features.py`; `game` and `instrument` do not (and their configs/run scripts don't use `--use_features`).

## Other files (all intact)
- `*.train/valid/test.jsonl`, `*.emb_map.json`, `256-512-256-128.rqvae.pth` — present and full for all 3 datasets.
- `python -m compileall` on all core modules: **OK**.
