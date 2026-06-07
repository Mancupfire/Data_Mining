"""
Convert ETEGRec .jsonl splits to baseline plain-text format.

ETEGRec train.jsonl is sliding-window (multiple rows per user).
The full training sequence per user lives in valid.jsonl's inter_history
field -- everything before the validation target.

Input:  data/etegrec/{ds}/{ds}.{train,valid,test}.jsonl
Output: data/{ds}/{train,valid,test,stats}.txt
"""
import json
from pathlib import Path

DATASETS = ["game", "instrument", "scientific"]
BASE     = Path(__file__).parent
SRC_ROOT = BASE / "data" / "etegrec"
DST_ROOT = BASE / "data"


for ds in DATASETS:
    src = SRC_ROOT / ds
    dst = DST_ROOT / ds
    dst.mkdir(parents=True, exist_ok=True)

    user2id, item2id = {}, {}

    def get_uid(u):
        if u not in user2id:
            user2id[u] = len(user2id) + 1
        return user2id[u]

    def get_iid(i):
        if i not in item2id:
            item2id[i] = len(item2id) + 1
        return item2id[i]

    # Build vocab in stable order: train -> valid -> test
    for split in ["train", "valid", "test"]:
        with open(src / f"{ds}.{split}.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                get_uid(rec["user_id"])
                for iid in rec.get("inter_history", []):
                    get_iid(iid)
                if rec.get("target_id"):
                    get_iid(rec["target_id"])

    # train.txt: use valid.jsonl's inter_history as the full training sequence.
    # train.jsonl is sliding-window (multiple rows/user); valid.jsonl has exactly
    # one row per user and inter_history = complete train seq.
    with open(src / f"{ds}.valid.jsonl") as fin, \
         open(dst / "train.txt", "w") as fout:
        for line in fin:
            rec = json.loads(line)
            uid = get_uid(rec["user_id"])
            hist = [str(get_iid(i)) for i in rec["inter_history"]]
            if hist:
                fout.write(f"{uid} {' '.join(hist)}\n")

    # valid.txt: one target per user
    with open(src / f"{ds}.valid.jsonl") as fin, \
         open(dst / "valid.txt", "w") as fout:
        for line in fin:
            rec = json.loads(line)
            uid = get_uid(rec["user_id"])
            tgt = get_iid(rec["target_id"])
            fout.write(f"{uid} {tgt}\n")

    # test.txt: one target per user
    with open(src / f"{ds}.test.jsonl") as fin, \
         open(dst / "test.txt", "w") as fout:
        for line in fin:
            rec = json.loads(line)
            uid = get_uid(rec["user_id"])
            tgt = get_iid(rec["target_id"])
            fout.write(f"{uid} {tgt}\n")

    # stats.txt
    total_interactions = 0
    with open(src / f"{ds}.valid.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            total_interactions += len(rec["inter_history"]) + 2  # +valid +test

    with open(dst / "stats.txt", "w") as f:
        f.write(f"num_users {len(user2id)}\n")
        f.write(f"num_items {len(item2id)}\n")
        f.write(f"num_interactions {total_interactions}\n")

    print(f"{ds}: {len(user2id)} users, {len(item2id)} items -> {dst}")

print("done")
