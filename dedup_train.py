import json
import sys

def load_keys(path, fields=("original", "actual")):
    keys = {f: set() for f in fields}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            for k in fields:
                keys[k].add(r[k])
    return keys

def filter_train(train_path, dev_path, test_path, out_path):
    dev_keys  = load_keys(dev_path)
    test_keys = load_keys(test_path)

    removed = 0
    kept = 0
    with open(train_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            r = json.loads(line)
            if r["original"] in dev_keys["original"] or r["original"] in test_keys["original"]:
                removed += 1
                continue
            if r["actual"] in dev_keys["actual"] or r["actual"] in test_keys["actual"]:
                removed += 1
                continue
            fout.write(line)
            kept += 1

    print(f"kept: {kept}, removed: {removed}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("usage: python dedup_train.py train.jsonl dev.jsonl test.jsonl train_dedup.jsonl")
        sys.exit(1)
    filter_train(*sys.argv[1:])