import json
import torch
from torch.utils.data import Dataset, DataLoader

# ── Vocabulary ────────────────────────────────────────────────────────────────

VOCAB    = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "(": 3, ")": 4, "[": 5, "]": 6}
INV_VOCAB = {v: k for k, v in VOCAB.items()}
PAD_ID, CLS_ID, SEP_ID = VOCAB["[PAD]"], VOCAB["[CLS]"], VOCAB["[SEP]"]

BRACKETS = ["(", ")", "[", "]"]          # fixed order for label offsets
MAX_LEN  = 80
IGNORE   = -100                          # CrossEntropyLoss ignore_index

# ── Correction label constants ────────────────────────────────────────────────

LBL_OK             = 0
LBL_DELETE         = 1
LBL_INSERT_OFFSET  = 2   # INSERT_( = 2, INSERT_) = 3, INSERT_[ = 4, INSERT_] = 5
LBL_REPLACE_OFFSET = 6   # REPLACE_( = 6, REPLACE_) = 7, REPLACE_[ = 8, REPLACE_] = 9
N_CORRECT_LABELS   = 10

# ── Tokenizer ─────────────────────────────────────────────────────────────────

def encode(s: str, max_len: int = MAX_LEN) -> list[int]:
    """[CLS] + tokens + [SEP], padded to max_len."""
    ids = [CLS_ID] + [VOCAB[c] for c in s] + [SEP_ID]
    ids = ids[:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


def decode(ids: list[int]) -> str:
    """Inverse of encode, skipping special tokens."""
    return "".join(
        INV_VOCAB[i] for i in ids
        if i not in (PAD_ID, CLS_ID, SEP_ID)
    )


def build_pad_mask(ids: list[int]) -> list[bool]:
    """True where the token is PAD (to be masked out in attention)."""
    return [i == PAD_ID for i in ids]

# ── Correction labels ─────────────────────────────────────────────────────────

def build_correction_labels(record: dict, max_len: int = MAX_LEN) -> list[int]:
    """Per-position correction label aligned with encode(record['actual']).

    Position 0   = CLS  → IGNORE
    Positions 1…n = actual tokens → OK by default, one position overridden
    Position n+1 = SEP  → OK  (INSERT slot for end-of-string E1)
    Positions after SEP  → IGNORE
    """
    actual     = record["actual"]
    error_type = record["error_type"]

    labels = [IGNORE] * max_len

    # Label every real token + SEP slot as OK
    last = min(1 + len(actual) + 1, max_len)   # +1 CLS, +1 SEP
    for t in range(1, last):
        labels[t] = LBL_OK

    if error_type is None:
        return labels

    pos     = record["error_position"]         # position in `actual` string
    err_tok = record["error_token"]
    enc_pos = 1 + pos                          # shift by 1 for CLS

    if enc_pos >= max_len:
        return labels  # error position is truncated out, can't label it

    if error_type == "e1":
        # Closer was deleted — we need to INSERT it back
        labels[enc_pos] = LBL_INSERT_OFFSET + BRACKETS.index(err_tok)

    elif error_type in ("e2", "e4"):
        # A token was inserted — DELETE it
        labels[enc_pos] = LBL_DELETE

    elif error_type == "e3":
        # Closer was replaced — REPLACE with the original token
        original    = record["original"]
        correct_tok = original[pos]
        labels[enc_pos] = LBL_REPLACE_OFFSET + BRACKETS.index(correct_tok)

    return labels

# ── Dataset ───────────────────────────────────────────────────────────────────

class DyckDataset(Dataset):
    def __init__(self, path: str, max_len: int = MAX_LEN):
        self.max_len = max_len
        with open(path) as f:
            self.records = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        ids      = encode(rec["actual"], self.max_len)
        pad_mask = build_pad_mask(ids)
        detect_label  = int(rec["is_corrupted"])
        correct_labels = build_correction_labels(rec, self.max_len)

        return {
            "input_ids":      torch.tensor(ids,            dtype=torch.long),
            "pad_mask":       torch.tensor(pad_mask,       dtype=torch.bool),
            "detect_label":   torch.tensor(detect_label,   dtype=torch.long),
            "correct_labels": torch.tensor(correct_labels, dtype=torch.long),
            "meta": {
                "id":             rec["id"],
                "error_type":     rec["error_type"],
                "error_position": rec["error_position"],
                "depth":          rec["depth"],
                "length":         rec["length"],
                "original":       rec["original"],
                "actual":         rec["actual"],
            },
        }

# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch: list[dict]) -> dict:
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "pad_mask":       torch.stack([b["pad_mask"]        for b in batch]),
        "detect_labels":  torch.stack([b["detect_label"]    for b in batch]),
        "correct_labels": torch.stack([b["correct_labels"]  for b in batch]),
        "meta":           [b["meta"] for b in batch],
    }

# ── Convenience loader factory ────────────────────────────────────────────────

def make_loader(
    path: str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    max_len: int = MAX_LEN,
) -> DataLoader:
    ds = DyckDataset(path, max_len=max_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )