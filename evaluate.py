"""evaluate.py — evaluation for Q4, Q5, Q7.

Q4: detection accuracy, macro-F1, confusion matrix broken down by error type.
Q5: correction token-level accuracy (all positions / non-OK positions),
    exact-match accuracy after applying predicted edits.
Q7: OOD detection accuracy stratified by nesting depth and sequence length.

Usage:
    # Q4 + Q5 on in-distribution test set
    python evaluate.py --split test_id.jsonl --checkpoint models/best.pt

    # Q7 on OOD test set
    python evaluate.py --split test_ood.jsonl --checkpoint models/best.pt --ood
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from data import (
    make_loader, IGNORE, LBL_OK, LBL_DELETE,
    LBL_INSERT_OFFSET, LBL_REPLACE_OFFSET, BRACKETS, VOCAB,
)
from model import Config, DyckTransformer


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",      required=True,  help="Path to JSONL split.")
    p.add_argument("--checkpoint", required=True,  help="Path to best.pt.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir",    default="figures")
    p.add_argument("--ood",        action="store_true",
                   help="Run OOD stratified analysis (Q7) instead of Q4+Q5.")
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> DyckTransformer:
    ckpt  = torch.load(checkpoint_path, map_location=device)
    config = Config(**ckpt["config"])
    model = DyckTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ── Dyck validity checker (PDA baseline for Q8) ───────────────────────────────

OPENER_TO_CLOSER = {"(": ")", "[": "]"}
CLOSER_TO_OPENER = {v: k for k, v in OPENER_TO_CLOSER.items()}

def is_valid_dyck(s: str) -> bool:
    stack = []
    for c in s:
        if c in OPENER_TO_CLOSER:
            stack.append(c)
        elif c in CLOSER_TO_OPENER:
            if not stack or stack[-1] != CLOSER_TO_OPENER[c]:
                return False
            stack.pop()
    return len(stack) == 0


# ── Correction: decode predicted labels back to a string edit ─────────────────

def apply_correction(actual: str, token_labels: list[int]) -> str:
    """Apply per-token predicted labels to `actual` and return corrected string.

    token_labels is aligned with encode(actual):
      index 0   = CLS  (always ignored)
      index 1…n = actual[0]…actual[n-1]
      index n+1 = SEP  (INSERT slot for end-of-string deletions)
    """
    result = []
    n = len(actual)

    for enc_pos in range(1, n + 2):        # 1-indexed, up to and including SEP slot
        lbl = token_labels[enc_pos]
        char_pos = enc_pos - 1             # position in `actual` (n = SEP slot)

        if enc_pos <= n:
            original_char = actual[char_pos]
        else:
            original_char = None           # SEP slot has no original char

        if lbl == LBL_OK:
            if original_char is not None:
                result.append(original_char)

        elif lbl == LBL_DELETE:
            pass                           # drop this token

        elif LBL_INSERT_OFFSET <= lbl < LBL_REPLACE_OFFSET:
            tok = BRACKETS[lbl - LBL_INSERT_OFFSET]
            result.append(tok)             # insert before this position
            if original_char is not None:
                result.append(original_char)

        elif lbl >= LBL_REPLACE_OFFSET:
            tok = BRACKETS[lbl - LBL_REPLACE_OFFSET]
            result.append(tok)             # replace

    return "".join(result)


# ── Inference loop ────────────────────────────────────────────────────────────

def run_inference(model, loader, device):
    """Returns flat lists of per-example results."""
    det_preds, det_labels = [], []
    cor_preds_all, cor_labels_all = [], []
    meta_all = []

    with torch.no_grad():
        for batch in loader:
            input_ids     = batch["input_ids"].to(device)
            pad_mask      = batch["pad_mask"].to(device)
            detect_lbls   = batch["detect_labels"]
            correct_lbls  = batch["correct_labels"]

            det_logits, cor_logits = model(input_ids, pad_mask)

            det_preds.extend(det_logits.argmax(-1).cpu().tolist())
            det_labels.extend(detect_lbls.tolist())

            # (B, L) predicted correction labels
            cor_preds_all.extend(cor_logits.argmax(-1).cpu().tolist())
            cor_labels_all.extend(correct_lbls.tolist())

            meta_all.extend(batch["meta"])

    return det_preds, det_labels, cor_preds_all, cor_labels_all, meta_all


# ── Q4: detection ─────────────────────────────────────────────────────────────

ERROR_ORDER = ["OK", "e1", "e2", "e3", "e4"]

def q4_detection(det_preds, det_labels, meta_all, out_dir: Path):
    acc = sum(p == l for p, l in zip(det_preds, det_labels)) / len(det_labels)
    f1  = f1_score(det_labels, det_preds, average="macro", zero_division=0)
    print(f"\n── Q4: Detection ────────────────────────")
    print(f"  accuracy  : {acc:.4f}")
    print(f"  macro-F1  : {f1:.4f}")

    # Per-error-type breakdown
    # Group: TP (corrupted, predicted corrupted), FN (corrupted, predicted OK)
    #        TN (OK, predicted OK),               FP (OK, predicted corrupted)
    counts = defaultdict(lambda: {"correct": 0, "wrong": 0, "total": 0})
    for pred, lbl, meta in zip(det_preds, det_labels, meta_all):
        etype = meta["error_type"] if meta["error_type"] is not None else "OK"
        counts[etype]["total"]   += 1
        counts[etype]["correct"] += int(pred == lbl)
        counts[etype]["wrong"]   += int(pred != lbl)

    print(f"\n  {'type':<6} {'total':>6} {'correct':>8} {'wrong':>7} {'acc':>7}")
    print(f"  {'─'*40}")
    for etype in ERROR_ORDER:
        if etype not in counts:
            continue
        c = counts[etype]
        print(f"  {etype:<6} {c['total']:>6} {c['correct']:>8} {c['wrong']:>7} "
              f"{c['correct']/c['total']:>7.4f}")

    # Confusion matrix plot (rows = true error type, cols = predicted OK/error)
    fig, ax = plt.subplots(figsize=(5, 4))
    row_labels = [e for e in ERROR_ORDER if e in counts]
    matrix = np.array([
        [counts[e]["correct"] if (e == "OK") else counts[e]["wrong"],
         counts[e]["wrong"]   if (e == "OK") else counts[e]["correct"]]
        for e in row_labels
    ])
    # columns: predicted OK, predicted corrupted
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pred: OK", "pred: corrupted"])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Detection confusion by error type")
    plt.colorbar(im, ax=ax)
    for i in range(len(row_labels)):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "black",
                    fontsize=9)
    fig.tight_layout()
    path = out_dir / "q4_confusion.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  confusion matrix saved → {path}")


# ── Q5: correction ────────────────────────────────────────────────────────────

def q5_correction(cor_preds_all, cor_labels_all, meta_all, out_dir: Path):
    print(f"\n── Q5: Correction ───────────────────────")

    # Token-level accuracy (all positions)
    correct_all = total_all = 0
    # Token-level accuracy (non-OK gold positions only)
    correct_non_ok = total_non_ok = 0

    for preds_seq, labels_seq in zip(cor_preds_all, cor_labels_all):
        for p, l in zip(preds_seq, labels_seq):
            if l == IGNORE:
                continue
            total_all   += 1
            correct_all += int(p == l)
            if l != LBL_OK:
                total_non_ok   += 1
                correct_non_ok += int(p == l)

    acc_all    = correct_all    / total_all    if total_all    > 0 else 0.0
    acc_non_ok = correct_non_ok / total_non_ok if total_non_ok > 0 else 0.0
    print(f"  token acc (all positions)     : {acc_all:.4f}")
    print(f"  token acc (non-OK positions)  : {acc_non_ok:.4f}")

    # Exact-match accuracy: apply edits and check validity
    exact_total = exact_match = 0
    for preds_seq, meta in zip(cor_preds_all, meta_all):
        if not meta["error_type"]:
            continue                       # skip clean examples
        exact_total += 1
        corrected = apply_correction(meta["actual"], preds_seq)
        if is_valid_dyck(corrected):
            exact_match += 1

    exact_acc = exact_match / exact_total if exact_total > 0 else 0.0
    print(f"  exact-match acc (corrupted)   : {exact_acc:.4f}  "
          f"({exact_match}/{exact_total})")


# ── Q7: OOD stratified ────────────────────────────────────────────────────────

def q7_ood(det_preds, det_labels, meta_all, out_dir: Path):
    print(f"\n── Q7: OOD Generalisation ───────────────")

    by_depth  = defaultdict(lambda: {"correct": 0, "total": 0})
    by_length = defaultdict(lambda: {"correct": 0, "total": 0})

    for pred, lbl, meta in zip(det_preds, det_labels, meta_all):
        d = meta["depth"]
        l = meta["length"]
        bucket = (l // 10) * 10            # bucket lengths: 40, 50, 60, 70, 80

        by_depth[d]["total"]    += 1
        by_depth[d]["correct"]  += int(pred == lbl)
        by_length[bucket]["total"]   += 1
        by_length[bucket]["correct"] += int(pred == lbl)

    # Print tables
    print(f"\n  by depth:")
    print(f"  {'depth':>6} {'total':>6} {'acc':>7}")
    for d in sorted(by_depth):
        c = by_depth[d]
        print(f"  {d:>6} {c['total']:>6} {c['correct']/c['total']:>7.4f}")

    print(f"\n  by length bucket:")
    print(f"  {'length':>8} {'total':>6} {'acc':>7}")
    for b in sorted(by_length):
        c = by_length[b]
        print(f"  {b:>8} {c['total']:>6} {c['correct']/c['total']:>7.4f}")

    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    depths = sorted(by_depth)
    accs_d = [by_depth[d]["correct"] / by_depth[d]["total"] for d in depths]
    ax1.plot(depths, accs_d, marker="o")
    ax1.set_xlabel("nesting depth n")
    ax1.set_ylabel("accuracy")
    ax1.set_title("OOD accuracy vs nesting depth")
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    buckets = sorted(by_length)
    accs_l  = [by_length[b]["correct"] / by_length[b]["total"] for b in buckets]
    ax2.plot(buckets, accs_l, marker="o", color="tab:orange")
    ax2.set_xlabel("sequence length (bucket)")
    ax2.set_ylabel("accuracy")
    ax2.set_title("OOD accuracy vs sequence length")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    fig.tight_layout()
    path = out_dir / "q7_ood.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  plots saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model  = load_model(args.checkpoint, device)
    loader = make_loader(args.split, batch_size=args.batch_size, shuffle=False)

    det_preds, det_labels, cor_preds, cor_labels, meta = run_inference(
        model, loader, device
    )

    if args.ood:
        q7_ood(det_preds, det_labels, meta, out_dir)
    else:
        q4_detection(det_preds, det_labels, meta, out_dir)
        q5_correction(cor_preds, cor_labels, meta, out_dir)


if __name__ == "__main__":
    main()
