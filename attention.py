"""attention.py — Section 7: Attention Analysis.

Q10: For each (layer, head), visualise the average attention matrix over
     100 correct Dyck strings. Identify and quantify bracket-matching heads.

Q11: Compare attention patterns on correct vs corrupted strings.
     Describe what happens at and around the corruption site.

Usage:
    python attention.py --checkpoint models/best.pt --split test_id.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

from data import encode, build_pad_mask, VOCAB, PAD_ID, CLS_ID, SEP_ID
from model import DyckTransformer, Config


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--split",       required=True, help="JSONL file to sample from.")
    p.add_argument("--n-correct",   type=int, default=100,
                   help="Number of correct strings for Q10.")
    p.add_argument("--n-corrupted", type=int, default=5,
                   help="Number of corrupted strings for Q11.")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--out-dir",     default="figures")
    p.add_argument("--q11",         action="store_true",
                   help="Also run Q11 (correct vs corrupted attention comparison).")
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(path: str, device: torch.device) -> DyckTransformer:
    ckpt  = torch.load(path, map_location=device)
    config = Config(**ckpt["config"])
    model = DyckTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_record(rec: dict, device: torch.device):
    """Return (input_ids, pad_mask, seq_len) tensors for one record."""
    ids      = encode(rec["actual"])
    pad_mask = build_pad_mask(ids)
    input_ids = torch.tensor(ids,      dtype=torch.long).unsqueeze(0).to(device)
    pad_mask  = torch.tensor(pad_mask, dtype=torch.bool).unsqueeze(0).to(device)
    seq_len   = len(rec["actual"]) + 2  # CLS + tokens + SEP
    return input_ids, pad_mask, seq_len


# ── Matched bracket pairs ─────────────────────────────────────────────────────

OPENER_TO_CLOSER = {"(": ")", "[": "]"}
CLOSER_TO_OPENER = {v: k for k, v in OPENER_TO_CLOSER.items()}

def get_matched_pairs(s: str) -> list[tuple[int, int]]:
    """Return list of (opener_enc_pos, closer_enc_pos) for every matched pair.

    Positions are in the *encoded* sequence (shifted by 1 for CLS).
    """
    stack  = []   # stack of encoded positions of unmatched openers
    pairs  = []
    for i, c in enumerate(s):
        enc_i = i + 1             # shift by 1 for CLS
        if c in OPENER_TO_CLOSER:
            stack.append(enc_i)
        elif c in CLOSER_TO_OPENER:
            if stack:
                opener_pos = stack.pop()
                pairs.append((opener_pos, enc_i))
    return pairs


# ── Attention extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def get_attentions(model, input_ids, pad_mask):
    """Return list of attention tensors, one per layer.
    Each tensor: (n_heads, L, L) — squeezed from (1, n_heads, L, L).
    """
    _, _, _, attentions = model(input_ids, pad_mask, return_internals=True)
    return [a.squeeze(0).cpu() for a in attentions]   # list of (H, L, L)


# ── Q10: average attention heatmaps ──────────────────────────────────────────

def q10(model, records, device, out_dir: Path):
    print(f"\n── Q10: Attention heatmaps + matcher heads ──────────────────")

    n_layers = model.config.n_layer
    n_heads  = model.config.n_head
    MAX_LEN  = model.config.max_len

    # Accumulators: sum of attention matrices and count, per (layer, head)
    attn_sum   = np.zeros((n_layers, n_heads, MAX_LEN, MAX_LEN))
    attn_count = 0

    # Per-head matcher scores: lists of (alpha_i2j, alpha_j2i) values
    # across all matched pairs across all strings
    matcher_scores = defaultdict(lambda: {"i2j": [], "j2i": []})

    for rec in records:
        input_ids, pad_mask, seq_len = encode_record(rec, device)
        attentions = get_attentions(model, input_ids, pad_mask)

        # Accumulate average attention (only over real token positions)
        for layer_idx, attn in enumerate(attentions):
            # attn: (H, L, L)
            attn_sum[layer_idx, :, :seq_len, :seq_len] += (
                attn[:, :seq_len, :seq_len].numpy()
            )
        attn_count += 1

        # Matched pairs for this string
        pairs = get_matched_pairs(rec["original"])

        for layer_idx, attn in enumerate(attentions):
            for head_idx in range(n_heads):
                A = attn[head_idx]          # (L, L)
                for (i, j) in pairs:
                    matcher_scores[(layer_idx, head_idx)]["i2j"].append(
                        A[i, j].item()      # opener → closer
                    )
                    matcher_scores[(layer_idx, head_idx)]["j2i"].append(
                        A[j, i].item()      # closer → opener
                    )

    attn_avg = attn_sum / attn_count        # (n_layers, n_heads, MAX_LEN, MAX_LEN)

    # ── Heatmap grid ──────────────────────────────────────────────────────
    # Show only the active region (trim to median seq length for readability)
    median_len = int(np.median([
        len(r["original"]) + 2 for r in records
    ]))
    trim = median_len

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(3 * n_heads, 3 * n_layers),
        squeeze=False,
    )
    for l in range(n_layers):
        for h in range(n_heads):
            ax  = axes[l][h]
            mat = attn_avg[l, h, :trim, :trim]
            im  = ax.imshow(mat, vmin=0, vmax=mat.max(), cmap="Blues", aspect="auto")
            ax.set_title(f"L{l+1}·H{h+1}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Mean attention per (layer, head) — 100 correct Dyck strings",
                 fontsize=10)
    fig.tight_layout()
    path = out_dir / "q10_attention_heatmaps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  heatmap grid saved → {path}")

    # ── Matcher head report ───────────────────────────────────────────────
    # Chance baseline: for a sequence of length L with P pairs,
    # expected attention to any single non-self position ≈ 1/(L-1).
    # We use a fixed conservative baseline of 1/trim.
    baseline = 1.0 / trim

    print(f"\n  Matcher head scores  (chance baseline ≈ {baseline:.4f})")
    print(f"  {'Head':<12} {'α(i→j) mean':>14} {'α(i→j) std':>12} "
          f"{'α(j→i) mean':>14} {'α(j→i) std':>12} {'ratio':>8}")
    print(f"  {'─'*76}")

    matcher_heads = []
    for l in range(n_layers):
        for h in range(n_heads):
            scores = matcher_scores[(l, h)]
            i2j    = np.array(scores["i2j"])
            j2i    = np.array(scores["j2i"])
            mean_i2j, std_i2j = i2j.mean(), i2j.std()
            mean_j2i, std_j2i = j2i.mean(), j2i.std()
            ratio = ((mean_i2j + mean_j2i) / 2) / baseline

            flag = " ★" if ratio >= 1.3 else ""   # flag heads elevated above baseline
            print(f"  L{l+1}·H{h+1:<8} {mean_i2j:>14.4f} {std_i2j:>12.4f} "
                  f"{mean_j2i:>14.4f} {std_j2i:>12.4f} {ratio:>7.2f}×{flag}")

            matcher_heads.append((l, h, ratio))

    # Sort by ratio descending; flag top heads for Q11
    matcher_heads.sort(key=lambda x: -x[2])
    print(f"\n  Top 3 heads by matcher ratio:")
    for (l, h, r) in matcher_heads[:3]:
        print(f"    L{l+1}·H{h+1}  ratio={r:.2f}×")

    return attn_avg, matcher_heads, matcher_scores


# ── Q11: correct vs corrupted attention ───────────────────────────────────────

def q11(model, correct_records, corrupted_records, matcher_heads, device, out_dir: Path):
    print(f"\n── Q11: Correct vs corrupted attention ──────────────────────")

    # matcher_heads is always sorted by ratio; use the top one
    best_layer, best_head, best_ratio = matcher_heads[0]
    print(f"  Using head L{best_layer+1}·H{best_head+1} (ratio={best_ratio:.2f}×)")

    n_correct   = min(3, len(correct_records))
    n_corrupted = min(3, len(corrupted_records))
    total       = n_correct + n_corrupted
    fig, axes   = plt.subplots(2, max(n_correct, n_corrupted),
                               figsize=(4 * max(n_correct, n_corrupted), 8),
                               squeeze=False)

    def plot_attn(ax, rec, is_corrupted):
        input_ids, pad_mask, seq_len = encode_record(rec, device)
        attentions = get_attentions(model, input_ids, pad_mask)
        A = attentions[best_layer][best_head, :seq_len, :seq_len].numpy()

        ax.imshow(A, vmin=0, vmax=A.max(), cmap="Blues", aspect="auto")

        # Annotate matched pairs on correct strings
        if not is_corrupted:
            pairs = get_matched_pairs(rec["original"])
            for (i, j) in pairs:
                if i < seq_len and j < seq_len:
                    ax.plot(j, i, "g+", markersize=6, markeredgewidth=1.2)
                    ax.plot(i, j, "g+", markersize=6, markeredgewidth=1.2)

        # Annotate error position on corrupted strings
        if is_corrupted and rec.get("error_position") is not None:
            ep = rec["error_position"] + 1   # shift for CLS
            ax.axvline(ep, color="red", linewidth=1.2, linestyle="--", alpha=0.7)
            ax.axhline(ep, color="red", linewidth=1.2, linestyle="--", alpha=0.7)

        kind  = f"CORRUPTED [{rec['error_type']}]" if is_corrupted else "correct"
        short = rec["actual"][:20] + ("…" if len(rec["actual"]) > 20 else "")
        ax.set_title(f"{kind}\n\"{short}\"", fontsize=7)
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")

    for i, rec in enumerate(correct_records[:n_correct]):
        plot_attn(axes[0][i], rec, is_corrupted=False)
    for i in range(n_correct, axes.shape[1]):
        axes[0][i].axis("off")

    for i, rec in enumerate(corrupted_records[:n_corrupted]):
        plot_attn(axes[1][i], rec, is_corrupted=True)
    for i in range(n_corrupted, axes.shape[1]):
        axes[1][i].axis("off")

    axes[0][0].set_ylabel("CORRECT\nquery position", fontsize=8)
    axes[1][0].set_ylabel("CORRUPTED\nquery position", fontsize=8)

    fig.suptitle(
        f"Q11: Head L{best_layer+1}·H{best_head+1} — correct (top) vs corrupted (bottom)\n"
        f"Green + = matched pairs · Red dashed = error position",
        fontsize=9,
    )
    fig.tight_layout()
    path = out_dir / "q11_correct_vs_corrupted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  plot saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    n_layers = model.config.n_layer
    n_heads  = model.config.n_head
    print(f"model: {n_layers} layers, {n_heads} heads")

    # Load and split records
    with open(args.split) as f:
        all_records = [json.loads(l) for l in f]

    correct_records   = [r for r in all_records if not r["is_corrupted"]]
    corrupted_records = [r for r in all_records if r["is_corrupted"]]

    random.shuffle(correct_records)
    random.shuffle(corrupted_records)

    q10_records = correct_records[:args.n_correct]
    print(f"Q10: using {len(q10_records)} correct strings")

    attn_avg, matcher_heads, matcher_scores = q10(
        model, q10_records, device, out_dir
    )

    if args.q11:
        q11_correct   = correct_records[:args.n_corrupted]
        q11_corrupted = corrupted_records[:args.n_corrupted]
        q11(model, q11_correct, q11_corrupted, matcher_heads, device, out_dir)


if __name__ == "__main__":
    main()
